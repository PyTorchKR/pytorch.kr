---
layout: blog_detail
title: "FBTriton으로 테이블 배치 임베딩(TBE) 현대화하기"
author: "Meta Team: Daohang Shi, Oleksandr Stashuk, Rupert Wu, Liangbei Xu, Rich Zhu"
ext_author: Junghwan Park (박정환)
category: ["pytorch.org", "translation"]
date: 2026-10-06 12:00:00
org_title: "Modernizing Table Batched Embeddings with FBTriton"
org_link: https://pytorch.org/blog/modernizing-table-batched-embeddings-with-fbtriton/
---

![FBTriton으로 테이블 배치 임베딩(TBE) 현대화하기 대표 이미지 / Modernizing Table Batched Embeddings with FBTriton](/assets/blog/2026-10-06-modernizing-table-batched-embeddings-with-fbtriton/hero.png){:style="width:100%"}

이번 글에서는 테이블 배치 임베딩(Table Batched Embedding, TBE)의 순전파(forward)와 역전파(backward)를 위한 FBTriton 커널 설계를 살펴봅니다. 이 핵심 연산자는 추천 시스템에서 샤딩(sharding)된 수천 개의 GPU에 걸쳐 임베딩 조회(lookup)를 처리합니다. Triton 구현은 이 워크로드에서 기존 CUDA 커널보다 높은 성능을 냅니다. 여기서는 아키텍처 설계를 설명하고, 측정한 성능 향상을 자세히 소개하고, 앞으로의 최적화 기회를 짚어 봅니다.
> This post explores the FBTriton kernel design for Table Batched Embedding (TBE) forward and backward passes. These core operators handle embedding lookups across thousands of sharded GPUs within recommendation systems. Our Triton implementation successfully outperforms the legacy CUDA kernels on these workloads. Here, we explain the architectural design, detail the measured performance gains, and highlight future optimization opportunities.

## 1. TBE란? / What’s TBE

TBE(Table-Batched Embedding) 커널은 GPU 연산 한 번으로 여러 테이블에 걸친 임베딩 조회와 풀링(pooling)을 효율적으로 수행합니다. TBE는 여러 테이블의 임베딩 조회와 풀링을 GPU 실행(launch) 한 번에 묶어, 실행 오버헤드를 줄이고 메모리 효율을 높입니다. 더 일반적인 내용은 [여기](https://medium.com/better-ml/table-manners-batching-embeddings-for-large-scale-recsys-training-4b5a856cb913)에서 읽을 수 있습니다.
> TBE (Table-Batched Embedding) kernel efficiently performs embedding lookups and pooling across many tables in one GPU operation. TBE combines embedding lookup and pooling for many tables in a single GPU launch, reducing launch overhead and improving memory efficiency. You can read more general info [here](https://medium.com/better-ml/table-manners-batching-embeddings-for-large-scale-recsys-training-4b5a856cb913)

## 2. Triton TBE 순전파 구현 / Implementation of Triton TBE Forward

순전파는 각 테이블과 백(bag)마다 인덱스가 가리키는 행들을 수집(gather)하고, 필요하면 샘플별 가중치(per-sample weight)를 곱한 뒤, FP32로 누적하고(FP32 가중치는 FP64로 누적), D 폭의 풀링된 출력 하나를 씁니다. 두 가지 구현을 만들었습니다. 하나는 일반 수집(generic gather) 구현이고, 다른 하나는 작은 테이블용 히스토그램(histogram)을 쓰는 빠른 경로(fast path) 구현입니다.
> For each table and bag, forward gathers the indexed rows, optionally multiplies them by per-sample weights, accumulates them in FP32 (FP64 for FP32 weights), and writes one D-wide pooled output. We built two implementations: a generic gather and a fast path implementation with a small-table histogram.

### 일반 수집 경로 / The general gather path

- **그리드(Grid).** 일반 실행은 `ceil(B / BAGS_PER_PROGRAM)`개의 프로그램을 사용합니다. 각 프로그램은 B×T 그리드를 실행하는 대신 T개의 특징(feature)을 순회합니다.
- **수집 폭(Gather width).** 내부 루프는 서로 독립적인 행 로드를 4개 발행합니다. 튜닝된 2-백(two-bag) 경로는 8개를 발행합니다
- **프로그램당 백 수(Bags per program).** VBE가 아니고 FP32도 아닌 대규모 워크로드는 프로그램당 백 2개를 사용합니다. 히스토그램 특징을 따로 떼어 내면, 남은 일반 특징 범위는 4개를 사용합니다. 그 밖의 형태(shape)는 1개를 사용합니다
- **인덱스와 오프셋 폭(Index and offset width).** TorchRec은 선형화된 범위가 2^31 미만에 들어가면 구성(config)에 따라 int32 인덱스와 오프셋을 받습니다. 이렇게 하면 기본값은 int64로 유지하면서도 인덱스/오프셋 저장 공간과 CUB 기수 정렬(radix sort)의 키 폭을 절반으로 줄입니다
- **누적(Accumulation).** FP16/BF16 가중치는 FP32로 누적합니다. FP32 가중치는 큰 D에서 정확도를 지키기 위해 FP64로 누적합니다

> - **Grid.** The generic launch uses `ceil(B / BAGS_PER_PROGRAM)` programs. Each program loops over T features instead of launching a B×T grid.
> - **Gather width.** The inner loop issues four independent row loads. The tuned two-bag path issues eight
> - **Bags per program.** Large non-VBE, non-FP32 workloads use two bags per program. When a histogram feature is split out, the remaining generic feature ranges use four. Other shapes use one
> - **Index and offset width.** TorchRec accepts config-driven int32 indices and offsets when the linearized range fits below 2^31. This halves index/offset storage and the CUB radix-sort key width while keeping int64 as the default
> - **Accumulation.** FP16/BF16 weights accumulate in FP32. FP32 weights accumulate in FP64 to preserve accuracy at large D

### 작은 테이블용 히스토그램과 텐서 코어 경로 / The small-table histogram and tensor-core path

이 특화 경로는 E≤64, 64≤D≤128, L≥64, FP16 가중치, FP32 출력이고 샘플별 가중치와 VBE가 없는 특징 하나에 대해 선택됩니다. 프로그램 하나가 백 16개를 처리합니다. 처음 256개 인덱스로 히스토그램을 만들고 `tl.dot`으로 개수(counts) × 테이블을 계산합니다. 나머지 인덱스는 스칼라 경로가 처리합니다. 그 밖의 형태는 일반 커널을 사용합니다.
> The specialized path is selected for one feature with E≤64, 64≤D≤128, L≥64, FP16 weights, FP32 output, no per-sample weights, and no VBE. One program handles 16 bags. It builds a histogram over the first 256 indices and evaluates counts × table with `tl.dot`; a scalar path handles the remaining indices. Other shapes use the generic kernel.

### 범위 검사 / Bounds checking

독립(standalone) 경로는 Triton 순전파 전에 개선된 CUDA 검증 단계를 사용합니다. B200에서 이 방식은 여러 워크로드에 걸쳐 범위 검사(bounds-check) 부분에서 최대 1.24배의 속도 향상을 얻습니다. `fused_bounds_check`를 켜면, 조건을 만족하는 커널이 잘못된 입력 텐서를 검증하고 바로잡습니다. 오프셋 검증과 수정은 여전히 별도의 작은 커널로 남아 있습니다. 가중치가 있는 경우, 가변 배치(variable batched), AMD, 전치 끌어올리기(transpose-hoist) 같은 다른 구성에서는 표준 검증으로 대체(fallback)합니다. 이 옵션은 TorchRec을 통해 노출되며 기본값은 꺼져 있습니다.
> The standalone path uses an updated CUDA validation step before the Triton forward pass. On B200, this achieves up to 1.24x speedup on the bounds-check component across workloads. When `fused_bounds_check` is enabled, the eligible kernel validates and repairs invalid input tensors. Offset validation and repair remain a separate small kernel. Other configurations, such as weighted, variable batched, AMD, and transpose-hoist cases, fall back to standard validation. The option is exposed through TorchRec and defaults off.

### 순전파 상태 재사용과 전처리 / Forward-state reuse and preprocessing

정확한 행 단위 Adagrad(exact row-wise Adagrad)는 순전파의 히스토그램을 저장해 둘 수 있습니다. 역전파는 이 개수를 FP32로 결과를 내는 보정된(compensated) FP16 high/low GEMM에 사용한 뒤, 옵티마이저를 적용합니다. 순전파와 역전파는 여전히 따로 실행되며, 재사용하는 것은 히스토그램 개수뿐입니다.
> Exact row-wise Adagrad can save the forward histogram. Backward uses those counts for the compensated FP16 high/low GEMM into FP32, then applies the optimizer. Forward and backward remain separate launches; only the histogram counts are reused.

또 다른 선택적 코어 모듈 경로는 인덱스 전치(transpose), 정렬, 런 길이 인코딩(run-length encoding)을 순전파로 옮기고, 그 메타데이터를 autograd를 통해 반환합니다. 기본값은 꺼져 있고, 현재 TorchRec 래퍼에서는 노출되지 않습니다. 큰 B200 구성에서 순전파는 22.844 ms에서 33.252 ms로, 역전파는 56.693 ms에서 32.931 ms로, 합산 지연 시간은 79.537 ms에서 66.183 ms(−16.8%)로 바뀝니다.
> Another optional core-module path moves index transpose, sort, and run-length encoding into forward and returns the metadata through autograd. It defaults off and is not exposed by the current TorchRec wrapper. On a large B200 configuration, forward moves from 22.844 ms to 33.252 ms, backward from 56.693 ms to 32.931 ms, and combined latency from 79.537 ms to 66.183 ms (−16.8%).

![TBE 순전파: 하나의 연산자, 두 가지 실행 경로 / TBE forward: one operator, two execution paths](/assets/blog/2026-10-06-modernizing-table-batched-embeddings-with-fbtriton/1-1.png){:style="width:100%"}

## 3. Triton TBE 역전파 커널 구현 / Implementation of Triton TBE Backward Kernel

TBE 역전파의 핵심 로직은 다음과 같습니다. 배치 안 어디에서든 접근한 고유한 (테이블, 행) 쌍마다, 그 행에 접근한 모든 배치 위치에서 온 상위(upstream) 변화도(gradient) 행을 합산한 뒤, 그 행에 옵티마이저 업데이트를 정확히 한 번 적용합니다.
> The core logic of the TBE backward propagation: for every unique (table, row) pair touched anywhere in the batch, sum the upstream gradient rows from every batch position that touched it, then apply exactly one optimizer update to that row.

기호 설명: T는 테이블 수, E는 임베딩 크기(즉, 행 수), D는 임베딩 차원(즉, 열 수), L은 풀링 계수(즉, [임베딩 백(embedding bag)](https://docs.pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html) 크기), B는 배치 크기(즉, 요청당 임베딩 백 수)입니다.
> Annotations: T is number of tables; E is embedding size (i.e. number of rows); D is embedding dimension (i.e. number of cols); L is pooling factor (i.e. [embedding bag](https://docs.pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html) size); B is batch size (i.e. number of embedding bags per request)

순전파. 임베딩 테이블 T는 GPU에 있는 2차원 텐서(ExD)입니다.
> Forward-pass. An embedding table T is a 2-D tensor (ExD) located in the GPU.

- 입력: \[id1, id2, id3, id4\] 같은 희소 특징(sparse feature, id-list 또는 id-score-list) 하나(여기서는 L=4, B=1), 또는 여러 개의 희소 특징(B>1).
- 목표: T\[id1%E\]+T\[id2%E\]+T\[id3%E\]+T\[id4%E\] 합을 계산해 순전파 출력을 얻습니다. B=100이면 순전파 출력은 100개입니다.

> - Given: a sparse feature (id-list or id-score-list) such as \[id1, id2, id3, id4\] (here L=4 and B=1) or multiple sparse features (B>1).
> - Goal: By computing the sum T\[id1%E\]+T\[id2%E\]+T\[id3%E\]+T\[id4%E\], we get its forward output. If B=100, we will have 100 forward outputs.

역전파

- 입력: 순전파 출력과 출력 변화도(output.backward(grads))
- 목표: weights.grad를 계산하고 가중치를 업데이트합니다
    - 변화도 계산: 각 인덱스 "id"에 대해, "id"를 포함한 임베딩 백들에서 온 출력 변화도를 모두 합산합니다. (TxB=4.2M, B=128K일 때 인덱스 통계는 전체/중복 제거 후/최고 빈도 = 83M/3M/125K 정도가 될 수 있습니다)
    - 가중치 업데이트: weight = weight – grad \* LR 처럼 간단합니다
- TBE 역전파에는 텐서 코어 연산이 없지만, 데이터 이동과 리덕션(reduction)이 많고 부하 불균형(load imbalance) 문제를 겪습니다.

> Backward-pass
> - Given: forward output and output grads (output.backward(grads))
> - Goal: calculate weights.grad and update weights
>     - Calc grads: for each index “id”, compute the sum of all the output grads from the embedding bags containing “id”. (With TxB=4.2M, B=128K, the indice stats can be total/dedup/highest\_freq = 83M/3M/125K)
>     - Update weights: as simple as weight = weight – grad \* LR
> - TBE backward has no tensor core ops but it requires heavy data move/reduction and suffers from load imbalance issues.

![출력 변화도와 임베딩 변화도의 형태 / Output grad and embedding grad shapes](/assets/blog/2026-10-06-modernizing-table-batched-embeddings-with-fbtriton/Screenshot-2026-10-06-at-3.03.34-PM.png){:style="width:100%"}

`transpose_embedding_input`은 배치를 런(run)으로 뒤집습니다. 런은 고유한 행 하나와 그 행에 접근한 샘플들을 짝지은 것입니다. 이 연산은 순전파로 끌어올려(hoist) 역전파의 임계 경로(critical path)에서 빠집니다.
> `transpose_embedding_input` inverts the batch into runs: one unique row paired with the samples that touched it. This operation is hoisted into the forward pass, off the backward critical path.

![인덱스 전치와 런 길이 인코딩 / Index transpose and run-length encoding](/assets/blog/2026-10-06-modernizing-table-batched-embeddings-with-fbtriton/4-1.png){:style="width:100%"}

세그먼트 길이(Segment length, SL)는 한 행에 접근한 샘플 수입니다. 모든 설계가 이 변수에 따라 갈리며, 이 값은 한 배치 안에서도 1부터 수백만까지 퍼져 있습니다. 런은 SL에 따라 세 가지 커널 중 하나로 보내집니다.
> Segment length (SL), the number of samples touching a row, is the variable everything keys on, and it spans one to millions inside a single batch. Runs are routed by SL to one of three kernels.

![세 가지 역전파 커널과 런이 각 커널로 가는 경로 / The three backward kernels and how runs reach them](/assets/blog/2026-10-06-modernizing-table-batched-embeddings-with-fbtriton/5.png){:style="width:100%"}

- **short\_run** (`SL < 256`): 프로그램 하나가 런을 처음부터 끝까지 맡습니다. 수집, 레지스터에서 누적, 옵티마이저 적용, 저장까지 모두 합니다. 런을 단독으로 맡으므로 원자적(atomic) 연산 없이 일반 저장(plain store)을 씁니다.
- **grad\_accum + apply** (`SL >= 256`, 기본값): 런을 256개 조회 단위의 청크(chunk)로 나누고, 부분 결과(partial)를 작업 공간(workspace)에 모은 뒤, 두 번째 커널이 옵티마이저를 적용합니다.
- **fused** (`SL >= 256`, Blackwell, 매우 큰 배치): 같은 방식으로 나누지만, 디바이스 범위 펜스(device-scope fence)를 써서 마지막 하위 프로그램(sub-program)이 실행 한 번 안에서 업데이트를 적용합니다.

> - **short\_run** (`SL < 256`): one program owns the run start to finish: gather, accumulate in registers, optimizer, store. Exclusive ownership means a plain store, no atomics.
> - **grad\_accum + apply** (`SL >= 256`, default): the run is split into 256-lookup chunks, partials land in a workspace, a second kernel applies the optimizer.
> - **fused** (`SL >= 256`, Blackwell, very large batches): same split, but a device-scope fence lets the last sub-program apply the update in one launch.

가중치가 있는(weighted) 테이블은 누적 전에 각 변화도 행에 샘플별 가중치를 곱한다는 점만 다릅니다.
> Weighted tables differ only in scaling each gradient row by its per-sample weight before accumulating.

## 4. 엣지 케이스와 성능 개선 / Edge Cases and performance improvements

### 문제 1. 불균형한 프로그램 / Problem 1. imbalanced programs

프로그램 하나가 2백만 건의 조회를 순회하는 동안 GPU는 놀고 있고, 반대로 조회가 1건뿐인 수많은 런(긴 꼬리)은 하는 일이 거의 없는데도 런마다 드는 비용을 고스란히 치릅니다.
> A single program walking two million lookups leaves the GPU idle, while a huge tail of one-lookup runs pays full per-run cost for almost no work.

**해결 1.1: 긴 런 나누기.** 임계값 이상인 런은 split-K 방식처럼 256개 조회 단위의 고정 청크로 나뉩니다. 2백만 조회짜리 런은 대략 8천 개의 하위 프로그램이 되므로, 한 행의 작업만으로도 머신 전체를 채울 수 있습니다.
> **Solution 1.1: split long runs.** Runs at or above the threshold become fixed 256-lookup chunks, split-K style. A two-million-lookup run turns into roughly eight thousand sub-programs, enough to fill the machine from one row’s work.

**해결 1.2:** [**CLC**](https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_cluster_launch_control.html#blackwell-warp-specialized-persistent-kernel) **(**[**TLX Blackwell**](https://github.com/facebookexperimental/triton/pull/516)**)** 커널은 한 번 시작되면 계속 상주(persist)하면서, 현재 'run\_id'를 마친 뒤 '비어 있는' run\_id를 계속 가져옵니다(steal). 작업을 가져오는 데 성공하면 같은 블록에서 처리하고, 실패하면 커널이 종료됩니다. 이제 부하 불균형 문제를 해결하려고 run\_id 처리를 빈도에 따라 둘로 나누는 식의 소프트웨어 기반 해법을 쓸 필요가 없습니다. 대신 CLC가 하드웨어 기반 해법을 제공합니다.
> **Solution 1.2:** [**CLC**](https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_cluster_launch_control.html#blackwell-warp-specialized-persistent-kernel) **(**[**TLX Blackwell**](https://github.com/facebookexperimental/triton/pull/516)**)** A kernel will persist once started initially and keep stealing ‘free’ run\_id after finishing the current ‘run\_id’. If the kernel steals a workload successfully, it will process it in the same block. Otherwise, the kernel exits. To address the load imbalance issues, we no longer need a software-based solution like bifurcating run\_id processing by frequency. Instead CLC provides a hardware-based solution.

### 문제 2. 수집 폭에 따라 레지스터 사용량이 급격히 늘어난다 / Problem 2. the gather width is a register cliff

버퍼에 담긴 각 행은 `BLOCK_SIZE`개의 *64비트 주소*를 살아 있는(live) 상태로 유지합니다. `dout_row_start_ptr[:, None] + col_offsets[None, :]`가 포인터 타일 전체를 실체화(materialize)하기 때문입니다. 비용이 `width × BLOCK_SIZE`이므로, 어떤 행 폭에 맞춰 튜닝한 폭은 다른 행 폭에서는 맞지 않습니다.
> Each buffered row keeps `BLOCK_SIZE` *64-bit addresses* live, because `dout_row_start_ptr[:, None] + col_offsets[None, :]` materializes a whole pointer tile. Cost is `width × BLOCK_SIZE`, so a width tuned at one row width is wrong at another.

**해결: 모든 단계(tier)에서 대상별로 구성하는 폭.** B200에서 측정한 결과입니다:
> **Solution: per-target config width, in every tier.** Measured on B200:

| 단계 / tier | 폭 / width | 레지스터 / registers | 점유율 / occupancy | 효과 / effect |
|---|---|---|---|---|
| 짧은 런, 가중치 없음 (short run, unweighted) | 8 → 2 | 184 → 64 | 12.5% → 49.9% | 0.41 → 1.05 |
| 긴 런, 누적 (long run, accumulate) | 8 → 2 | 158 → 62 | 17.6% → 44.7% | 동등 이상 샤드 비율(fleet parity) 82% → 87% |
| 짧은 런, 가중치 있음 (short run, weighted) | 4 → 2 | 125 → 64 | 24.8% → 49.3% | 가중치 있는 샤드의 동등 이상 비율(weighted parity) 51% → 69% |

### 문제 3. BLOCK\_SIZE는 constexpr이다 / Problem 3. BLOCK\_SIZE is a constexpr

실행 한 번에서 *모든* 테이블에 걸쳐 `BLOCK_SIZE`를 `next_pow2(max_D)`로 정해야 합니다. 그래서 조회가 몰리는 테이블이 가장 넓은 테이블보다 훨씬 좁으면, 수집한 행마다 대부분이 마스킹된(masked-off) 레인(lane)입니다.
> One launch must size `BLOCK_SIZE` to `next_pow2(max_D)` across *all* tables, so when the lookup-dominant table is much narrower than the widest, most of every gathered row is masked-off lanes.

**이를 극복하려고 차원별로 버킷(bucket)을 나눕니다.** 짧은 런은 분류 단계에서 `next_pow2(D)`마다 하나씩 있는 버킷으로 보내지고, 버킷마다 자체 `BLOCK_SIZE`로 실행합니다. 이 과정은 분류 커널에 합쳐지므로 추가 패스(pass) 비용이 없습니다. 프로파일링으로 확인해 보면, 이 방식으로 얻는 것은 점유율 향상이 아니라 레인 낭비 감소입니다.
> **To overcome that, we bucket by dimension.** Short runs are routed to one bucket per `next_pow2(D)` during classification, each launching with its own `BLOCK_SIZE`. It folds into the classification kernel, so it costs no extra pass; profiling confirms it buys reduced lane waste, not occupancy.

어떤 런이 긴지는 데이터에 따라 달라 GPU에서만 알 수 있습니다. 이 개수를 `.item()`으로 다시 읽어 오면 역전파 때마다 `cudaStreamSynchronize`가 일어납니다.
> Which runs are long is data-dependent and known only on the GPU, and reading those counts back with `.item()` is a `cudaStreamSynchronize` every backward.

**해결: 형태 정보를 GPU에 둡니다.** 작업 공간은 인덱스 개수로 계산한 상한에 맞춰 미리 할당하고, 분류는 스트림 압축(stream compaction)을 위해 원자적 카운터를 쓰는 커널에서 실행합니다:
> **Solution: keep the shapes on the GPU.** Workspace is preallocated to a bound computed from the index count, and classification runs in a kernel using atomic counters for stream compaction:

```python
is_long        = (run_len >= threshold) & mask
num_long_block = tl.sum(is_long.to(tl.int32))
long_base      = tl.atomic_add(num_long_ptr, num_long_block)
long_local     = tl.cumsum(is_long.to(tl.int32), axis=0) - 1
tl.store(long_run_ids_ptr + (long_base + long_local).to(tl.int64),
         offsets.to(tl.int32), mask=is_long)
```

원자적 연산은 원소마다가 아니라 블록마다 한 번만 하고, 블록 안의 오프셋은 접두사 합(prefix sum)으로 구합니다. 그다음 커널들은 이 개수를 디바이스 포인터로 받아 while 루프로 작업을 스스로 나눠 가집니다.
> One atomic per block rather than one per element, intra-block offsets from a prefix sum. Kernels then consume the counts as device pointers and self-distribute with while-loops.

### 문제 5. 분할된 런에는 프로그램 간 배리어가 필요하다 / Problem 5. split runs need a cross-program barrier

런을 나누고 나면, 모든 하위 프로그램의 부분 결과가 도착한 뒤에야 옵티마이저 업데이트를 실행할 수 있습니다. Triton에는 디바이스 범위 펜스가 없었기 때문에, 그 대가로 두 번째 커널과 전역 메모리 왕복(round trip)이 필요했습니다.
> Once a run is split, the optimizer update can only run after every sub-program’s partial has landed. Triton had no device-scope fence, so this cost a second kernel and a global-memory round trip.

**해결: 펜스 후 카운트다운(TLX Blackwell).** TLX가 펜스를 노출하므로, 마지막 하위 프로그램이 같은 실행 안에서 업데이트를 적용할 수 있습니다:
> **Solution: fence, then countdown (TLX Blackwell).** TLX exposes the fence, letting the last sub-program apply the update in the same launch:

```python
tl.atomic_add(temp_grad_buffer_ptr + temp_grad_offset + col_offsets, grad, mask=mask)
tlx.fence("gpu")
remaining = tl.atomic_add(grad_accum_counter_ptr + grad_buffer_id, -1)
if remaining == 1:
    ...  # 마지막 하위 프로그램이 옵티마이저를 적용하고 저장
```

정확성의 근거는 순서입니다. 펜스는 카운트다운이 감소하기 전에 각 부분 결과가 디바이스 전체에 보이게 만듭니다. 그래서 remaining == 1을 본 프로그램은 완전한 합을 읽습니다. 펜스가 없으면 다른 프로그램의 atomic\_add가 아직 진행 중일 때 어떤 프로그램이 카운트다운을 먼저 차지할 수 있고, 이는 드러나지 않는(silent) 수치 오류가 됩니다.
> The ordering is the correctness argument: the fence makes each partial visible device-wide before the countdown decrements, so the program that sees remaining == 1 reads a complete sum. Without it one could win the countdown while another’s atomic\_add was in flight, a silent numerical error.

### 문제 6. 부분 결과 병합은 행 전체에 걸친 원자적 연산이다 / Problem 6. merging partials is a row-wide atomic

모든 하위 프로그램은 BLOCK\_SIZE 크기의 행 전체를 런의 작업 공간 슬롯에 원자적으로 더합니다. 런 하나가 8천 개 청크로 나뉘면 8천 개 프로그램이 같은 행을 두고 경합하며, tl.atomic\_add는 이 병합을 원소 단위 연산으로 하나씩 발행합니다.
> Every sub-program atomically adds an entire BLOCK\_SIZE row into the run’s workspace slot. A run split into eight thousand chunks means eight thousand programs contending on the same row, and tl.atomic\_add issues that merge one element-wise operation at a time.

**해결: TMA를 통한 리덕션(TLX Blackwell).** Blackwell에는 부분 결과 병합에 `tl.atomic_add`보다 나은 명령어인 `cp.reduce.async.bulk.tensor`가 있으며, 이는 `tlx.async_descriptor_store(..., store_reduce="add")`로 노출됩니다.
> **Solution: reduce through TMA (TLX Blackwell).** Blackwell has a better instruction than `tl.atomic_add` for merging partials `cp.reduce.async.bulk.tensor`, exposed as `tlx.async_descriptor_store(..., store_reduce="add")`.

## 5. 결과와 분석 / Results and Analysis

### 결과 / Results

GB200에서 307개 샤드 구성(서로 다른 형태 283개), 정확한 행 단위 Adagrad, FP16 가중치로 측정했습니다. 지표는 Triton 성능을 CUDA TBE 성능으로 나눈 값입니다. 순전파 속도 향상의 중앙값은 1.28배입니다.
> 307 shard configurations (283 distinct shapes) on GB200, exact row-wise Adagrad, FP16 weights. Metric: Triton performance divided by CUDA TBE. Median forward speedup is 1.28×.

![B200에서 측정한 Triton TBE 순전파 대 CUDA TBE / Triton TBE forward vs CUDA TBE, B200 measurements](/assets/blog/2026-10-06-modernizing-table-batched-embeddings-with-fbtriton/5-1.png){:style="width:100%"}

역전파 데이터는 다음과 같습니다:
> For backward data:

![307개 프로덕션 랭크 샤드에서 Triton TBE 역전파 대 CUDA TBE / Triton TBE backward vs CUDA TBE, 307 production rank shards](/assets/blog/2026-10-06-modernizing-table-batched-embeddings-with-fbtriton/6.png){:style="width:100%"}

### 분석: 왜 여기서는 Triton이 CUDA를 앞서는가? / Analysis: why does Triton beat CUDA here?

이득의 대부분은 런 길이 기준을 바꾼 데서 나옵니다. CUDA는 `SL = 32`에서 행마다 협력형 CTA(Cooperative Thread Array)를 쓰는 커널로 전환하지만, Triton은 `SL = 256`까지 단순 스트리밍을 유지합니다.
> We gain most of the win through changing the run-length. CUDA escalates to a cooperative CTA(Cooperative Thread Array)-per-row kernel at `SL = 32`; Triton stays on simple streaming to `SL = 256`.

![두 구현이 전략을 바꾸는 지점 / Where the two implementations switch strategy](/assets/blog/2026-10-06-modernizing-table-batched-embeddings-with-fbtriton/7.png){:style="width:100%"}

**속도 향상은 더 나은 메모리 처리량에서 나옵니다.** 이 구간 깊숙한 곳에서 Triton은 패스 전체에서 같은 양의 DRAM 바이트를 옮기고(0.91배) 로드 요청은 29% 더 많이 발행하면서도 4.3배 빠르게 실행됩니다. Nsight Compute를 보면 차이가 어디서 나는지 알 수 있습니다. 이 형태를 맡는 CUDA 커널은 678 GB/s를 달성하는 반면, Triton 커널은 같은 점유율에서 3,948 GB/s(5.8배)를 달성합니다. SL = 32 바로 위에서는 CTA 전체 동기화 비용을 분산(amortize)할 만큼의 작업이 없고, Triton 경로에는 협력(cooperation) 자체가 없습니다.
> **The speedup is from better memory throughputput.** Deep in that band Triton runs 4.3x faster while moving the same DRAM bytes across the pass (0.91x) and issuing 29% more load requests. Nsight Compute shows where the difference lives: the CUDA kernel carrying this shape achieves 678 GB/s, versus 3,948 GB/s for the Triton kernel (5.8x) at identical occupancy. Just above SL = 32 there is nothing to amortize CTA-wide synchronization against, and Triton’s path has no cooperation at all.

![Nsight Compute: 비슷한 바이트, 5.8배의 달성 대역폭 / Nsight Compute: comparable bytes, 5.8x the achieved bandwidth](/assets/blog/2026-10-06-modernizing-table-batched-embeddings-with-fbtriton/8.png){:style="width:100%"}

성과는 구성 값에서 나왔습니다. 2절의 모든 수정은 구성 변경으로 노출됩니다. 템플릿으로 생성하는 CUDA에서 전환 지점이나 수집 폭을 옮기려면, 어떤 커널이 무엇을 처리할지부터 구조를 다시 짜야 합니다.
> The wins came from config values. Every fix in Section 2 is exposed as a config change. Moving an escalation point or gather width in template-generated CUDA means restructuring which kernel handles what.

**Triton이 여전히 지는 곳.** 작업이 모두 길이 4 미만의 런에 있는 형태는 순수한 의존 로드 지연(dependent-load latency) 때문에 동등 수준(parity)에 못 미칩니다. CUDA의 행당 워프(warp-per-row) 방식이 런별 메타데이터 비용을 더 잘 분산합니다. 이런 형태는 307개 샤드 중 11개이고, 각각 약 1밀리초 미만입니다. SL = 256을 넘는 형태는 앞서지 않고 동등한 수준입니다. 이런 형태가 실제 워크로드에서 병목이 되면 더 투자하겠습니다.
> **Where Triton still loses.** Shapes whose work is entirely in runs shorter than 4 sit below parity because of pure dependent-load latency; CUDA’s warp-per-row amortizes per-run metadata better. That is 11 of 307 shards, each under about a millisecond. Shapes above SL = 256 are at parity rather than ahead. We will invest more if they become a bottleneck on workloads.

## 6. 성능 너머: FBTriton이 열어 주는 것 / Beyond Performance: What FBTriton Unlocks

FBTriton의 가장 오래 남을 성과는 희소 경로 전체(순전파와 역전파 모두)를 이제 일반 Python으로 작성하며, 그 코드가 원래의 CUDA 템플릿만 놓고 비교해도 더 작다는 점입니다. 그 결과 순전파에서 보여 준 것처럼 앞으로 더 많은 융합(fusion)을 적용할 수 있고, 다음과 같은 이점을 얻습니다:

- **빠른 개발 속도:** FBTriton은 높은 머신 효율과 개발자 효율을 동시에 제공합니다. CUDA Jinja 템플릿과 비교해, Triton TBE를 쓰면 랭킹/인프라 엔지니어(rank/infra engineers)가 더 나은 모델 성능을 위한 최신(SOTA) 임베딩 알고리즘을 훨씬 쉽고 빠르게 구현할 수 있습니다.
- **이식성과 민첩성을 함께:** FBTriton은 Blackwell, Hopper, AMD 중 어느 아키텍처에서 실행하든 커널 본문을 똑같이 유지합니다. 하드웨어마다 별도의 코드 분기(fork)를 만드는 대신, 특정 기능(CLC, TMA 벌크 원자적 리덕션, 디바이스 범위 펜스 등)은 덧붙이는 플래그(additive flag)로 통합합니다.
- **"메가 희소 커널(Mega Sparse Kernel)"로 가는 길:** CUDA 커널을 Triton으로 다시 쓰면 `[forward, backward]`와 `[prologue, epilogue]`에 걸친 대규모 융합 기회가 열립니다. 옵티마이저를 역전파의 에필로그(epilogue)로 다루어 FBTriton은 추가 메모리 패스를 없앱니다. 이 융합은 희소 연산 시퀀스 전체를 커널 하나로 합치며, FP8 모멘텀 스케일링 같은 복잡한 기능도 누적 코드 몇 줄로 줄여 줍니다.

> The most durable outcome of FBTriton is that the entire sparse path (both forward and backward) is now written in ordinary Python and is smaller than the original CUDA templates alone. As a result, we can apply more possible fusions in the future like demonstrated in forward and unlock these benefits:
> - **Rapid Developer Velocity:** FBTriton delivers high machine efficiency and developer efficiency simultaneously. Compared to CUDA Jinja templates, Triton TBE makes it much easier for rank/infra engineers to rapidly implement SOTA embedding algorithms for better model performance.
> - **Simultaneous Portability and Agility:** FBTriton maintains identical kernel bodies whether running on Blackwell, Hopper, or AMD architectures. Instead of creating separate code forks for different hardware, specific features (like CLC, TMA bulk atomic reductions or device-scope fences) are integrated as additive flags.
> - **Pathway to a “Mega Sparse Kernel”:** Rewriting CUDA kernels in Triton unlocks massive `[forward, backward]` and `[prologue, epilogue]` fusion opportunities. By treating the optimizer as a backward epilogue, FBTriton eliminates extra memory passes. This fusion collapses the entire sparse sequence into one kernel, reducing complex features like FP8 momentum scaling to just a few lines of accumulation.

코드: [링크](https://github.com/meta-pytorch/torchrec/tree/main/torchrec/distributed/triton_tbe)
> Code: [link](https://github.com/meta-pytorch/torchrec/tree/main/torchrec/distributed/triton_tbe)
