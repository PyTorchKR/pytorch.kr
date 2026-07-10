---
layout: blog_detail
title: "TLX Block Attention: 고정 블록 희소 셀프 어텐션을 위한 워프 특화 Blackwell 커널"
author: Jake Siso, Dev (Devashish) Shankar, Jackie (Jiaqi) Xu, Jacky Zhou, Darren Liu, Han Xu, Yasmine Badr, Dan Chanpuriya, Hongtao Yu, Max Leung
category: ["pytorch.org", "translation"]
org_title: "TLX Block Attention: A Warp-Specialized Blackwell Kernel for Fixed-Block Sparse Self-Attention"
org_link: https://pytorch.org/blog/tlx-block-attention-a-warp-specialized-blackwell-kernel-for-fixed-block-sparse-self-attention/
---

*코드는 다음에서 확인할 수 있습니다: [https://github.com/facebookresearch/ads\_model\_kernel\_library](https://github.com/facebookresearch/ads_model_kernel_library)*

*이 글에서는 TLX Block Attention의 설계를 소개합니다. TLX Block Attention은 NVIDIA Blackwell GPU를 겨냥한 Triton 커널로, 블록 대각(block-diagonal) 어텐션 패턴을 컴파일 시점에 알고 있다는 점을 활용해 범용 어텐션 구현에 존재하는 여러 종류의 알고리즘적 오버헤드를 통째로 제거합니다. NVIDIA B200 GPU에서 이 커널은 Flash Attention v2 대비 순전파(forward) 약 1.85배, 역전파(backward) 약 2.50배의 속도 향상을 달성하며, 회전 임베딩(rotary embedding)을 어텐션 에필로그(epilogue)에 융합한 경우 어텐션과 회전 임베딩을 합친 역전파에서 약 3.5배의 속도 향상을 냅니다.*
> *In this post, we present the design of TLX Block Attention — a Triton kernel targeting NVIDIA Blackwell GPUs that exploits compile-time knowledge of a block-diagonal attention pattern to eliminate entire categories of algorithmic overhead present in general-purpose attention implementations. On NVIDIA B200 GPUs, the kernel achieves a ~1.85× forward and ~2.50× backward speedup over Flash Attention v2, and a ~3.5× speedup for the combined attention-and-rotary backward pass when rotary embeddings are fused into the attention epilogue.*

이 작업은 TLX(Triton Language Extensions) 위에 구축되었습니다. TLX는 Triton 컴파일러에 대한 저수준 확장의 집합으로, NVIDIA Blackwell GPU에서 워프 특화(warp specialization), 비동기 텐서 코어 연산, 메모리 계층 관리에 대한 하드웨어 네이티브 제어를 노출합니다. TLX는 Triton의 고수준 Python 생산성과, 전통적으로 순수 CUDA나 CUTLASS를 요구하던 세밀한 하드웨어 제어 사이의 간극을 메웁니다. TLX에 대한 자세한 내용은 [triton-ext 저장소](https://github.com/triton-lang/triton-ext)를 참고하세요.
> This work is built on TLX (Triton Language Extensions) — a set of low-level extensions to the Triton compiler that expose hardware-native control over warp specialization, asynchronous tensor core operations, and memory hierarchy management on NVIDIA Blackwell GPUs. TLX bridges the gap between Triton’s high-level Python productivity and the fine-grained hardware control traditionally requiring raw CUDA or CUTLASS. For more on TLX, see the [triton-ext repository](https://github.com/triton-lang/triton-ext)

## 1. 서론 / Introduction

셀프 어텐션(self-attention)은 시퀀스 안의 각 요소가 다른 모든 요소에 대해 얼마나 관련 있는지를 모델이 저울질하도록 해 주는 메커니즘으로, 본질적으로 "이 입력의 어느 부분이 다른 부분을 이해하는 데 정보를 주어야 하는가?"를 묻는 것입니다. 이는 트랜스포머(Transformer) 아키텍처의 핵심 구성 요소이며, 이러한 모델이 데이터에서 문맥 의존적이고 풍부한 관계를 포착할 수 있게 하는 요인입니다. 직관적으로 비유하자면 "과거의 결정이 현재와 미래의 결정에 어떻게 정보를 주는가?"라고 볼 수 있습니다.
> Self-attention is a mechanism that lets a model weigh how relevant each element in a sequence is to every other element — essentially asking “which parts of this input should inform my understanding of each other part?” It’s the core building block of Transformer architectures and is what allows these models to capture rich, context-dependent relationships in data. A good intuition might be: how do one’s past decisions inform present and future ones?

블록 대각 셀프 어텐션(block-diagonal self-attention) — 시퀀스를 고정 크기 그룹으로 분할하고 각 그룹은 자기 그룹 안에서만 어텐션을 수행하는 방식 — 은 추천 및 특징 상호작용(feature-interaction) 모델에서 널리 쓰이는 패턴입니다([BlockBERT, Qiu et al., EMNLP 2020](https://arxiv.org/abs/1911.02972)) \[1\]. 우리의 광고 랭킹 스택에서 운영 워크로드는 일반적으로 배치 크기 1152, 최대 약 4k 토큰의 시퀀스, 64 또는 128의 헤드 차원(head dimension)으로 실행되며, 시퀀스 길이가 길어질수록 어텐션 구조의 희소성(sparsity)은 약 70%에 이릅니다. 이러한 모델이 더 깊고 넓어질수록 어텐션 비용이 지배적인 병목이 됩니다.
> Block-diagonal self-attention — where the sequence is partitioned into fixed-size groups that attend only within themselves — is a widely-used pattern in recommendation and feature-interaction models ([BlockBERT, Qiu et al., EMNLP 2020](https://arxiv.org/abs/1911.02972)) \[1\]. In our ads ranking stack, production workloads typically run batch sizes of 1152 with sequences up to ~4k tokens, head dimensions of 64 or 128, and ~70% sparsity in the attention structure with increasing sequence lengths. As these models grow deeper and wider, attention cost becomes the dominant bottleneck.

![슬라이딩 윈도우와 블록 어텐션의 어텐션 패턴 비교 / Comparison of sliding window and block attention patterns](/assets/blog/2026-05-26-tlx-block-attention-a-warp-specialized-blackwell-kernel-for-fixed-block-sparse-self-attention/unnamed-4-1.png){:style="width:100%"}

오늘날 이러한 워크로드는 블록 마스킹(block masking)이나 슬라이딩 윈도우를 적용한 Flash Attention v2 같은 범용 커널로 실행됩니다. FlexAttention(FA4) *\[7\]* 은 블록 희소(block-sparse) 패턴을 지원하지만 최소 타일 크기가 256이라 이러한 모델이 요구하는 64토큰 블록과는 호환되지 않습니다. 이 타일 크기에서는 블록 마스킹을 적용한 Flash Attention v2가 여전히 가장 강력한 기준선(baseline)이지만, 성능 면에서 상당한 여지를 남깁니다. Flash Attention의 타일 단위 반복(tiled iteration), 온라인 소프트맥스(online softmax) 보정, 로그섬익스프(logsumexp) 기록, 보조 커널 실행은 임의 길이의 인과(causal) 어텐션에는 필수적이지만, 패턴이 블록 대각이고 컴파일 시점에 알려져 있을 때는 순수한 오버헤드일 뿐입니다.
> Today these workloads run on general-purpose kernels like Flash Attention v2 with block masking or sliding window. FlexAttention (FA4) *\[7\]* supports block-sparse patterns but operates at a minimum tile size of 256 — incompatible with the 64-token blocks these models require. Flash Attention v2 with block masking remains the strongest available baseline at this tile size, but leaves significant performance on the table. Flash Attention’s tiled iteration, online softmax correction, logsumexp bookkeeping, and auxiliary kernel launches are essential for arbitrary-length causal attention — but pure overhead when the pattern is block-diagonal and known at compile time.

**이 작업의 핵심 논지는 다음과 같습니다. 어텐션 패턴을 컴파일 시점에 알고 있다면 훨씬 빠른 것을 만들 수 있다.** 우리는 모든 Q 타일이 정확히 하나의 K/V 타일에만 어텐션한다는 고정 제약을 활용하고, 이 지식을 알고리즘 전체에 전파하여 여러 번 반복되던 누산기(accumulator)를 단일 GEMM으로 축약하고, 보정 단계를 제거하며, 보조 커널 실행을 없앱니다.
> **The central thesis of this work: when you know your attention pattern at compile time, you can build something much faster.** We exploit the fixed constraint that every Q tile attends to exactly one K/V tile, propagating this knowledge through the entire algorithm to collapse multi-iteration accumulators into single GEMMs, eliminate correction stages, and remove auxiliary kernel launches.

## 2. 왜 Block Attention인가? / Why Block Attention?

### 2.1 고정 블록 제약과 그로 인한 단순화의 연쇄 / The Fixed-Block Constraint and Its Cascade of Simplifications

표준 Flash Attention \[2\] 은 하나의 Q 타일을 여러 K/V 타일에 걸쳐 반복시키면서, 실행 중 통계량(행별 최댓값과 로그-합-지수)을 유지하고 각 단계마다 수치 안정성을 지키기 위한 보정 계수를 적용함으로써 임의 길이의 시퀀스를 처리합니다.
> Standard Flash Attention \[2\] handles sequences of arbitrary length by iterating a Q tile over multiple K/V tiles, maintaining running statistics (row-wise max and log-sum-exp) and applying a correction factor at each step to preserve numerical stability:

*리스팅 1: 다중 타일 반복과 온라인 소프트맥스 보정을 보여주는 표준 Flash Attention 내부 루프. / Listing 1: Standard Flash Attention inner loop showing multi-tile iteration and online softmax correction.*

```python
# Flash Attention inner loop (standard)
for k_tile in K_tiles:
    S = Q @ k_tile.T                   # partial scores
    m_new = max(m_old, rowmax(S))
    alpha = exp(m_old - m_new)         # correction factor
    O = alpha * O + exp(S - m_new) @ v_tile
    l = alpha * l + rowsum(exp(S - m_new))
O = O / l                              # final normalization
# Store L = m + log(l) to HBM for backward
```

이는 임의 시퀀스에 대해 정확하고 우아합니다. 하지만 고정된 64토큰 블록 크기의 블록 대각 어텐션에서는, Q 타일을 여러 K 타일에 걸쳐 도는 전체 루프가 **단 한 번의 반복**으로 축소됩니다. 모든 Q 타일과 그에 대응하는 K/V 타일은 동일한 타일입니다. 이 하나의 제약이 알고리즘 전체로 연쇄됩니다.
> This is correct and elegant for arbitrary sequences. But for block-diagonal attention with a fixed 64-token block size, the entire Q-tile-over-K-tiles loop is reduced to a **single iteration**. Every Q tile and its corresponding K/V tile are the same tile. That single constraint cascades through the algorithm:

1.  **다중 타일 반복이 없습니다.** 스코어 행렬 S = Q · Kᵀ ∈ ℝ^{64×64} 은 GEMM 한 번으로 완성됩니다. 상태를 유지하며 돌아야 할 루프가 없습니다.
2.  **온라인 소프트맥스 보정이 없습니다.** 타일이 하나뿐이므로 S에 대해 계산한 행별 최댓값과 합이 즉시 전역적으로 정확합니다. 보정 계수 α = exp(m\_old − m\_new) 는 항상 1이므로 완전히 제거할 수 있습니다.
3.  **로그섬익스프(L) 저장이 없습니다.** Flash Attention은 역전파에서 소프트맥스를 재계산할 수 있도록 행별 로그-합-지수 L을 HBM에 저장합니다. 타일이 하나면 역전파에서 보조 텐서 없이도 Q, K, V로부터 곧바로 P = softmax(S) 를 재계산할 수 있어, 순전파/역전파 쌍마다 발생하던 HBM 기록과 읽기를 통째로 제거합니다.
4.  **Di 전처리 커널이 없습니다.** 표준 Flash Attention 역전파는 본 역전파 이전에 Di = rowsum(dO ⊙ O) 를 계산하기 위한 별도 커널을 실행합니다. TLX Block Attention에서는 Di를 dP/dS 역전파 단계 안에서 **인라인(inline)** 으로 계산하여 커널 실행 한 번과 그에 딸린 메모리 트래픽을 없앱니다.
5.  **재스케일링을 동반한 출력 누산이 없습니다.** 타일이 하나면 출력 O = P · V 는 여러 부분 결과를 재스케일링해 누산한 것이 아니라 단일 GEMM에서 나온 새 결과입니다. 이 덕분에 모든 async\_dot 호출에서 `use_acc=False` 를 쓸 수 있으며, 이는 호출 간 TMEM 누산기를 보존할 필요가 없다고 텐서 코어 하드웨어에 알려 누산기를 자유롭게 재사용하게 합니다.

> 1.  **No multi-tile iteration.** The score matrix S = Q · Kᵀ ∈ ℝ^{64×64} is complete after one GEMM. There is no loop to maintain state across.
> 2.  **No online softmax correction.** Since there is only one tile, the row-wise max and sum computed over S are globally correct immediately. The correction factor α = exp(m\_old − m\_new) is identically 1 and can be dropped entirely.
> 3.  **No logsumexp (L) storage.** Flash Attention stores the per-row log-sum-exp L to HBM so that the backward pass can recompute softmax. With a single tile, the backward pass can recompute P = softmax(S) directly from Q, K, V without any auxiliary tensor — eliminating an entire HBM write and read per forward/backward pair.
> 4.  **No Di preprocessing kernel.** The standard Flash Attention backward launches a separate kernel to compute Di = rowsum(dO ⊙ O) before the main backward pass. In TLX Block Attention, Di is computed **inline** within the dP/dS backward stage, eliminating a kernel launch and its associated memory traffic.
> 5.  **No output accumulation with rescaling.** With a single tile, the output O = P · V is a fresh result from a single GEMM, not an accumulation of multiple rescaled partial results. This enables `use_acc=False` on all async\_dot calls — telling the tensor core hardware that the TMEM accumulator need not be preserved across calls, allowing it to be freely reused.

*리스팅 2: `use_acc=False` 는 타일 간 누산이 필요 없음을 하드웨어에 알려 TMEM 재사용을 가능하게 합니다. / Listing 2: `use_acc=False` signals to the hardware that no cross-tile accumulation is needed, enabling TMEM reuse.*

```python
# From the kernel: use_acc=False signals no accumulation needed
tlx.async_dot(
    q_tile[buff_idx],
    k_tile_T,
    TMEMqk[tmem_idx],
    use_acc=False,           # Fresh result — no accumulation
    mBarriers=[qk_SMEM_free[buff_idx], qk_TMEM_full[tmem_idx]],
)
```

### 2.2 표준 Flash Attention과의 비교 / Comparison with Standard Flash Attention

다음 표는 알고리즘적 차이를 정리한 것입니다.
> The following table summarizes the algorithmic differences:

| 항목 / Aspect | 표준 Flash Attention / Standard Flash Attention | TLX Block Attention |
|---|---|---|
| Q 타일당 K 타일 수 | 여러 개(전체 시퀀스) | 정확히 1개(같은 블록) |
| 스코어 행렬 | 여러 타일을 누산 | 단일 [64, 64] — 완성 |
| 로그섬익스프 L 텐서 | 역전파용으로 HBM에 저장 | 필요 없음 |
| 실행 중 최댓값/합 | 타일 간 유지 | 한 번 계산 후 레지스터에서 소비 |
| 보정 계수 α | 반복마다 필요 | 필요 없음(제거) |
| 출력 누산 | 재스케일링을 동반한 점진적 누산 | 단일 P·V GEMM |
| `use_acc` 모드 | True(타일 간 누산) | False(새 결과) |
| Di 전처리 | 별도 커널 실행 | 인라인 계산 |

*표 1: 표준 Flash Attention과 TLX Block Attention의 알고리즘적 차이. / Table 1: Algorithmic differences between standard Flash Attention and TLX Block Attention.*

이는 미세 최적화가 아니라 알고리즘 단계 전체의 제거를 의미합니다. 특히 역전파가 상당한 이점을 얻습니다. 저장된 L 텐서가 없으면 배치 × 헤드 × 시퀀스마다 발생하던 HBM 왕복이 제거되고, 인라인 Di 계산은 커널 실행과 그에 딸린 드라이버 오버헤드 및 메모리 대역폭을 제거합니다.
> These are not micro-optimizations — they represent the elimination of entire algorithmic stages. The backward pass in particular benefits substantially: the absence of a stored L tensor removes a round-trip through HBM per batch × heads × sequence, and inline Di computation removes a kernel launch with its associated driver overhead and memory bandwidth.

## 3. 커널 아키텍처: 워프 특화 파이프라인 / Kernel Architecture: A Warp-Specialized Pipeline

### 3.1 TLX

우리가 저작 프레임워크로 Triton을 선택한 이유는, 아래에서 설명할 워프 특화 파이프라인 구조에 자연스럽게 대응되는 Python 네이티브의 타일 지향 프로그래밍 모델을 제공하면서도, 순수 CUDA나 CUTLASS의 상용구(boilerplate)를 피하고 컴파일러 진화에 걸쳐 이식성을 유지하기 때문입니다. Triton의 TLX(Triton Language Extensions)는 async\_dot, local\_trans, 명시적 TMEM/SMEM 배리어 관리 같은 Blackwell 전용 프리미티브를, 하드웨어 제어와 개발자 생산성 사이에서 균형을 잡는 추상화 수준으로 더 노출합니다. 우리 경험상 TLX는 더 저수준의 대안과 대등하거나(종종 능가하는) 성능을 내면서도, Python 네이티브의 단순함 덕분에 훨씬 빠른 반복(iteration)을 가능하게 합니다.
> We chose Triton as the authoring framework because it provides a Python-native, tile-oriented programming model that maps naturally to the warp-specialized pipeline structure described below — while avoiding the boilerplate of raw CUDA or CUTLASS and remaining portable across compiler evolution. Triton’s TLX (Triton Language Extensions) further expose Blackwell-specific primitives like async\_dot, local\_trans, and explicit TMEM/SMEM barrier management at a level of abstraction that balances hardware control with developer productivity. In our experience, TLX delivers performance on par with (and often exceeding) lower-level alternatives while enabling significantly faster iteration due to its Python-native simplicity.

구체적으로 이 커널은 기본 Triton을 넘어서는 여러 TLX 프리미티브에 의존합니다. 명시적 누산기 제어와 함께 워프 특화 tcgen05 MMA 연산을 발행하는 tlx.async\_dot, TMA 기반 SMEM 채우기를 위한 tlx.async\_descriptor\_load, TMEM에서 레지스터로의 전송을 위한 tlx.local\_trans, 그리고 워프 그룹 간 생산자-소비자 파이프라인을 조율하는 mBarrier 동기화 모델이 그것입니다. 이 확장들은 [triton-ext 저장소](https://github.com/triton-lang/triton-ext)에서 제공됩니다.
> Specifically, this kernel relies on several TLX primitives that go beyond base Triton: tlx.async\_dot for issuing warp-specialized tcgen05 MMA operations with explicit accumulator control; tlx.async\_descriptor\_load for TMA-driven SMEM fills; tlx.local\_trans for TMEM-to-register transfers; and the mBarrier synchronization model that coordinates the producer-consumer pipeline across warp groups. These extensions are available in the [triton-ext repository](https://github.com/triton-lang/triton-ext).

### 3.2 워프 특화 / Warp Specialization

TLX Block Attention은 **워프 특화(warp specialization)** *\[8\]* 를 사용합니다. 같은 CTA 안의 서로 다른 워프가 각기 다른 하드웨어 유닛에 영구적으로 배정되어 커널의 생애 동안 서로 다른 코드 경로를 실행합니다. 이는 모든 워프가 같은 코드를 실행하고 조건문을 통해서만 갈라지는 전통적인 CUDA 모델과 대조됩니다.
> TLX Block Attention uses **warp specialization** *\[8\]* — different warps within the same CTA are permanently assigned to different hardware units and execute different code paths throughout the kernel’s lifetime. This contrasts with the traditional CUDA model where all warps execute the same code and diverge only through conditionals.

| 단계 / Stage | 워프 / Warps | 레지스터 / Registers | 하드웨어 유닛 / Hardware Unit | 역할 / Role |
|---|---|---|---|---|
| Load | 1 | 48 | TMA 엔진 | Q, K, V에 대한 `async_descriptor_load` |
| QK MMA | 1 | 48 | tcgen05 텐서 코어 | `async_dot(Q, Kᵀ)` → TMEMqk |
| Softmax | 4 | 120 | CUDA 코어 + SFU | 마스크 / 스케일 / exp2 / 정규화 → P를 SMEM으로 |
| PV MMA | 1 | 48 | tcgen05 텐서 코어 | `async_dot(P, V)` → TMEMpv |
| Epilogue | 8 | 200 | CUDA 코어 + L2 + TMA 엔진 | TMEM → 레지스터 → BF16 → SMEM → TMA 저장 |
| **합계 / Total** | **15** | — | — | **CTA당 480 스레드** |

*표 2: 순전파 파이프라인 단계 구성. 레지스터 할당은 의도적으로 비대칭입니다 — 하드웨어 가속 단계에는 최소한의 레지스터를, CUDA 코어 단계에는 가장 많은 레지스터를 줍니다. / Table 2: Forward pipeline stage configuration. Register allocations are deliberately asymmetric — hardware-accelerated stages receive minimal registers; CUDA core stages receive the most.*

```
그림 1 — 순전파 파이프라인 워프 타임라인 (개념도, 한 번의 반복):

Time →
Load     [─ TMA Q,K ─][─ TMA V ─]
QK MMA         [── async_dot Q·Kᵀ ──]
Softmax                  [── exp2/normalize → P ──]
PV MMA                            [── async_dot P·V ──]
Epilogue                                   [── local_load → BF16 → store ──]
```

![순전파 파이프라인 워프 타임라인 프로파일 / Forward pipeline warp timeline profile](/assets/blog/2026-05-26-tlx-block-attention-a-warp-specialized-blackwell-kernel-for-fixed-block-sparse-self-attention/unnamed-3-1.png){:style="width:100%"}

각 단계의 출력은 배리어에 신호를 보내 다음 단계를 풀어 주며, 이렇게 하드웨어 유닛에 걸친 생산자-소비자 파이프라인이 형성됩니다. Epilogue 워프가 타일 *i* 를 전역 메모리에 기록하는 동안 MMA 워프는 타일 *i+1* 을 계산하고 Load 워프는 TMA로 타일 *i+2* 를 가져옵니다 — 세 타일이 동시에 진행 중인 것입니다.
> Each stage’s output signals a barrier that unblocks the next stage, creating a producer-consumer pipeline across hardware units. While the Epilogue warp writes tile *i* to global memory, the MMA warps are computing tile *i+1*, and the Load warp is fetching tile *i+2* via TMA — three tiles in flight simultaneously.

### 3.3 루프라인(Roofline) 관점 / The Roofline Context

BLOCK\_D=64, HEAD\_DIM=128 에서 산술 강도(arithmetic intensity)는 약 33 FLOP/byte 로, B200의 리지 포인트(ridge point)인 약 281 FLOP/byte \[4\] 보다 한참 낮습니다. 이 커널은 설계상 메모리 대역폭에 종속(memory-bandwidth bound)됩니다. 그래서 TMA를 통한 지연 시간 은닉(latency hiding)과 불필요한 메모리 트래픽 최소화(제거된 L 텐서, 융합된 회전 임베딩)가 지배적인 최적화 지렛대가 됩니다.
> At BLOCK\_D=64, HEAD\_DIM=128, arithmetic intensity is ~33 FLOP/byte — well below the B200’s ridge point of ~281 FLOP/byte \[4\]. The kernel is memory-bandwidth bound by design. This is why latency hiding via TMA and minimizing unnecessary memory traffic (the eliminated L tensor, the fused rotary) are the dominant optimization levers.

### 3.4 버퍼 관리 / Buffer Management

하드웨어 유닛을 쉼 없이 바쁘게 유지하기 위해 커널은 삼중 버퍼링된 SMEM(3슬롯)과 이중 버퍼링된 TMEM(2슬롯)을 사용하며, 256 KB SMEM 예산 중 약 169 KB를 소비합니다. SMEM 슬롯이 3개이므로, MMA 워프가 타일 i+1 을 처리하고 Epilogue 워프가 타일 i 를 비우는 동안 Load 워프는 타일 i+2 를 미리 가져올 수 있습니다. 역전파 커널은 같은 256 KB 예산 안에서 추가 변화도 타일(gradient tile)을 수용하기 위해 이중 버퍼링 SMEM(약 162 KB)으로 낮춥니다.
> To keep hardware units continuously busy, the kernel uses triple-buffered SMEM (3 slots) and double-buffered TMEM (2 slots), consuming ~169 KB of the 256 KB SMEM budget. With three SMEM slots, the Load warp can prefetch tile i+2 while the MMA warp processes tile i+1 and the Epilogue warp drains tile i. The backward kernel drops to double-buffered SMEM (~162 KB) to accommodate additional gradient tiles within the same 256 KB budget.

## 4. 역전파: 로그섬익스프 텐서 없이 구하는 변화도 / The Backward Pass: Gradients Without the Logsumexp Tensor

표준 Flash Attention에서 역전파는 순전파가 로그섬익스프 텐서(L)를 고대역폭 메모리(HBM, High Bandwidth Memory)에 저장하도록 요구합니다. 이 텐서는 역전파 중 어텐션 확률(P)을 재구성하는 데 필요합니다. 또한 표준 어텐션은 Δᵢ(dO ⊙ out 의 행별 합)를 계산하기 위한 별도의 전처리 커널을 요구합니다.
> In standard Flash Attention, the backward pass requires the forward pass to save the logsumexp tensor (L) to High Bandwidth Memory (HBM). This tensor is necessary to reconstruct the attention probabilities (P) during the backward pass. Furthermore, standard attention requires a separate preprocessing kernel to compute Δᵢ (row-wise sum of dO ⊙ out).

블록 대각 어텐션은 64×64 스코어 행렬 전체를 단일 타일로 계산하므로, 두 요구 사항을 모두 완전히 우회할 수 있습니다. 역전파 커널은 어떤 로그섬익스프 텐서도 읽지 않으며, 별도의 전처리 단계도 요구하지 않습니다. 대신 S = Q · Kᵀ 와 P = softmax(S) 를 인라인으로 완전히 재계산합니다 — 타일이 한 번의 패스에 들어맞을 때는 저렴한 연산입니다.
> Because block-diagonal attention computes the entire 64×64 score matrix in a single tile, we can bypass both requirements completely. The backward kernel does not read any logsumexp tensor, nor does it require a separate preprocessing step. Instead, it fully recomputes S = Q · Kᵀ and P = softmax(S) inline — a cheap operation when the tile fits in a single pass.

이 단순화의 연쇄 덕분에 완전히 융합된 7단계 워프 특화 역전파 파이프라인을 구축할 수 있습니다.
> This cascade of simplifications allows us to build a fully fused, 7-stage warp-specialized backward pipeline:

| 단계 / Stage | 워프 / Warps | 레지스터 / Registers | 하드웨어 유닛 / Hardware Unit | 역할 / Role |
|---|---|---|---|---|
| Load | 1 | 48 | TMA 엔진 | Q, K, V, dO 로드 (+ 회전 임베딩용 sin/cos) |
| QK MMA | 1 | 48 | tcgen05 텐서 코어 | S = Q · Kᵀ 재계산 |
| Softmax/P | 4 | 120 | CUDA 코어 + SFU | P = softmax(S) 재계산 |
| dV MMA | 1 | 48 | tcgen05 텐서 코어 | dV = Pᵀ · dO |
| dP/dS | 4 | 120 | 텐서 코어 + CUDA 코어 | dP = dO · Vᵀ, Δᵢ, dS |
| dQ/dK MMA | 1 | 48 | tcgen05 텐서 코어 | dQ = dS · K, dK = dSᵀ · Q |
| Epilogue | 8 | 200 | CUDA 코어 + L2 + TMA 엔진 | dQ, dK, dV 저장 (+ 융합 회전 임베딩) |
| **합계 / Total** | **20** | — | — | **CTA당 640 스레드** |

*표 4: 7단계 역전파 파이프라인 구성. / Table 4: 7-stage backward pipeline configuration.*

역전파는 본질적으로 순전파보다 복잡합니다. 집약적인 연산 요구를 균형 있게 처리하려면 20개의 워프(CTA당 640 스레드)가 필요합니다. 가장 두드러지는 점은 SM의 256 KB 텐서 메모리(Tensor Memory)를 완전히 포화시킨다는 것입니다. 다섯 개의 구분된 TMEM 버퍼 — TMEMqk, TMEMdv, TMEMdp, TMEMdq, TMEMdk — 가 합쳐져 TMEM 사용률 100%에 도달합니다. 이를 수용하기 위해 역전파 커널은 순전파의 삼중 버퍼링 SMEM에서 이중 버퍼링 SMEM(약 162 KB / 256 KB, 63%)으로 낮추고, TMEM은 이중 버퍼링을 유지합니다.
> The backward pass is inherently more complex than the forward pass. It requires 20 warps (640 threads per CTA) to balance the intense computational requirements. Most notably, it fully saturates the 256 KB Tensor Memory on the SM. The five distinct TMEM buffers — TMEMqk, TMEMdv, TMEMdp, TMEMdq, and TMEMdk — collectively hit 100% TMEM utilization. To accommodate this, the backward kernel drops from triple-buffered SMEM in the forward pass to double-buffered SMEM (~162 KB / 256 KB, 63%), while keeping double-buffered TMEM.

## 5. 가변 길이 시퀀스를 위한 스케줄링 / Scheduling for Variable-Length Sequences

![이진 탐색을 이용한 지속(persistent) 커널 스케줄링 / Persistent kernel scheduling with binary search](/assets/blog/2026-05-26-tlx-block-attention-a-warp-specialized-blackwell-kernel-for-fixed-block-sparse-self-attention/unnamed-2-1.png){:style="width:100%"}

실제 추천 및 특징 상호작용 모델은 균일하게 정돈된 시퀀스 길이를 처리하지 않습니다. 오히려 트래픽은 하나의 평탄화된 버퍼로 묶인 들쭉날쭉한 가변 길이 시퀀스가 지배합니다. 시퀀스당 CTA 하나를 순진하게 매핑하면, 짧은 시퀀스가 먼저 끝나고 다른 SM이 긴 시퀀스를 처리하는 동안 일부 SM이 놀게 되어 심각한 워크로드 불균형이 생깁니다.
> Real-world recommendation and feature interaction models do not process neatly uniform sequence lengths. Instead, traffic is dominated by jagged, variable-length sequences packed into a single flattened buffer. Naively mapping one CTA per sequence would leave SMs idle when short sequences finish early while others process long sequences — a severe workload imbalance.

SM 점유율을 극대화하기 위해 커널은 `min(NUM_SMS, total_blocks)` 개의 지속(persistent) 프로그램을 실행합니다 — SM당 정확히 하나의 지속 스레드 블록입니다. 워크로드는 미리 계산된 두 개의 배열로 균형을 맞춥니다.
> To maximize SM occupancy, the kernel launches `min(NUM_SMS, total_blocks)` persistent programs — exactly one persistent thread block per SM. Workload is balanced across two precomputed arrays:

1.  **BLOCK\_PER\_BATCH**: 시퀀스당 64토큰 타일 개수의 접두사 합(prefix-sum).
2.  **BLOCK\_PER\_PROGRAM**: 각 SM에 배정된 균형 잡힌 타일 범위 — 누적 합이 아니라 닫힌 형태(closed-form)의 divmod 연산으로 계산.

> 1.  **BLOCK\_PER\_BATCH**: A prefix-sum of the number of 64-token tiles per sequence.
> 2.  **BLOCK\_PER\_PROGRAM**: The balanced tile ranges assigned to each SM — computed using closed-form divmod arithmetic rather than cumulative sums.

GPU 동기화 오버헤드를 제거하기 위해, CPU 측 오프셋 텐서를 사용할 수 있을 때(`cpu_offsets`)는 모든 스칼라 스케줄링 연산(타일 개수, divmod, 접두사 합)을 커널 실행 전에 CPU에서 계산합니다 — GPU 동기화 지점이 전혀 없습니다.
> To eliminate GPU synchronization overhead, when CPU-side offset tensors are available (`cpu_offsets`), all scalar scheduling arithmetic (tile counts, divmod, prefix sums) is computed on the CPU before the kernel launches — zero GPU sync points.

커널 내부에서 각 SM은 주어진 전역 타일 인덱스가 어느 시퀀스(배치 인덱스)에 속하는지 판별해야 합니다. 이는 정확히 32번의 반복으로 실행되는(합리적인 어떤 배치 크기에도 충분함) 분기 없는(branchless) 이진 탐색으로 이루어지며, 스레드 동기화가 전혀 없습니다.
> Inside the kernel, each SM must determine which sequence (batch index) a given global tile index belongs to. This uses a branchless binary search that executes in exactly 32 iterations (sufficient for any reasonable batch size) with zero thread synchronization.

## 6. 융합된 회전 임베딩 역전파: 더 높은 정밀도와 더 빠른 속도 / Fused Rotary Backward: Higher Precision at Higher Speed

셀프 어텐션 계층에서는 셀프 어텐션 앞에 사영(projection)과 사인파(sinusoidal) 연산 \[6\] 이 놓입니다. 역전파에서는 이것이 어텐션 역전파 → 사인파 순서가 되며, 관례적으로 서로 다른 두 번의 커널 실행으로 처리됩니다.
> For self attention layers, self attention is preceded by projection + sinusoidals \[6\]. In the backward pass this becomes attention backward -> sinusoidals which conventionally happen with 2 different kernel launches.

### 6.1 기준선: 두 커널 역전파 / Baseline: Two-Kernel Backward Pass

![기준선 대비 융합 역전파 커널 비교 / Comparison of baseline versus fused backward kernels](/assets/blog/2026-05-26-tlx-block-attention-a-warp-specialized-blackwell-kernel-for-fixed-block-sparse-self-attention/unnamed-1-1.png){:style="width:100%"}

관례적인 역전파는 서로 분리된 두 번의 커널 실행을 요구합니다.
> The conventional backward pass requires two separate kernel launches:

1.  어텐션 역전파 커널 — dQ, dK, dV를 텐서 코어로 FP32로 누산한 뒤, 전역 메모리에 저장할 때 BF16으로 절단(truncate)합니다.
2.  회전 임베딩 역전파 커널 — BF16 변화도를 전역 메모리에서 다시 읽어 회전 켤레(rotary conjugate) R(−θ)를 적용하고, 최종 BF16 결과를 저장합니다.

> 1.  Attention Backward Kernel — accumulates dQ, dK, dV in FP32 via tensor cores, then truncates to BF16 on store to global memory.
> 2.  Rotary Backward Kernel — reloads the BF16 gradients from global memory, applies the rotary conjugate R(−θ), and stores the final BF16 result.

이 분리에는 세 가지 비용이 따릅니다.
> This separation has three costs:

| 문제 / Problem | 영향 / Impact |
|---|---|
| **정밀도 손실** | FP32 변화도가 회전 변환 *이전에* BF16으로 절단되고, 최종 저장 시 다시 절단됩니다. 두 개의 양자화 지점이 각각 약 0.4%의 상대 오차를 주입합니다(BF16은 가수 비트가 7개뿐). 이후의 사영 GEMM이 누적된 오차를 증폭시킵니다. |
| **메모리 대역폭 낭비** | dQ, dK, dV가 기록된 뒤 곧바로 다시 읽힙니다 — [total\_seq\_len, 1152] 텐서(head\_dim=128, KV 헤드 3개)에 대한 완전한 왕복입니다. 시퀀스 길이가 수백만에 이르면 이 트래픽은 상당합니다. |
| **커널 실행 오버헤드** | 한 번이면 충분한 것을 두 번 디스패치합니다. |

### 6.2 융합 방식 / Fused Approach

어텐션 역전파 커널은 이미 변화도 저장 에필로그에 하나의 워프 그룹을 전담시키고 있습니다. 우리는 변화도가 아직 FP32 레지스터에 있는 동안 그 에필로그에 회전 켤레를 주입해 이를 활용합니다.
> The attention backward kernel already dedicates a single warp group to the gradient store epilogue. We take advantage of this by injecting the rotary conjugate into that epilogue, while gradients are still in FP32 registers:

1.  텐서 코어가 dQ, dK, dV를 FP32(TMEM)로 저장합니다.
2.  FP32 값을 레지스터로 로드합니다.
3.  R(−θ)를 완전한 FP32 정밀도로 적용합니다 — 가벼운 sin/cos 로드와 요소별 곱셈입니다.
4.  BF16으로 캐스팅하고 단일 전역 저장을 발행합니다.

> 1.  Tensor cores store dQ, dK, dV in FP32 (TMEM).
> 2.  Load FP32 values into registers.
> 3.  Apply R(−θ) in full FP32 precision — a lightweight sin/cos load + element-wise multiply.
> 4.  Cast to BF16 and issue a single global store.

단계별 비교는 다음과 같습니다.
> The per-step comparison:

| 항목 / Aspect | 기준선(분리) / Baseline (Separate) | 융합 커널 / Fused Kernel |
|---|---|---|
| 어텐션 역전파 연산 | FP32 | FP32 |
| 중간 저장 | BF16 → 전역 메모리 | FP32 레지스터 |
| 회전 임베딩 sin/cos 연산 | BF16 | FP32 |
| BF16 양자화 지점 | 2 | 1(최종 저장만) |
| 전역 메모리 왕복 | 2 | 0 |
| 커널 실행 | 2 | 1 |

역전파 에필로그에서의 융합된 회전 켤레. 인터리브(interleave) 연산은 아직 FP32인 상태에서 쌍을 이룬 \[cos, sin\] 성분에 R(−θ)를 적용합니다.
> Fused rotary conjugate in the backward epilogue. The interleave operation applies R(−θ) to paired \[cos, sin\] components while still in FP32.

```python
# Apply rotary conjugate to dV (neg_sin handles the conjugate)
dv0, dv1 = dvLocal.reshape(BLOCK_D, HALF_DIM, 2).split()
dvLocal = tl.interleave(
    dv0 * cos_local - dv1 * neg_sin,
    dv1 * cos_local + dv0 * neg_sin,
)
```

## 7. 성능 결과 / Performance Results

모든 벤치마크는 NVIDIA B200 GPU(x86 CPU)에서 BF16 정밀도로 수행되었습니다. 기본 구성은 B=1152 시퀀스, HEAD\_DIM=128, H=4 헤드, max\_seq\_len=2000, sparsity=0.7 – 이산 균등 분포(discrete uniform)를 사용합니다(운영 트래픽 분포를 대표함).
> All benchmarks were conducted on NVIDIA B200 GPUs (x86 cpu) with BF16 precision. The primary configuration uses B=1152 sequences, HEAD\_DIM=128, H=4 heads, max\_seq\_len=2000, and sparsity=0.7 – discrete uniform (representative of production traffic distributions).

### 7.1 커널 수준 속도 향상 / Kernel-Level Speedup

| 패스 / Pass | 블록 어텐션 적용 Flash Attention v2 (ms) / Flash Attention v2 with block attention (ms) | TLX Block Attention (ms) | 속도 향상 / Speedup |
|---|---|---|---|
| 순전파 / Forward | 1.81 | 0.98 | **1.85×** |
| 역전파 / Backward | 5.89 | 2.36 | **2.50×** |
| **합계 / Total** | **7.70** | **3.33** | **2.31×** |

*표 5: 커널 수준 성능 비교(B=1152, D=128, H=4, BF16, B200, max\_seq\_len=2000, sparsity=0.7). / Table 5: Kernel-level performance comparison (B=1152, D=128, H=4, BF16, B200, max\_seq\_len=2000, sparsity=0.7).*

역전파 속도 향상(2.50배)이 순전파 속도 향상(1.85배)보다 큰 주된 이유는, 역전파가 서로 독립적인 두 가지 단순화의 이점을 받기 때문입니다. (1) 제거된 로그섬익스프 저장과 Di 전처리, 그리고 (2) 표준 Flash Attention 역전파가 요구하는 L 텐서 HBM 왕복을 피하는 인라인 P 재계산이 그것입니다.
> The backward speedup (2.50×) is larger than the forward speedup (1.85×) primarily because the backward pass benefits from two independent simplifications: (1) eliminated logsumexp storage and Di preprocessing, and (2) inline P recomputation that avoids the L-tensor HBM round-trip that standard Flash Attention backward requires.

### 7.2 워크로드에 따른 확장성 / Scaling Across Workloads

![시퀀스 길이와 희소성 비율에 따른 속도 향상 확장성 / Speedup scaling across sequence lengths and sparsity ratios](/assets/blog/2026-05-26-tlx-block-attention-a-warp-specialized-blackwell-kernel-for-fixed-block-sparse-self-attention/unnamed-9.png){:style="width:100%"}

*표 6: 시퀀스 길이와 희소성 비율에 따른 확장 성능. 속도 향상은 분포 형태와 무관하게 일관적입니다(batch=1152, >7000 인 경우 batch=768). Flash Attention v2(jfa) 대비 커널 속도 향상. / Table 6: Scaling performance across sequence lengths and sparsity ratios. Speedups are consistent regardless of distribution shape (batch=1152, for >7000 batch=768). Kernel speed up over flash attention v2 (jfa).*

### 7.3 융합된 회전 임베딩 역전파 / Fused Rotary Backward

회전 임베딩 역전파를 어텐션 에필로그에 융합했을 때의 효과는 특히 두드러집니다.
> The impact of fusing rotary backward into the attention epilogue is particularly striking:

| 구성 / Configuration | 시간 / Time (ms) |
|---|---|
| 어텐션 역전파(단독) / Attention backward (standalone) | 1.556 |
| 회전 임베딩 역전파(단독) / Rotary backward (standalone) | 4.880 |
| **비융합 합계 / Unfused total** | **6.436** |
| **융합 attention_rotary 역전파 / Fused attention_rotary backward** | **1.819** |
| **속도 향상 / Speedup** | **3.54×** |

*표 7: 융합 대 비융합 회전 임베딩 역전파의 시간 분해. 단독 회전 임베딩 커널이 비융합 합계를 지배합니다. seq\_len=1735537, heads=3, head\_dim=128, batch=1152. / Table 7: Fused vs. unfused rotary backward timing breakdown. The standalone rotary kernel dominates the unfused total. seq\_len=1735537, heads=3, head\_dim=128, batch=1152.*

단독 회전 임베딩 역전파는 어텐션 역전파 자체보다 3배 넘게 비쌉니다 — 이는 순수하게 메모리 대역폭에 종속되며, 의미 있는 연산 없이 [M, D] 텐서를 읽고 씁니다. 이를 어텐션 에필로그에 융합하면 그 대역폭 비용이 기존 TMEM → 레지스터 파이프라인에 분산 상환되어, 합쳐진 연산이 6.436 ms에서 1.819 ms로 줄어듭니다.
> The standalone rotary backward is more than 3× more expensive than the attention backward itself — it is purely memory-bandwidth bound, reading and writing \[M, D\] tensors with no meaningful compute. Fusing it into the attention epilogue amortizes this bandwidth cost over the existing TMEM → register pipeline, reducing the combined operation from 6.436 ms to 1.819 ms.

종단 간(end-to-end)으로 이 커널을 셀프 어텐션 계층에 통합하면 해당 계층에서 **모델 FLOPs 활용률(MFU, Model FLOPs Utilization)이 +30.6% 향상**됩니다.
> End-to-end, integrating this kernel into self-attention layers results in a **+30.6% Model FLOPs Utilization (MFU) gain** on those layers.

### 7.4 수치 정확도 / Numerical Accuracy

회전 임베딩 역전파를 FP32 에필로그에 융합하면 측정 가능한 정확도 개선도 얻습니다. 고정밀 PyTorch 레퍼런스와 비교했을 때, TLX Block Attention은 쿼리 변화도(dQ)의 최대 변화도 오차를 2배 이상 줄입니다.
> Fusing the rotary backward into the FP32 epilogue also yields measurable accuracy improvements. Comparing against a high-precision PyTorch reference, TLX Block Attention reduces the maximum gradient error in the query gradients (dQ) by over 2×:

| 지표 / Metric | Flash Attention v2 | TLX Block Attention | 더 정확한 쪽 / More Accurate |
|---|---|---|---|
| 최대 dQ 차이 / Max dQ diff | 0.2559 | 0.1201 | **TLX** |
| 최대 dK 차이 / Max dK diff | 0.1689 | 0.1689 | 동률 / Tie |
| 최대 dV 차이 / Max dV diff | 0.0112 | 0.0112 | 동률 / Tie |
| 평균 dQ 차이 / Avg dQ diff | 0.000309 | 0.000220 | **TLX** |

*표 8: PyTorch 레퍼런스 구현 대비 변화도 수치 정확도. TLX Block Attention은 단일 양자화 지점의 융합 회전 임베딩 경로 덕분에 최대 dQ 오차를 53% 줄입니다. / Table 8: Gradient numerical accuracy against a PyTorch reference implementation. TLX Block Attention reduces max dQ error by 53% due to the single-quantization-point fused rotary path.*

dQ가 가장 큰 이점을 받는 이유는, 쿼리 변화도(dQ = dS · K)가 양자화 지점 2개가 아니라 1개만 거쳐 융합 회전 켤레를 통과하기 때문입니다. dK도 회전 켤레를 통과하지만(RoPE는 Q와 K를 모두 회전시킴), dK의 최대 절대 오차는 회전 임베딩의 메모리 왕복보다 MMA 누산 자체에 의해 지배됩니다. 그래서 중간 BF16 캐스트 제거로 인한 요소별 개선이 최댓값에서는 드러나지 않습니다.
> dQ benefits most because the query gradient (dQ = dS · K) flows through the fused rotary conjugate with 1 quantization point instead of 2. dK also passes through the rotary conjugate (RoPE rotates both Q and K), but its maximum absolute error happens to be dominated by the MMA accumulation itself rather than the rotary memory round-trip, so the per-element improvement from eliminating the intermediate BF16 cast does not surface at the maximum.

## 8. 적용 가능성 / Applicability

모델이 블록 대각 어텐션 — 각 토큰이 고정된 지역 그룹 안의 다른 토큰에만 어텐션하는 방식 — 을 사용한다면, 이 커널은 곧바로 들어맞습니다.
> If your model uses block-diagonal attention — where each token attends only to others within a fixed local group — this kernel is a direct fit.

-   **NVIDIA Blackwell GPU에서의 학습.** 이 커널은 tcgen05 MMA 명령, TMEM 할당, Blackwell 세대 TMA 디스크립터를 사용하며, 이들 중 어느 것도 Ampere나 Hopper에는 존재하지 않습니다. async\_dot / local\_trans / tlx API는 Blackwell 아키텍처(sm\_100+)를 구체적으로 겨냥합니다.
-   **HEAD\_DIM ∈ {64, 128}.** 지원되는 헤드 차원은 이 값들이며, 다른 값은 재컴파일과 잠재적으로 새로운 SMEM/TMEM 예산 계산을 요구합니다.

> -   **Training on NVIDIA Blackwell GPUs.** The kernel uses tcgen05 MMA instructions, TMEM allocation, and Blackwell-era TMA descriptors — none of which exist on Ampere or Hopper. The async\_dot / local\_trans / tlx APIs target the Blackwell architecture (sm\_100+) specifically.
> -   **HEAD\_DIM ∈ {64, 128}.** These are the supported head dimensions; other values require recompilation and potentially new SMEM/TMEM budget calculations.

## 9. 결론 / Conclusion

TLX Block Attention은 하나의 아키텍처적 제약이 갖는 복리적(compounding) 위력을 보여 줍니다. 넓은 부류의 특징 상호작용 및 시퀀스 모델이 엄격한 블록 대각 어텐션만을 필요로 한다는 점을 인식하면, 단순화의 연쇄가 가능해집니다.
> TLX Block Attention demonstrates the compounding power of a single architectural constraint. By recognizing that a broad class of feature interaction and sequence models only require strict block-diagonal attention, a cascade of simplifications becomes possible.

블록 간 어텐션을 제거하면 다중 타일 누산이 사라집니다. 다중 타일 누산이 없으면 온라인 소프트맥스 보정 계수가 사라집니다. 온라인 소프트맥스 보정이 없으면 역전파에서 로그섬익스프 텐서를 통째로 버릴 수 있습니다. 별도의 로그섬익스프 텐서가 없으면 회전 임베딩을 역전파 에필로그에 완전히 융합할 만큼의 레지스터와 메모리 대역폭 예산이 확보되며, 이는 독립적으로 속도와 수치 정확도를 함께 개선합니다.
> Eliminating cross-block attention means no multi-tile accumulation. No multi-tile accumulation means no online softmax correction factors. No online softmax correction means the logsumexp tensor can be discarded entirely in the backward pass. No separate logsumexp tensor frees enough register and memory bandwidth budget to fully fuse the rotary embeddings directly into the backward epilogue, which independently improves both speed and numerical accuracy.

그 결과물은 Blackwell 아키텍처의 TMA 및 TMEM 하드웨어 프리미티브에 완벽하게 맞춘 워프 특화 커널입니다. 순전파에 15개, 역전파에 20개의 워프를 두고, 각 워프 그룹은 자신의 병목에 맞는 하드웨어 유닛에 영구적으로 배정됩니다. 이 설계는 Flash Attention v2 대비 2.3배의 커널 수준 속도 향상, 회전 임베딩을 융합했을 때 3.5배의 합산 역전파 속도 향상, 그리고 운영 셀프 어텐션 계층에서 +30.6%의 MFU 향상을 달성합니다.
> The result is a warp-specialized kernel perfectly tailored for the Blackwell architecture’s TMA and TMEM hardware primitives: 15 warps in the forward pass, 20 in the backward, each warp group permanently assigned to the hardware unit that matches its bottleneck. This design achieves 2.3× kernel-level speedups over Flash Attention v2, a 3.5× combined backward speedup when rotary is fused, and a +30.6% MFU gain on production self-attention layers.

이 커널은 [github.com/facebookresearch/ads\_model\_kernel\_library](https://github.com/facebookresearch/ads_model_kernel_library) 에서 오픈소스로 공개되어 있습니다 — 여러분의 블록 희소 어텐션 워크로드에서 직접 시험해 보고 어떤 결과를 얻었는지 알려 주세요.
> The kernel is open-source at [github.com/facebookresearch/ads\_model\_kernel\_library](https://github.com/facebookresearch/ads_model_kernel_library) — try it on your own block-sparse attention workloads and let us know what you find.

## 감사의 말 / Acknowledgements

저자들은 이 커널을 가능하게 한 [tlx Blackwell 확장](https://github.com/triton-lang/triton-ext)을 지속적으로 개발해 준 Triton \[5\] 및 PyTorch 팀에 감사드립니다. Flash Attention, 워프 특화 파이프라인, 지속 커널 스케줄링에 대한 작업으로 이러한 최적화의 토대를 마련해 준 더 넓은 GPU 커널 연구 커뮤니티에 특별히 감사드립니다.
> The authors thank the Triton \[5\] and PyTorch teams for their continued development of the [tlx Blackwell extension](https://github.com/triton-lang/triton-ext) that made this kernel possible. Special thanks to the broader GPU kernel research community whose work on Flash Attention, warp-specialized pipelines, and persistent kernel scheduling provided the foundation for these optimizations.

## 참고 문헌 / References

1.  Qiu, J., Ma, H., Levy, O., Yih, S. W., Wang, S., & Tang, J. (2020). BlockBERT: Efficient Attention Using Block Structures. EMNLP Findings 2020. [https://arxiv.org/abs/1911.02972](https://arxiv.org/abs/1911.02972)
2.  Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022. [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
3.  Dao, T. (2024). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ICLR 2024. [https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)
4.  NVIDIA Corporation. (2024). NVIDIA Blackwell Architecture Technical Brief. [https://resources.nvidia.com/en-us-blackwell-architecture](https://resources.nvidia.com/en-us-blackwell-architecture)
5.  Tillet, P., Kung, H. T., & Cox, D. (2019). Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. MAPL 2019. [https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
6.  Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
7.  He, H. & Guessous, D. (2024). FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention. PyTorch Blog. [https://pytorch.org/blog/flexattention/](https://pytorch.org/blog/flexattention/)
8.  Yu, H., Ren, M., Maher, B., Nay, S., Zhu, G., & Jiang, S. (2024). Enabling Advanced GPU Features in PyTorch – Warp Specialization. PyTorch Blog. [https://pytorch.org/blog/warp-specialization/](https://pytorch.org/blog/warp-specialization/)
