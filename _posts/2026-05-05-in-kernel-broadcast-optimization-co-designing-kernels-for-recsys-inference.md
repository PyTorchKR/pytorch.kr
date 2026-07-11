---
layout: blog_detail
title: "커널 내 브로드캐스트 최적화(IKBO): RecSys 추론을 위한 커널 공동 설계"
author: Jian Jiao, Boda Li, Hongtao Yu, Yuanwei (Kevin) Fang, Zhengkai Zhang, Zhuoran Zhao, Yuxin Chen, Sijia Chen†, Yang Chen†, Zijian Shen, Shuyao Bi, Ao Cai, Junhan Hu†, Shuqi Yang†, Wei Wei, Lu Fang, Rengan Xu, Manman Ren, Alex Zhong, Xiaohan Wei, Zeliang Chen, Ellie Wen, Wenlin Chen
category: ["pytorch.org", "translation"]
org_title: "In-Kernel Broadcast Optimization: Co-Designing Kernels for RecSys Inference"
org_link: https://pytorch.org/blog/in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/
---

**TL;DR:**

- 전통적인 RecSys 추론은 공유되는 사용자 임베딩/시퀀스를 모든 후보(candidate)마다 명시적으로 복제합니다. **커널 내 브로드캐스트 최적화(In-Kernel Broadcast Optimization, IKBO)** 는 브로드캐스트 로직을 사용자-후보 상호작용 커널에 직접 융합하는 **커널-모델-시스템 공동 설계(co-design)** 를 통해 이 오버헤드를 제거합니다. 메모리 사용량과 IO 사용률을 모두 줄임으로써, IKBO는 더 높은 처리량을 이끌어냅니다.
- IKBO는 연산 집약적인 순(net) 지연 시간을 최대 **2/3까지 감소**시키며, [Meta Adaptive Ranking Model](https://engineering.fb.com/2026/03/31/ml-applications/meta-adaptive-ranking-model-bending-the-inference-scaling-curve-to-serve-llm-scale-models-for-ads/)을 구동하는 요청 중심(request-centric)의 추론 효율적 프레임워크에 **확장성의 근간(scalability backbone)** 역할을 합니다.
- Meta의 다단계 추천 퍼널(recommendation funnel) 전반에 걸쳐, GPU와 [MTIA](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/)(Meta Training and Inference Accelerator) 양쪽에서 종단 간(end-to-end)으로 배포되었습니다.
- IKBO 선형 압축(Linear Compression) 커널은 네 단계에 걸친 점진적 공동 설계 끝에 [TLX](https://github.com/facebookexperimental/triton/tree/tlx)를 통한 워프 특화(warp-specialized) 융합으로 마무리되어, H100 SXM5에서 누적 **약 4배**의 속도 향상을 달성했습니다.
- IKBO 공동 설계는 Flash Attention 커널을 IO 바운드(IO-bound)에서 연산 바운드(compute-bound)로 전환시켰습니다(H100 SXM5에서 **621 BF16 TFLOPs** 달성). [TLX](https://github.com/facebookexperimental/triton/tree/tlx) 워프 특화 최적화와 결합하면, 공동 설계되지 않은 CuTeDSL FA4 Hopper 기준선(baseline) 대비 **2.4배/6.4배**의 처리량 향상을 얻습니다(커널만 / 커널 + 브로드캐스팅 기준).

> TL;DR:
>
> - Traditional RecSys inference explicitly replicates shared user embeddings/sequences for every candidate. **In-Kernel Broadcast Optimization** (IKBO) eliminates this overhead via a **kernel-model-system co-design** that fuses broadcast logic directly into user-candidate interaction kernels. By decreasing both the memory footprint and IO utilization, IKBO unlocks even higher throughput.
> - IKBO delivers up to a **2/3 reduction** in compute-intensive net latency, serving as the **scalability backbone** for the request-centric, inference-efficient framework that powers the [Meta Adaptive Ranking Model](https://engineering.fb.com/2026/03/31/ml-applications/meta-adaptive-ranking-model-bending-the-inference-scaling-curve-to-serve-llm-scale-models-for-ads/).
> - Deployed end-to-end across Meta’s multi-stage recommendation funnel on both GPU and [MTIA](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/) (Meta Training and Inference Accelerator).
> - The IKBO Linear Compression kernel achieved a cumulative **~4×** speedup on H100 SXM5 after four stages of progressive co-design, culminating in warp-specialized fusion via [TLX](https://github.com/facebookexperimental/triton/tree/tlx).
> - The IKBO co-design shifted the Flash Attention kernel from IO-bound to compute-bound (hitting **621 BF16 TFLOPs** on H100 SXM5). Coupled with [TLX](https://github.com/facebookexperimental/triton/tree/tlx) warp-specialized optimization, this results in a **2.4x/6.4×** throughput gain over the non-co-designed CuTeDSL FA4 Hopper baseline (kernel only/kernel + broadcasting).

이번 글에서는 추천 모델 추론에서 중복되는 사용자 임베딩 브로드캐스트를 제거하는 커널-모델-시스템 공동 설계 접근법인 커널 내 브로드캐스트 최적화(In-Kernel Broadcast Optimization, IKBO)를 소개합니다. 프로덕션 RecSys에서 사용자 임베딩은 주어진 요청의 모든 후보에 대해 동일하지만, 표준적인 방식은 명시적인 복제를 요구하여 후보 수에 비례해 늘어나는 메모리 대역폭과 연산을 낭비합니다. IKBO는 간단한 통찰을 담고 있습니다. 브로드캐스트는 연산적 필연이 아니라 데이터 레이아웃의 문제라는 것입니다. 각 IKBO 커널은 사용자 입력과 후보 입력을 서로 맞지 않는 자연스러운 배치 크기(batch size) 그대로 받아 브로드캐스트를 내부에서 처리하므로, 복제된 텐서(tensor)가 실제로 만들어지는 일이 없습니다. 이 방법론을 선형 압축(Linear Compression)과 Flash Attention이라는 두 개의 커널 심층 분석을 통해 소개합니다.
> In this post, we present In-Kernel Broadcast Optimization (IKBO), a kernel-model-system co-design approach that eliminates redundant user-embedding broadcast in recommendation model inference. In production RecSys, user embeddings are identical across all candidates for a given request, yet standard approaches require explicit replication, wasting memory bandwidth and compute that scale with candidate count. IKBO encodes a simple insight: broadcast is a data layout concern, not a computational necessity. Each IKBO kernel accepts user and candidate inputs at their natural, mismatched batch sizes and handles broadcast internally, so no replicated tensors ever materialize. We showcase the methodology through two kernel deep dives: Linear Compression and Flash Attention.

Meta의 RecSys 추론 스택 전반, 즉 초기 단계부터 후기 단계 랭킹(ranking) 모델까지 GPU와 MTIA(Meta Training and Inference Accelerator) 양쪽에 배포된 IKBO는, 공동 설계된 모델에서 연산 집약적인 순(net) 지연 시간을 최대 2/3까지 감소시킵니다. IKBO는 [Meta Adaptive Ranking Model](https://engineering.fb.com/2026/03/31/ml-applications/meta-adaptive-ranking-model-bending-the-inference-scaling-curve-to-serve-llm-scale-models-for-ads/)(프로덕션에서 LLM 규모 모델을 서빙)을 뒷받침하는 요청 중심의 추론 효율적 프레임워크에 확장성의 근간 역할을 합니다. H100 SXM5에서 IKBO 선형 압축 커널은 네 단계의 점진적 공동 설계, 즉 행렬곱 분해(matmul decomposition), 메모리 정렬(memory alignment), 브로드캐스트 융합(broadcast fusion), 그리고 [TLX(Triton Low-Level Extensions)](https://github.com/facebookexperimental/triton/tree/tlx)를 통한 워프 특화 다단계 융합(warp-specialized multi-stage fusion)을 거쳐 약 4배의 속도 향상을 달성합니다. Flash Attention의 경우, IKBO는 621 BF16 TFLOPs로 공동 설계되지 않은 CuTeDSL FA4-Hopper 대비 2.4배/6.4배의 처리량을 제공합니다(커널만 / 커널 + 브로드캐스팅 기준). 복제를 우회하는 시스템 수준 브로드캐스트나 넷 분할(net-splitting)과 달리, IKBO는 연산 프리미티브(primitive) 계층에서 복제를 제거하여, 거의 독립적인 비용으로 밀집 상호작용(dense interaction) 수준의 품질을 달성합니다.
> Deployed across Meta’s RecSys inference stack—from early-stage to late-stage ranking models, spanning both GPU and MTIA (Meta Training and Inference Accelerator)—IKBO delivers up to a 2/3 reduction in compute-intensive net latency on co-designed models. It serves as the scalability backbone for the request-centric, inference-efficient framework underlying the [Meta Adaptive Ranking Model](https://engineering.fb.com/2026/03/31/ml-applications/meta-adaptive-ranking-model-bending-the-inference-scaling-curve-to-serve-llm-scale-models-for-ads/) (serving LLM-scale models in production). On H100 SXM5, our IKBO Linear Compression kernel achieves ~4× speedup through four progressive co-design stages: matmul decomposition, memory alignment, broadcast fusion, and warp-specialized multi-stage fusion via [TLX (Triton Low-Level Extensions)](https://github.com/facebookexperimental/triton/tree/tlx). For Flash Attention, IKBO delivers a 2.4×/6.4× throughput compared to non-co-designed CuTeDSL FA4-Hopper (kernel only / kernel + broadcasting) with 621 BF16 TFLOPs. Unlike system-level broadcast or net-splitting that work around replication, IKBO eliminates it at the computational primitive layer, achieving dense interaction quality at near-independent cost.

코드 저장소: [https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/ikbo](https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/ikbo)
> Code Repository: [https://github.com/pytorch/FBGEMM/tree/main/fbgemm\_gpu/experimental/ikbo](https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/ikbo)

† *Meta 재직 중 수행한 작업입니다.*
> † *Work done while at Meta*

## 1. 커널 내 브로드캐스트 최적화(IKBO): 메모리 및 연산 중복 제거 / In-Kernel Broadcast Optimization: Eliminating Memory and Compute Redundancy

사용자가 자신의 피드를 열면, 추천 시스템은 무엇을 보여줄지 결정하기 위해 수백에서 수천 개의 후보 항목(candidate item)에 점수를 매겨야 합니다. 모델의 입력은 두 범주로 나뉩니다. 요청의 모든 후보에 대해 동일한 **사용자 특징(user features)** (예: 열람 이력, 프로필, 컨텍스트)과, 각 항목마다 고유한 **후보 특징(candidate features)** (예: 항목 ID, 카테고리, 참여 통계)입니다. 둘 다 임베딩 조회(embedding lookup)와 후속 처리를 거쳐 임베딩 표현(representation)을 만들어냅니다. 모델의 여러 지점에서 **상호작용 계층(interaction layers)** (예: 선형 투영(linear projection), 특징 교차(feature cross), 타깃 어텐션(target attention))이 사용자 임베딩과 후보 임베딩을 결합합니다. 요청 내 모든 후보에 걸쳐 공유되는 임베딩을 **요청 전용(Request-Only, RO)**, 후보별 임베딩을 **비-요청 전용(Non-Request-Only, NRO)** 이라고 부릅니다.
> When a user opens their feed, the recommendation system must score hundreds to thousands of candidate items to decide what to show. The model’s inputs split into two categories: **user features** (e.g., browsing history, profile, context) that are identical for every candidate in a request, and **candidate features** (e.g., item ID, category, engagement statistics) that are unique to each item. Both pass through embedding lookups and subsequent processing to produce embedding representations. At various points in the model, **interaction layers** (e.g., linear projections, feature crosses, target attention) combine user and candidate embeddings. We call embeddings shared across all candidates in a request **Request-Only (RO)**, and per-candidate embeddings **Non-Request-Only (NRO)**.

![매우 단순화한 RecSys 추론 데이터 흐름 / A very simplified RecSys inference data flow](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/Screenshot-2026-05-01-at-3.50.58-PM.jpg){:style="width:100%"}
*그림 1. 매우 단순화한 RecSys 추론 데이터 흐름. 요청 전용(RO) 사용자 임베딩은 상호작용 계층 이전에 비-요청 전용(NRO) 후보 배치 차원(batch dimension)에 맞도록 브로드캐스트(복제)되어야 합니다. IKBO는 각 커널 내부에서 브로드캐스트를 처리하여 이 실체화(materialization)를 제거합니다. / Fig. 1. A very simplified RecSys inference data flow. Request-Only (RO) user embeddings must be broadcast (replicated) to match the Non-Request-Only (NRO) candidate batch dimension before interaction layers. IKBO eliminates this materialization by handling broadcast internally within each kernel.*

상호작용 계층은 배치 차원이 일치하는 텐서를 요구합니다. 약 15명의 사용자가 서빙하는 1,024개 후보의 배치에서, RO 임베딩은 상호작용 이전에 NRO 배치 크기에 맞추기 위해 약 70번 복제되어 **브로드캐스트**되어야 합니다(그림 1). 아키텍처가 DLRM \[1\], DCN \[2\] 에서 HSTU \[3\], X의 Phoenix \[4\] 같은 시퀀스 모델(sequential model)로 진화하면서, 사용자-후보 상호작용은 꾸준히 풍부해져 왔습니다. 하지만 더 풍부한 상호작용에는 대가가 따릅니다. 사용자 특징이 모든 후보에 걸쳐 브로드캐스트되어야 하기 때문입니다. 추론에서 10 – 10,000+ 규모의 배치 크기에 대해, 이 복제 오버헤드는 후보 수에 선형적으로 비례하는 상당한 연산 및 메모리 비용을 유발합니다.
> Interaction layers require tensors with matching batch dimensions. In a batch of 1,024 candidates served by ~15 users, RO embeddings must be **broadcast**, replicated ~70 times, to match the NRO batch size before any interaction (Fig. 1). As architectures have evolved from DLRM \[1\] and DCN \[2\] through sequential models like HSTU \[3\] and X’s Phoenix \[4\], they have steadily enriched user-candidate interaction. But richer interaction comes at a cost: user features must be broadcast across all candidates. For batch sizes of 10 – 10,000+ in inference, this replication overhead incurs significant computation and memory cost that scales linearly with candidate count.

**브로드캐스트는 연산적 필연이 아니라 데이터 레이아웃의 문제입니다**. 모델과 추론 시스템을 이 관점으로 바라보면 모든 계층에서 최적화의 여지가 열립니다. 추론 런타임(runtime)은 시스템 수준 브로드캐스트를 제거하고, 사용자 전용 모델 계층은 더 작은 사용자 배치 크기로 실행되며, 둘을 섞는 커널은 브로드캐스트를 내부에서 처리하도록 재설계됩니다. 복제된 텐서가 실제로 만들어지는 일이 없습니다. Meta의 RecSys 추론 스택 전반, 즉 초기 단계부터 후기 단계 랭킹 모델까지 GPU와 MTIA 양쪽에 배포된 IKBO는, 공동 설계된 모델에서 연산 집약적인 순 지연 시간을 최대 2/3까지 감소시킵니다.
> **Broadcast is a data layout concern, not a computational necessity**. Viewing the model and inference system through this lens opens optimization at every layer: the inference runtime eliminates system-level broadcast, user-only model layers run at the smaller user batch size, and kernels that mix both are redesigned to handle broadcast internally—no replicated tensors ever materialize. Deployed across Meta’s RecSys inference stack, from early-stage to late-stage ranking models, spanning both GPU and MTIA, IKBO delivers up to 2/3 reduction in compute-intensive net latency on co-designed models.

이번 글은 선형 압축과 Flash Attention이라는 두 개의 심층 분석을 통해 커널 계층에 초점을 맞춥니다.
> This post focuses on the kernel layer through two deep dives: Linear Compression and Flash Attention.

### 1.1. 커널 최적화 유형 / Kernel Optimization Type

**유형 I — 분해 가능한 연산(Decomposable Operations)**. 수학적 재구성을 통해 요청 전용(RO) 부분을 작은 배치 크기에서 독립적으로 계산하고, 마지막에만 비-요청 전용(NRO) 부분과 결합할 수 있습니다. 이는 메모리 대역폭과 연산을 모두 절약합니다.
> **Type I — Decomposable Operations**. Mathematical restructuring lets the Request-Only (RO) portion be computed independently at small batch size, combining with the Non-Request-Only (NRO) portion only at the end. This saves both memory bandwidth and compute.

**유형 II — 메모리 전용 최적화(Memory-Only Optimization).** RO-NRO 브로드캐스팅을 커널 내부에서 처리하여 중복된 데이터 이동을 피하고, 커널을 IO 바운드에서 벗어나게 합니다.
> **Type II — Memory-Only Optimization.** Handling RO-NRO broadcasting within the kernel avoids redundant data movement, pushing the kernel away from IO bound.

### 1.2. 종단 간(E2E) 시스템 설계 / E2E System Design

IKBO를 배포하려면 인프라 스택의 세 계층을 건드려야 합니다.
> Deploying IKBO touches three layers of the infra stack:

1. **커널(Kernels)**: 서로 맞지 않는 RO/NRO 배치 크기를 받아 브로드캐스트를 내부에서 처리하는 커스텀 GPU 커널(2절과 3절).
2. **컴파일 명세(Compilation Specification)**: ML 컴파일러는 적절한 형태의 커널을 선택하기 위해 연산자(operator)별 동적 형태(dynamic shape) 범위를 알아야 합니다. 배치 크기가 하나이면 이는 간단하지만, 둘(사용자와 후보) 또는 그 이상이면, 각 연산자가 어느 것을 사용하는지를 안정적으로 해소하는 일은 — 상호작용이 배치 계보(lineage)를 흐리는 프로덕션 모델 전반에서 — 체계적인 자동화를 요구합니다.
3. **추론(Inference)**: 런타임은 브로드캐스트를 실체화하는 대신 후보-사용자 매핑(candidate-to-user mapping)을 모델에 전달합니다.

> 1. **Kernels**: Custom GPU kernels that accept mismatched RO/NRO batch sizes and handle broadcast internally (Sections 2 and 3).
> 2. **Compilation Specification**: The ML compiler needs per-operator dynamic shape ranges to select appropriately shaped kernels. With one batch size this is trivial; with two (user and candidate) or even more, reliably resolving which each operator uses—across production models where interactions obscure batch lineage—requires systematic automation.
> 3. **Inference**: The runtime passes the candidate-to-user mapping into the model instead of materializing the broadcast.

이 커널들은 두 가지 경로 중 하나를 통해 모델에 들어옵니다.
> These kernels enter the model through one of two paths:

1. 직접 채택(Direct adoption): 모델 작성자가 IKBO 커널을 모델 정의에 직접 통합합니다. 학습 중 후보-사용자 비율이 1보다 크면, 동일한 커널이 학습 비용도 줄여줍니다.
2. 추론 시점 변환(Inference-time transformation): 한 번의 패스(pass)가 추론 시점에 표준 연산을 IKBO 등가물로 자동 교체합니다 — 모델 코드 변경이 필요 없습니다.

> 1. Direct adoption: Model authors integrate IKBO kernels directly into their model definitions. When candidate-to-user ratio > 1 during training, the same kernels reduce training cost as well.
> 2. Inference-time transformation: A pass automatically swaps standard ops for IKBO equivalents at inference time — no model code changes required.

순효과는 다음과 같습니다. 브로드캐스트는 추론의 모든 단계에서 사라지며, 모델에 대한 아키텍처적 제약도, 추론 런타임의 매핑 인터페이스 외의 인프라 변경도 없습니다.
> The net effect: broadcast disappears from every stage of inference, with no architectural constraints on the model and no infrastructure changes beyond the inference runtime’s mapping interface.

### 1.3. 다른 접근법과의 비교 / Comparison with Other Approaches

기존 접근법은 브로드캐스트를 제거하기보다 우회합니다.
> Existing approaches work around broadcast rather than eliminating it.

1. 시스템 수준 브로드캐스트는 GPU 디스패치(dispatch) 이전에 복제된 텐서를 실체화합니다 — 간단하지만 낭비적이며, 비용이 후보 수에 선형적으로 비례해 증가합니다.
2. 넷 분할(net-splitting, ROO) \[5\] 은 모델을 RO 및 NRO 하위 네트워크로 분할하여 중복 작업을 줄이지만, 사용자-후보 상호작용이 일어날 수 있는 위치를 제약하며 작은 RO 배치 크기에서 여전히 추가 비용을 유발합니다.

> 1. System-level broadcast materializes the replicated tensor before GPU dispatch—simple but wasteful, with cost scaling linearly with candidate count.
> 2. Net-splitting (ROO) \[5\] partitions the model into RO and NRO sub-networks, reducing redundant work but constraining where user-candidate interactions can occur and still introduce extra cost at small RO batch sizes.

두 방법 모두 브로드캐스트를 실체화된 텐서로 보존합니다. IKBO는 이를 연산 프리미티브 계층에서 제거합니다. 절감액은 후보-사용자 비율에 비례해 커지고, 어떤 상호작용 패턴이든 브로드캐스트 비용 없이 동작하며, 전체 NRO 배치 차원이 융합된 커널 내에서 GPU 점유율(occupancy)을 제공합니다.
> Both preserve broadcast as a materialized tensor. IKBO eliminates it at the computational primitive layer: savings scale with the candidate-to-user ratio, any interaction pattern works without broadcast cost, and the full NRO batch dimension provides GPU occupancy within fused kernels.

IKBO는 GPU와 MTIA 가속기 양쪽에 배포되었습니다. 이번 글에서는 핵심 최적화 원리를 설명하기 위해 H100 GPU 커널 설계에 초점을 맞춥니다.
> IKBO has been deployed on both GPU and MTIA accelerators. In this blog post, we focus on H100 GPU kernel design to illustrate the core optimization principles.

## 2. 커널 심층 분석 I: IKBO 선형 압축 / Kernel Deep Dive I: IKBO Linear Compression

선형 압축 임베딩(Linear Compress Embedding, LCE)은 입력 임베딩 `(B, K, N)` 을 학습된 투영 `(M, K) @ (B, K, N) → (B, M, N)` 을 통해 압축하며, Wukong \[6\] 등 Meta RecSys 모델에서 널리 채택되고 있습니다. 네 단계의 점진적 최적화를 살펴봅니다.
> Linear Compress Embedding (LCE) compresses input embeddings `(B, K, N)` via a learned projection `(M, K) @ (B, K, N) → (B, M, N)`, and is widely adopted in Meta RecSys models, e.g., Wukong \[6\]. We go through four progressive optimization stages.

### 2.1 행렬곱 분해 / Matmul Decomposition

![LCE 분해 / LCE decomposition](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/Screenshot-2026-05-01-at-3.51.41-PM.jpg){:style="width:100%"}
*그림 2. LCE 분해: 기준선 배치 행렬곱(batched matmul)(좌상단), K를 따라 임베딩 분리 및 사용자 중복 제거(deduplication)(우상단), 압축 출력에 대한 브로드캐스트-덧셈(broadcast-add)을 포함한 두 개의 독립적인 GEMM(하단). / Fig. 2. LCE decomposition: baseline batched matmul (top-left), embedding separation and user deduplication along K (top-right), two independent GEMMs with broadcast-add on compressed output (bottom).*

기준선 LCE는 모든 B개 후보에 걸쳐 단일 배치 행렬곱을 계산합니다. 입력 임베딩은 사용자 부분과 후보 부분을 K를 따라 연결(concatenate)하지만, 사용자 임베딩은 같은 사용자에 대한 모든 후보에서 동일합니다.
> The baseline LCE computes a single batched matmul across all B candidates. The input embeddings concatenate user and candidate parts along K — but user embeddings are identical across all candidates for the same user.

**브로드캐스트를 행렬곱 뒤로 밀어냅니다.** W는 배치와 무관하므로, 선형성을 이용해 분해합니다. 사용자와 후보 임베딩 블록을 K를 따라 분리하고, 반복되는 사용자 임베딩을 중복 제거한 뒤, 두 개의 독립적인 GEMM을 각자의 자연스러운 배치 크기에서 계산합니다. 행렬곱 이전에 사용자 임베딩을 복제하는 대신, 작은 압축 결과만 브로드캐스트합니다. 그림 2를 참고하세요. 후보-사용자 비율이 약 70(대표적인 설정)일 때, 사용자 배치는 `B=1024` 에서 `B_user ≈ 15` 로 줄어듭니다 — 사용자 측 연산이 **70배 감소**하는 것입니다. 이 분해는 표준 PyTorch로 구현됩니다.
> **Push broadcast past the matmul.** Since W is batch-independent, we decompose by linearity: separate user and candidate embedding blocks along K, deduplicate the repeated user embeddings, and compute two independent GEMMs at their natural batch sizes. Instead of replicating user embeddings before the matmul, we broadcast only the small compressed result. See Fig. 2. With a candidate-to-user ratio of ~70 (a representative setting), the user batch shrinks from `B=1024` to `B_user ≈ 15` — a **70x reduction** in user-side compute. The decomposition is implemented in standard PyTorch.

**결과.** 1.944 ms → 1.389 ms (**28.5% 감소**; 벤치마크 설정은 부록 1). 원래의 배치 GEMM(산술 강도(arithmetic intensity) ~ 356 FLOPs/Byte, H100의 ~495 FLOPs/Byte 기계 균형점(machine balance point) 미만; 유도 과정은 부록 2)과 분해된 두 GEMM 모두 메모리 바운드이므로, 속도 향상은 메모리 비용 감소에서 비롯됩니다. 중복 제거는 사용자 측 GEMM(B_user ≈ 15 vs. B = 1024)의 비용을 무시할 수 있는 수준으로 만들면서, 메모리 비용을 절반 이상 줄입니다.
> **Result.** 1.944 ms → 1.389 ms (**28.5% reduction**; benchmark setup in Appendix 1). Both the original batched GEMM (arithmetic intensity ~ 356 FLOPs/Byte, below H100’s ~495 FLOPs/Byte machine balance point; see Appendix 2 for derivations) and the two decomposed GEMMs are memory-bound, so the speedup is driven by memory cost reduction. Deduplication cuts memory cost more than half — as the user-side GEMM (B\_user ≈ 15 vs. B = 1024) becomes negligible in cost.

분해가 브로드캐스트를 행렬곱 뒤로 밀어낸다는 점에 주목하세요. GEMM 이전에 전체 K 차원 입력 임베딩을 복제하는 대신, 훨씬 저렴한 작은 압축 결과만 브로드캐스트합니다. 2.3절에서는 커널 내 브로드캐스트 융합을 통해 이 남은 브로드캐스트마저 완전히 제거합니다.
> Note that the decomposition pushes broadcast past the matmul: instead of replicating full K-dimensional input embeddings before the GEMM, we broadcast only the small compressed result, which is far cheaper. In Section 2.3, we will further eliminate this remaining broadcast entirely via in-kernel broadcast fusion.

현재 병목은 DRAM 사용률이 아니라 L1/TEX 파이프라인 사용률(84%)입니다 — 다음 절에서 자세히 들여다볼, 의심스러운 불균형입니다. 자세한 프로파일링 분석은 부록 3에 있습니다.
> The current bottleneck is L1/TEX pipeline utilization (84%) rather than DRAM utilization — a suspicious imbalance we will zoom into in the next section. Detailed profiling breakdown in Appendix 3.

### 2.2 메모리 레이아웃 최적화 / Memory Layout Optimization

분해된 GEMM의 상세 결과 분석은 불균형을 드러냅니다. L1/TEX는 최대 성능의 84%에 있는 반면 DRAM은 19%에만 도달하며, 이는 불필요하게 좁은 메모리 로드(load)를 나타냅니다. SASS가 이를 확인해 줍니다. 모든 `cp.async` 가 단일 128비트 로드 대신 4바이트만 복사합니다.
> Detailed result analysis of the decomposed GEMM reveals an imbalance: L1/TEX sits at 84% of peak while DRAM reaches only 19%, indicating unnecessarily narrow memory loads. SASS confirms: every `cp.async` copies only 4 bytes instead of a single 128-bit load.

```
LDGSTS.E.LTC128B P0, [R203],      [R38.64]       // 4바이트
LDGSTS.E.LTC128B P1, [R203+0x4],  [R38.64+0x4]   // 4바이트  (×4회, 총 16바이트만 로드)
```

`cp.async` 폭은 소스 포인터의 자연 정렬(natural alignment)에 의해 상한이 정해집니다. 행렬 A는 스트라이드(stride)가 `K × 2` 바이트인 `(M, K)` 행 우선(row-major) 배치라서, K가 8의 배수가 아니면 스트라이드가 128비트 정렬을 깨뜨립니다.
> `cp.async` width is capped by the source pointer’s natural alignment. Matrix A is `(M, K)` row-major with stride `K × 2` bytes, so when K is not a multiple of 8, the stride breaks 128-bit alignment.

**모델-커널 공동 설계 통찰.** 메모리 정렬은 잘 알려진 GPU 최적화이지만, 분해는 이를 모델-커널 공동 설계 과제로 바꿔놓습니다. `K` 는 여러 모델 설정 요인에 따라 크기가 정해지는 임베딩 텐서들을 `torch.cat` 으로 이어붙여 형성됩니다. 분해는 분해된 임베딩이 정확히 배수가 되도록 이런 요인들을 수동으로 설계하기를 매우 어렵게 만듭니다. 체계적인 해법이 필요합니다.
> **Model-kernel co-design insights.** Memory alignment is a well-understood GPU optimization — but decomposition turns it into a model-kernel co-design challenge. `K` is formed by `torch.cat` of embedding tensors whose sizes depend on many model config factors. Decomposition makes it very hard to manually engineer these factors so that decomposed embeddings remain perfect multiples. A systematic solution is needed.

**해법.** 연결(concat) 목록에 0을 덧붙여 분해된 각 K를 8의 다음 배수로 패딩(pad)합니다. 이것이 순전파와 역전파 모두에서 수학적으로 동등함을 증명하며(아래 증명 1 참고), ML 컴파일러의 메모리 플래너(memory planner)를 사용하면 값싼 상수 복사로 축소됩니다.
> **Solution.** Pad each decomposed K to the next multiple of 8 by appending zeros to the concat list. We prove this is mathematically equivalent in both forward and backward passes (see Proof 1 below), and with the ML compiler’s memory planner, reduces to a cheap constant copy.

![증명 1 / Proof 1](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/unnamed-6.png){:style="width:100%"}
*증명 1. K에 0을 패딩해도 순전파와 역전파 모두에서 정확한 수치적 동등성이 보존됩니다. / Proof 1. Zero-padding K preserves exact numerical equivalence in both forward and backward passes.*

**결과.** 1.389 ms → 0.798 ms (**42.5% 감소**). 패딩은 CUTLASS가 TMA 기반 커널을 선택하도록 하여 L1/TEX를 완전히 우회하고(섹터 351M → 0), GEMM 지연 시간을 0.984 ms에서 0.400 ms로 줄입니다. GEMM이 해결되면서, 융합되지 않은 브로드캐스트와 덧셈(0.398 ms)이 이제 전체 지연 시간의 절반을 차지합니다 — 다음 절에서 다룹니다. 자세한 결과 분석은 부록 5에 있습니다.
> **Result.** 1.389 ms → 0.798 ms (**42.5% reduction**). Padding enables CUTLASS to select a TMA-based kernel, bypassing L1/TEX entirely (sectors 351M → 0) and cutting GEMM latency from 0.984 ms to 0.400 ms. With the GEMM resolved, the unfused broadcast and add (0.398 ms) now accounts for half the total latency — to be addressed in the next section. Detailed result analysis in Appendix 5.

### 2.3 후보 GEMM 커널 내 브로드캐스트 융합 / Candidate GEMM In-Kernel Broadcast Fusion

융합되지 않은 브로드캐스트와 덧셈은 메모리 바운드입니다. 후보 GEMM 결과를 HBM에 쓰고, 사용자 결과와 함께 다시 읽어와서, 더한 뒤, 또 씁니다. 우리는 브로드캐스트를 후보 GEMM의 에필로그(epilogue)에 융합하여 이를 제거합니다(그림 3). 각 타일(tile)의 누적(accumulation) 이후, 에필로그는 사용자 인덱스를 조회하고, 미리 계산된 사용자 결과를 로드하여, 레지스터(register)에서 더한 뒤, 최종 합을 씁니다 — 중간 텐서는 결코 실체화되지 않습니다. 우리는 이를 Triton 커널로 구현합니다. 커스텀 누적 후(post-accumulation) 에필로그 블록을 갖춘 표준 배치 GEMM입니다.
> The unfused broadcast and add are memory-bound: write the candidate GEMM result to HBM, read it back alongside the user result, add, and write again. We eliminate this by fusing the broadcast into the candidate GEMM epilogue (Fig. 3). After each tile’s accumulation, the epilogue looks up the user index, loads the pre-computed user result, adds it in registers, and writes the final sum — the intermediate tensor is never materialized. We implement this as a Triton kernel: a standard batched GEMM with a custom post-accumulation epilogue block.

![커널 내 브로드캐스트 융합 / In-kernel broadcast fusion](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/Screenshot-2026-05-01-at-3.52.54-PM.jpg){:style="width:100%"}
*그림 3. 커널 내 브로드캐스트 융합: GEMM 에필로그가 인덱스 조회를 통해 미리 계산된 사용자 결과를 로드하여 레지스터에서 더합니다. / Fig. 3. In-kernel broadcast fusion: the GEMM epilogue loads the pre-computed user result via index lookup and adds it in-register.*

**결과.** 0.798 ms → 0.580 ms (**27.4% 감소**). 융합은 0.87 GB의 중간 DRAM 트래픽을 제거하여 지연 시간 개선에 기여합니다. 하지만 점유율은 6.25%(스케줄러당 워프 1개)에 불과해, 모든 스톨(stall)이 그대로 노출됩니다. 사이클의 42% 이상을 전역 로드(global load) 대기에 쓰는 것을 넘어, 20%는 WGMMA 대기에 소비됩니다 — 에필로그로 숨길 수 없는 스톨이며, 지속성(persistence)이 없으면 겹칠 다음 타일 로드도 없습니다. 이는 까다로운 트레이드오프입니다. 텐서 코어(tensor core)를 계속 채우려면 큰 타일과 깊은 파이프라인이 필요하지만, 이들은 공유 메모리(shared memory) 예산의 대부분을 소비하여 점유율을 통해 지연을 숨길 여지를 거의 남기지 않습니다. 자세한 결과 분석은 부록 6에 있습니다.
> **Result.** 0.798 ms → 0.580 ms (**27.4% reduction**). Fusion eliminates 0.87 GB of intermediate DRAM traffic, contributing to the latency win. However, occupancy is just 6.25% (1 warp per scheduler), leaving every stall fully exposed. Beyond 42% of cycles waiting on global loads, 20% are spent waiting on WGMMA — stalls that cannot be hidden by the epilogue, and without persistence there is no next-tile load to overlap with. This is a challenging tradeoff: large tiles and deep pipelines are needed to keep tensor cores fed, but they consume most of the shared memory budget, leaving little room to hide latency through occupancy. Detailed result analysis in Appendix 6.

### 2.4 TLX를 이용한 워프 특화 다단계 융합 / Warp-Specialized Multi-Stage Fusion with TLX

[**TLX (Triton Low-level Language Extensions)**](https://github.com/facebookexperimental/triton/tree/tlx) 는 Triton의 Python DSL과 오토튜닝(autotuning) 인프라를 유지하면서 Hopper의 워프 특화, TMA, mbarrier, 그리고 명명된 배리어(named barrier)를 노출합니다.
> [**TLX (Triton Low-level Language Extensions)**](https://github.com/facebookexperimental/triton/tree/tlx) exposes Hopper’s warp specialization, TMA, mbarriers, and named barriers while preserving Triton’s Python DSL and autotuning infrastructure.

TLX를 사용하여, 2.3절의 점유율 한계를 워프 특화로 해결합니다 — 추가 워프가 아니라 기능적 분할(functional partitioning)을 통해 지연을 숨깁니다.
> Using TLX, we address the occupancy limitation from Section 2.3 with warp specialization — hiding latency through functional partitioning rather than additional warps.

2.1 – 2.3절은 원래의 LCE를 두 개의 독립적인 계산으로 분해했습니다. 사용자 GEMM(Stage 1)과, 브로드캐스트-덧셈 에필로그가 융합된 후보 GEMM(Stage 2)입니다. 먼저 지배적 병목인 Stage 2 내부의 지연 은닉(latency hiding)을 최적화한 뒤, 두 스테이지를 단일 지속(persistent) 커널로 융합합니다.
> Sections 2.1 – 2.3 decomposed the original LCE into two independent computations: the user GEMM (Stage 1) and the candidate GEMM with fused broadcast-add epilogue (Stage 2). We first optimize latency hiding within Stage 2, the dominant bottleneck, then fuse both stages into a single persistent kernel.

**스테이지 내 지연 중첩 / Intra-Stage Latency Overlap**

후보 IKBO 커널은 메모리 바운드입니다 — 설계 목표는 메모리 파이프라인을 지속적으로 채우는 것입니다. Triton의 소프트웨어 파이프라이닝(software pipelining)(2.3절)은 이미 로드와 WGMMA를 중첩하지만, 에필로그는 여전히 직렬화되어 미래의 로드를 막고 WGMMA 대기 스톨을 노출합니다. 우리는 각 CTA를 특화된 워프 그룹(warp group)으로 분할하여 둘 다 해결합니다. 전용 프로듀서(producer)가 TMA 로드를 지속적으로 발행하고(중첩 #1, Triton의 소프트웨어 파이프라인과 유사), 두 컨슈머(consumer)가 타일을 핑퐁(ping-pong)하여 한쪽의 에필로그가 다른 쪽의 WGMMA와 겹칩니다(중첩 #2). 지속성이 있으면, 타일이 스테이지 간 간극 없이 지속적으로 흐릅니다. 그림 4를 참고하세요.
> The candidate IKBO kernel is memory-bound — the design goal is to keep the memory pipeline continuously fed. Triton’s software pipelining (Section 2.3) already overlaps Loads with WGMMA, but the epilogue remains serialized — it blocks future Loads and exposes the WGMMA wait stalls. We resolve both by partitioning each CTA into specialized warp groups: a dedicated producer issues TMA loads continuously (Overlap #1, analogous to Triton’s software pipeline), while two consumers ping-pong tiles so one’s epilogue overlaps the other’s WGMMA (Overlap #2). With persistence, tiles flow continuously with no cross-tile gaps. See Fig. 4.

![두 개의 스테이지 내 지연 중첩과 워프 그룹 역할 배정을 갖춘 후보 IKBO 커널 구조 / Candidate IKBO kernel structure with two intra-stage latency overlaps and warp group role assignments](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/Screenshot-2026-05-01-at-3.53.35-PM.jpg){:style="width:100%"}
*그림 4. 두 개의 스테이지 내 지연 중첩과 워프 그룹 역할 배정을 갖춘 후보 IKBO 커널 구조. / Fig. 4. Candidate IKBO kernel structure with two intra-stage latency overlaps and warp group role assignments.*

**다단계 융합 / Multi-Stage Fusion**

우리는 사용자 IKBO(Stage 1)와 후보 IKBO(Stage 2)를 단일 메가 커널(mega-kernel)로 융합하여 웨이브 양자화(wave quantization)를 줄이고, 커널 실행(launch) 오버헤드를 제거하며, L2 캐시 사용률을 개선합니다. 높은 후보-사용자 비율은 Stage 1의 웨이브 양자화를 증폭시킵니다. 후보 GEMM은 에필로그 전까지 사용자 결과와 독립적이므로, 두 스테이지를 동시에 스케줄링합니다.
> We fuse user IKBO (Stage 1) and candidate IKBO (Stage 2) into a single mega-kernel to reduce wave quantization, eliminate kernel launch overhead, and improve L2 cache utilization. High candidate-to-user ratios amplify wave quantization in Stage 1. Since the candidate GEMM is independent of user results until its epilogue, we schedule both stages concurrently.

이 동시 스케줄링은 두 개의 추가적인 스테이지 간(cross-stage) 중첩을 열어, 중첩의 총합이 넷이 됩니다. 그림 5를 참고하세요.
> This concurrent scheduling unlocks two additional cross-stage overlaps, bringing the total overlaps to four. See Fig. 5.

![동시 스테이지 스케줄링 / Concurrent stage scheduling](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/Screenshot-2026-05-01-at-3.54.04-PM.jpg){:style="width:100%"}
*그림 5. 동시 스테이지 스케줄링: 사용자 타일이 없는 SM은 곧바로 Stage 2에 진입하여 Stage 1의 부분 웨이브(partial wave)와 겹칩니다. 다단계 융합 이후의 네 가지 지연 중첩으로, 스테이지 내(#1, #2)와 스테이지 간(#3, #4) 중첩 기회를 보여줍니다. SM 0-49, 50-131은 예시 숫자입니다. / Fig. 5. Concurrent stage scheduling: SMs without user tiles enter Stage 2 immediately, overlapping with Stage 1’s partial wave. All four latency overlaps after multi-stage fusion, showing intra-stage (#1, #2) and cross-stage (#3, #4) overlap opportunities. SM 0-49, 50-131 are example numbers.*

**워프 그룹 특화 및 동기화 설정 / Warp Group Specialization & Synchronization Setup**

네 가지 중첩을 모두 실현하기 위해, 각 CTA는 하나의 프로듀서와 두 개의 컨슈머 워프 그룹으로 분할됩니다. 결정적으로, 두 스테이지는 동일한 순환 버퍼(circular buffer)와 `mbarrier` 인프라를 공유합니다 — 스테이지 경계에서 파이프라인 드레인(drain)이나 배리어 재초기화가 일어나지 않습니다. 마지막 사용자 K-블록과 첫 번째 후보 K-블록이 서로 다른 버퍼 슬롯에 동시에 공존합니다. 그림 6을 참고하세요.
> To realize all four overlaps, each CTA is partitioned into one producer and two consumer warp groups. Critically, both stages share the same circular buffer and `mbarrier` infrastructure — no pipeline drain or barrier reinitialization occurs at the stage boundary. The last user K-block and the first candidate K-block coexist in different buffer slots simultaneously. See Fig. 6.

![CTA별 워프 그룹 설정과 세 가지 동기화 메커니즘 / Per-CTA warp group setup and the three synchronization mechanisms](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/Screenshot-2026-05-01-at-3.54.29-PM.jpg){:style="width:100%"}
*그림 6. CTA별 워프 그룹 설정과 세 가지 동기화 메커니즘. / Fig. 6. Per-CTA warp group setup and the three synchronization mechanisms.*

**양방향 스테이지 교대 타일 스케줄링 / Bidirectional Stage-Alternating Tile Scheduling**

어느 스테이지의 타일 수도 SM 수로 균등하게 나눠지지 않을 때, 순진한 단방향 디스패치는 작업 부하 불균형을 일으킵니다. 우리는 스테이지 간 타일 배정 방향을 반대로 합니다. Stage 1은 `pid` 에서 시작하고, Stage 2는 `NUM_SM - 1 - pid` 에서 시작합니다. 그림 7을 참고하세요.
> When neither stage’s tile count divides evenly by the SM count, naive unidirectional dispatch causes workload imbalance. We reverse tile assignment direction between stages: Stage 1 starts at `pid`, Stage 2 at `NUM_SM - 1 - pid`. See Fig. 7.

![단방향(좌) vs. 양방향 스테이지 교대 디스패치(우) / Unidirectional (left) vs. bidirectional stage-alternating dispatch (right)](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/Screenshot-2026-05-01-at-3.55.05-PM.jpg){:style="width:100%"}
*그림 7. 단방향(좌) vs. 양방향 스테이지 교대 디스패치(우)로, 부분 웨이브 전반에 걸쳐 SM별 작업 부하를 균형 잡습니다. / Fig. 7. Unidirectional (left) vs. bidirectional stage-alternating dispatch (right), balancing per-SM workload across partial waves.*

**타일 단위 CTA 간 동기화 / Tile-Granularity Cross-CTA Synchronization**

사용자 타일과 후보 타일은 서로 다른 CTA에서 실행될 수 있어 CTA 간 동기화를 요구하지만, 장치 전역 배리어(device-wide barrier)는 모든 작업을 직렬화하여 중첩을 파괴합니다. 우리는 세 단계 릴리스-어콰이어(release-acquire) 프로토콜을 사용하여 타일 단위로 동기화합니다.
> User and candidate tiles may execute on different CTAs, requiring cross-CTA synchronization — but a device-wide barrier would serialize all work and destroy the overlap. We synchronize at per-tile granularity using a three-step release-acquire protocol:

1. 워프 그룹당 단일 스레드가 `ld.relaxed` 로 타일 플래그(flag)를 스핀(spin)하여 메모리 트래픽을 최소화합니다.
2. 설정되면, 단일 `ld.acquire` 가 선행 발생(happens-before) 관계를 확립합니다.
3. 명명된 배리어가 워프 그룹의 128개 스레드 전체에 준비 완료를 브로드캐스트합니다.

> 1. A single thread per warp group spins on the tile flag with `ld.relaxed`, minimizing memory traffic
> 2. Once set, a single `ld.acquire` establishes the happens-before edge
> 3. A named barrier broadcasts readiness to all 128 threads in the warp group

이는 폴링(polling) 중 값비싼 펜스(fence)를 피하고, 서로 다른 사용자 타일에 있는 후보 CTA가 완전히 독립적으로 진행하도록 합니다. 자세한 내용은 부록 7에 있습니다.
> This avoids expensive fences during polling and lets candidate CTAs on different user tiles proceed fully independently. Details in Appendix 7.

**결과 / Results**

모든 최적화를 결합하면, 지연 시간이 0.580 ms에서 0.482 ms로 개선됩니다(**16.9% 감소**). 명료한 워프 내(intra-warp) [Proton 트레이서(tracer)](https://github.com/triton-lang/triton/tree/main/third_party/proton/tutorials/intra_kernel) 타임라인은 네 가지 중첩이 모두 실제로 실현되었음을 확인해 줍니다.
> With all optimizations combined, latency improves from 0.580 ms to 0.482 ms (**16.9% reduction**). The clear intra-warp [Proton tracer](https://github.com/triton-lang/triton/tree/main/third_party/proton/tutorials/intra_kernel) timeline confirms all four overlaps are realized in practice.

![두 CTA에 대한 Proton 프로파일러 타임라인 / Proton profiler timeline for two CTAs](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/unnamed-5.png){:style="width:100%"}
*그림 8. 두 CTA에 대한 Proton 프로파일러 타임라인으로, 네 가지 중첩이 모두 색상별로 구분되어 있습니다. 메모리 파이프라인이 지속적으로 채워진 상태를 유지합니다. / Fig. 8. Proton profiler timeline for two CTAs, with all four overlaps color-coded. The memory pipeline remains continuously fed.*

주된 이득은 중첩 #2에서 나옵니다. 핑퐁하는 컨슈머가 모든 타일에서 WGMMA와 에필로그 스톨을 숨깁니다 — 2.3절의 지배적인 낭비 사이클을 직접적으로 해결하는 것입니다. 중첩 #1(로드↔WGMMA)은 Triton의 기존 소프트웨어 파이프라이닝에서 이어져 옵니다. 중첩 #3과 #4는 사용자-후보 스테이지 전환에서 발생하는 유휴 시간을 숨깁니다. 그림 8을 참고하세요.
> The primary gain comes from Overlap #2: ping-ponging consumers hide WGMMA and epilogue stalls on every tile — directly addressing the dominant wasted cycles from Section 2.3. Overlap #1 (Load↔WGMMA) carries forward from Triton’s existing software pipelining. Overlaps #3 and #4 hide idle time at the user-to-candidate stage transition. See Fig. 8.

NCU가 이를 확인해 줍니다. 점유율이 6.25%에서 18.75%로(워프 그룹 3개 vs. 1개), DRAM 처리량이 39%에서 52%로, 그리고 병목인 L2가 최대 성능의 74%에서 84%로 상승합니다. 이는 점유율만의 효과가 아닙니다. 네 가지 중첩 전반의 공격적인 지연 은닉이 메모리 파이프라인을 포화 상태로 유지하며, 이것이 L2를 80% 이상으로 밀어 올립니다. 자세한 NCU 지표는 부록 8에 있습니다.
> NCU confirms: occupancy rises from 6.25% to 18.75% (3 warp groups vs. 1), DRAM throughput from 39% to 52%, and L2 — the bottleneck — from 74% to 84% of peak. This is not occupancy alone: the aggressive latency hiding across all four overlaps keeps the memory pipeline saturated, which is what pushes L2 past 80%. Detailed NCU metrics in Appendix 8.

기본 설정(batch=1024, ratio=70)을 두고 여러 배치 크기와 후보-사용자 비율에 걸쳐 벤치마크합니다. 그림 9를 참고하세요.
> We benchmark across batch sizes and candidate-to-user ratios, with the default (batch=1024, ratio=70) settings. See Fig. 9.

![배치 크기(좌)와 후보-사용자 비율(우)에 걸친 누적 IKBO 속도 향상 / Cumulative IKBO speedup across batch sizes (left) and candidate-to-user ratios (right)](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/Screenshot-2026-05-01-at-3.56.12-PM.jpg){:style="width:100%"}
*그림 9. 배치 크기(좌, ratio=70)와 후보-사용자 비율(우, batch=1024)에 걸친 누적 IKBO 속도 향상. / Fig. 9. Cumulative IKBO speedup across batch sizes (left, ratio=70) and candidate-to-user ratios (right, batch=1024).*

IKBO 융합은 여러 시나리오에 걸쳐 견고한 이득을 제공합니다. 배치 크기(좌)와 후보-사용자 비율(우) 전반에서 약 4배의 속도 향상을 보입니다. 낮은 후보-사용자 비율에서도, 커널은 여전히 의미 있는 속도 향상을 달성합니다.
> The IKBO fusion delivers robust gains across scenarios: ~4x speedup across batch sizes (left) and candidate-to-user ratios (right). Even at low candidate-to-user ratios, the kernel still achieves meaningful speedup.

## 3. 커널 심층 분석 II: IKBO Flash Attention / Kernel Deep Dive II: IKBO Flash Attention

추천 모델이 더 풍부한 사용자 순차 행동(sequential behavior)을 포착하도록 확장되면서, 어텐션을 포함한 **시퀀스 아키텍처(sequential architectures)** 가 핵심적인 연산 병목으로 부상했으며, 1K 시퀀스 길이에서 추론 지연 시간의 약 **40%** 를 차지합니다. 이는 RecSys 특유의 배칭(batching) 의미론과 공동 설계된 IKBO 인식(IKBO-aware) Flash Attention에 초점을 맞추게 된 동기입니다.
> As recommendation models scale to capture richer user sequential behavior, **sequential architectures** – including **attention** – have emerged as a critical compute bottleneck, accounting for approximately **40% of inference latency** at 1K sequence lengths. This motivates our focus on IKBO-aware Flash Attention, co-designed with RecSys’s unique batching semantics.

Transformer와 Set Transformer \[7, 8\] 에서 영감을 받아, RecSys에서는 두 가지 근본적인 사용자 이력 상호작용 모듈이 널리 채택되었습니다.
> Inspired by Transformers and Set Transformers \[7, 8\], two fundamental user history interaction modules have been widely adopted in RecSys:

- **타깃 어텐션(Target attention)** (크로스 어텐션과 유사)은 예측 후보와 사용자의 과거 상호작용 사이의 관계를 포착합니다.
- **셀프 어텐션(Self-attention)** 은 사용자 이력 자체 내의 순차적 의존성을 모델링합니다.

> - **Target attention** (analogous to cross-attention) captures the relationship between the prediction candidate and the user’s historical interactions.
> - **Self-attention** models sequential dependencies within the user history itself

사용자 이력은 RO 특징인 반면 타깃은 별개의 후보(non-RO) 배치 차원에서 동작하므로, 이 아키텍처적 비대칭성은 IKBO가 모델 확장성과 연산 효율성을 개선할 기회를 제공합니다. 타깃 어텐션이 우리의 주요 최적화 대상이 되며, 약간의 공동 설계로 셀프 어텐션도 3.3절에서 IKBO 타깃 어텐션에 융합될 수 있습니다. 우리 모델은 인코더 기반(encoder-driven)이므로, 인과 마스킹(causal masking) 없이 전체 어텐션(full attention)이 적용됩니다.
> Since user history is a RO feature while the target operates on a distinct candidate (non-RO) batch dimension, this architectural asymmetry presents an opportunity for IKBO to improve model scalability and computational efficiency. Target attention will be our main focus for optimization, while with minor co-design, self attention could also be fused into IKBO target attention in Section. 3.3. As our model is encoder-driven, full attention is applied without causal masking.

종단 간 공동 설계를 활용한 최종 최적화 타깃 어텐션 버전은 공동 설계되지 않은 CuTeDSL FA4-Hopper 대비 **2.4배/6.4배**의 처리량을 달성하며(어텐션 커널만 / 어텐션 커널 + 브로드캐스팅 비용), 지연 시간을 각각 **0.320ms / 1.232ms** 줄입니다(표 2).
> The ultimate optimized target attention version leveraging e2e co-design achieves **2.4×/6.4×** the throughput of non-co-designed CuTeDSL FA4-Hopper (attn kernel only / attn kernel + broadcasting cost), reducing latency by **0.320ms / 1.232ms** respectively (Table. 2).

### 3.1 IKBO flash attention은 RecSys 경계 조건에서의 IO 바운드 문제를 해결합니다 / IKBO flash attention solves the IO bound issues under RecSys boundary conditions

![후보-사용자 브로드캐스팅을 포함한 전통적 SDPA(좌) vs. 융합된 IKBO 타깃 어텐션(우) / Traditional SDPA with candidate-user broadcasting (left) vs. fused IKBO target attention (right)](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/unnamed-4.png){:style="width:100%"}
*그림 10. 후보-사용자 브로드캐스팅을 포함한 전통적 SDPA(좌) vs. 융합된 IKBO 타깃 어텐션(우). / Fig. 10: Traditional SDPA with candidate-user broadcasting (left) vs. fused IKBO target attention (right).*

IKBO는 K/V 브로드캐스팅을 어텐션 커널에 융합하며, 비균일한 후보-사용자 비율을 처리하는 추론 런타임의 후보-사용자 매핑 텐서를 통해 수학적 동등성을 유지합니다. 그림 10은 두 접근법을 대조합니다. 전통적 SDPA 경로는 어텐션 이전에 K와 V를 전체 후보 배치 크기로 브로드캐스트하는 반면, IKBO 경로는 이 실체화를 완전히 제거합니다 — 각 후보가 즉석에서 자신의 사용자 K/V로 인덱싱합니다.
> IKBO fuses K/V broadcasting into the attention kernel, maintaining mathematical equivalence via a candidate-user mapping tensor from the inference runtime that handles non-uniform candidate-to-user ratios. Fig. 10 contrasts the two approaches: the traditional SDPA path broadcasts K and V to the full candidate batch size before attention, while the IKBO path eliminates this materialization entirely — each candidate indexes into its user’s K/V on the fly.

**IKBO 공동 설계로 IO 바운드를 연산 바운드로 전환 / Shifting IO-Bound to Compute-Bound by IKBO co-design**

RecSys 경계 조건에서, 타깃 어텐션은 사용자의 열람 이력에 비해 상대적으로 적은 수의 후보 임베딩으로 후보 속성을 표현합니다. 표준 어텐션의 루프라인(roofline) 분석은 산술 강도가 ~60 FLOPs/Byte임을 드러내는데 — 이는 H100(SXM5 HBM2e 버전)의 최대치 ~495 FLOPs/Byte(부록 2)에 한참 못 미치며 — 표준 flash attention조차 크게 IO 바운드로 만듭니다. IKBO는 동일한 사용자 컨텍스트를 공유하는 여러 후보에 걸쳐 K/V 메모리 접근을 분할 상환(amortize)하여, 산술 강도를 ~60 FLOPs/Byte에서 ~833 FLOPs/Byte로(B_candidate : B_user = 70:1에서) 개선하고 커널을 확실히 연산 바운드 영역으로 옮김으로써 이 문제를 해결합니다.
> In RecSys boundary conditions, target attention uses a relatively small number of candidate embeddings to represent the candidate attributes compared to the user’s browsing history. Roofline analysis of standard attention reveals an arithmetic intensity of ~60 FLOPs/Byte – well below the H100 (SXM5 HBM2e version) peak of ~495 FLOPs/Byte (Appendix 2)—making even standard flash attention heavily IO-bound. IKBO addresses this by amortizing K/V memory accesses across multiple candidates sharing the same user context, improving arithmetic intensity from ~60 FLOPs/Byte to ~833 FLOPs/Byte (at B\_candidate : B\_user = 70:1) and shifting the kernel firmly into compute-bound territory.

이 이점을 극대화하기 위해, 우리 구현은 스레드블록(threadblock) 실행 그리드(launch grid)를 재정렬하여 batch_size_candidate가 num_heads보다 앞에 오도록 합니다. 이는 서로 다른 후보를 처리하되 동일한 사용자 K/V를 공유하는 스레드블록이 동시에 스케줄링되도록 하여 L2 캐시 재사용을 개선합니다.
> To maximize this benefit, our implementation reorders the threadblock launch grid so that batch\_size\_candidate comes before num\_heads. This ensures threadblocks processing different candidates — but sharing the same user K/V — are scheduled concurrently, improving L2 cache reuse.

| 그리드 차원 / Grid dimension | Flash attention (SDPA) | IKBO target attention |
|---|---|---|
| x | num_q_seq_block | num_q_seq_block |
| y | num_heads | batch_size_candidate |
| z | batch_size_candidate | num_heads |

**표 1**: 실행 그리드 구성 비교. SDPA는 num_heads를 grid.y에 배치하여 GQA 최적화를 우선합니다. IKBO는 헤드와 후보 차원을 맞바꿔 batch_size_candidate를 grid.y에 배치함으로써 후보 전반의 효율적인 K/V 공유를 가능하게 합니다.
> **Table 1**: Launch grid configuration comparison. SDPA prioritizes GQA optimization by placing num\_heads in grid.y. IKBO swaps head and candidate dimensions, placing batch\_size\_candidate in grid.y to enable efficient K/V sharing across candidates.

표 2는 우리의 IKBO Triton 구현(FA2 로직 + IKBO)을 Hopper의 최신(state-of-the-art) Flash Attention 구현들(IKBO 공동 설계 없음)과 비교합니다. 처리량과 IO는 어텐션에 대해서만 측정합니다. Key와 Value의 브로드캐스팅 지연 시간은 어텐션 비용 자체보다도 큽니다.
> Table 2 compares our IKBO Triton implementation (FA2 logic + IKBO) against state-of-the-art Flash Attention implementations on Hopper (without IKBO co-design). Throughput and IO are measured on attention only; the broadcasting latency for Key and Value is even larger than the attention cost itself.

| | 처리량 / Throughput (TFLOPs/s) | IO (GB/s) | 지연 시간 / Latency (ms) |
|---|---|---|---|
| Triton IKBO FA2 | 425 | 487 | 0.321 (브로드캐스트 융합) |
| TLX FA3 | 245 | 2152 | 0.561 + 0.912 (K·V 브로드캐스트) |
| CuTeDSL FA4 Hopper | 250 | 2193 | 0.550 + 0.912 (K·V 브로드캐스트) |
| TLX IKBO FA3 persistence generalized | 594 | 681 | 0.230 (브로드캐스트 융합) |

**표 2**: RecSys 경계 조건에서의 어텐션 커널 비교(B_candidate = 2048, B_u = 32, 균일 후보-사용자 비율). 공동 설계가 없으면, 최첨단 Hopper 구현조차 IO 바운드로 남습니다.
> **Table 2**: Attention kernel comparison under RecSys boundary conditions (B\_candidate = 2048, B\_u = 32, uniform candidate-to-user ratio). Without co-design, even cutting-edge Hopper implementations remain IO-bound.

### 3.2 TLX에서 IKBO와 함께 현대적 커널 기법(FA3, FA4) 채택 / Adopting Modern Kernel Techniques (FA3, FA4) with IKBO on TLX

IKBO가 커널을 IO 바운드에서 연산 바운드로 옮기면서, 자연스러운 다음 단계는 Hopper에서 Flash Attention 3(FA3 \[10\])와 Flash Attention 4(FA4 \[11\])의 최신 연산 최적화 — 구체적으로 워프 특화와 파이프라이닝 — 를 채택하는 것이었습니다. 하지만 쿼리(query) 임베딩 수에 대한 우리의 경계 조건(q_seq = 32 또는 64)은 FA3의 핑퐁이나 협력적(cooperative) 워프 특화를 직접 채택하기 어렵게 만듭니다.
> With IKBO shifting the kernel from IO-bound to compute-bound, the natural next step was to adopt the state-of-the-art compute optimizations from Flash Attention 3 (FA3 \[10\]) and Flash Attention 4 (FA4 \[11\]) on Hopper – specifically warp specialization and pipelining. However, our boundary conditions on the number of query embeddings (q\_seq = 32 or 64) make it difficult to directly adopt FA3’s ping-pong or cooperative warp specialization.

Hopper에서의 워프 특화는 비동기 WGMMA 명령을 요구하며, 이는 최소 BLOCK_M ≥ 64를 부과합니다. 두 컨슈머 워프 그룹 사이의 버블(bubble)을 최소화하기 위해 두 개의 컨슈머 워프 그룹도 필요합니다. 이 제약을 만족시키기 위해, 우리는 동일한 B_user를 공유하는 B_candidate = i와 B_candidate = i + 1을 단일 스레드블록 내에서 실행하도록 커널을 커스터마이징했습니다. 아래 논의에서는 모든 사용자가 짝수 개의 후보를 q_seq = 64로 랭킹한다고 가정하며, 홀수 후보 처리는 그다음에 다룹니다.
> Warp specialization on Hopper requires asynchronous WGMMA instructions, which impose a minimum BLOCK\_M ≥ 64. Two consumer warp groups are also necessary to minimize bubbles between them. To satisfy these constraints, we customized the kernel to launch both B\_candidate = i and B\_candidate = i + 1 within a single threadblock, sharing the same B\_user. In the discussion below, we assume all users rank an even number of candidates with q\_seq = 64; odd-candidate handling follows afterward.

**IKBO FA3 커널 성능 개선 / Performance improvement for IKBO FA3 kernel**

FA3의 방법론 — 워프 내 파이프라이닝, 워프그룹 특화, 핑퐁 스케줄링 — 에서 출발한 초기 TLX IKBO FA3 커널은 FA2 기준선과 비슷한 성능을 보였습니다(그림 12, 파란색 vs. 빨간색, 부록 11). 처리량은 동등한 수준이었습니다.
> Starting from FA3’s recipe — intra-warp pipelining, warpgroup specialization, and ping-pong scheduling — the initial TLX IKBO FA3 kernel performed similarly to the FA2 baseline (Fig. 12, blue vs. red, Appendix 11), with on-par throughput.

병목을 진단하기 위해, GPU 사이클을 지연 시간 단위로 삼아 [Proton 트레이서](https://github.com/triton-lang/triton/tree/main/third_party/proton/tutorials/intra_kernel)로 워프 내 파이프라이닝을 시각화했습니다(그림 10). 표 3은 Proton 트레이서를 통해 GPU 사이클로 측정한, 지속성(persistence) 전후의 핵심 병목을 요약합니다.
> To diagnose the bottleneck, we visualized intra-warp pipelining using the [Proton tracer](https://github.com/triton-lang/triton/tree/main/third_party/proton/tutorials/intra_kernel) with GPU cycles as the latency unit (Fig. 10). Table 3 summarizes the key bottlenecks before and after persistence, measured in GPU cycles via the Proton tracer.

![TLX IKBO FA3 커널의 Proton 기반 워프 내 프로파일링 / Proton-based intra-warp profiling of the TLX IKBO FA3 kernel](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/unnamed-3.png){:style="width:100%"}
*그림 11. TLX IKBO FA3 커널의 Proton 기반 워프 내 프로파일링. 각 워프 그룹의 대표 워프가 표시됩니다: 워프 0(프로듀서), 워프 4(컨슈머 1), 워프 8(컨슈머 2). 텐서 코어 버블을 식별하기 위해 softmax_PV_overlap 영역과 순수 softmax 영역이 별도로 표시됩니다. (A) 지속성 이전, B의 확대 뷰 (B) 지속성 이전, 2 웨이브 (C) 지속성 이후, 2 웨이브. / Fig. 11: Proton-based intra-warp profiling of the TLX IKBO FA3 kernel. Representative warps from each warp group are shown: warp 0 (producer), warp 4 (consumer 1), and warp 8 (consumer 2). The softmax\_PV\_overlap and pure softmax regions are marked separately to identify the tensor core bubbles. (A) Before persistence zoomed in view of B (B) Before persistence with 2 waves (C) After persistence with 2 waves*

| 병목 / Bottlenecks | 이전 / Before | 이후 / After | 핵심 변화 / Key change |
|---|---|---|---|
| **텐서 코어 버블** (웨이브당 첫 QK<sup>T</sup>, 파란색) | ~1,300 사이클 (워프 스케줄러 전환에서 400 사이클) | ~1,300 사이클 | 변화 없음 |
| **텐서 코어 버블** (웨이브당 마지막 PV, 파란색) | ~2,000 사이클 | ~300 사이클 | 비동기 TMA 저장 + 마지막 PV와의 역수(reciprocal) 중첩 |
| **CTA 간 스톨** (주황색) | ~14,000 사이클 | 제거됨 | 지속성이 CTA 재실행을 완전히 제거 |
| **버퍼 및 배리어 초기화** (초록색) | ~1,600 사이클/웨이브 | ~1,600 사이클 (첫 웨이브만) | 지속성이 공유 버퍼와 배리어를 웨이브 전반에 분할 상환 |
| **첫 Q/K 로드 대기** (짙은 보라색) | 2,100~4,000 사이클/웨이브 (길이는 HBM 대역폭 경합에 따라 달라짐) | ~2,000 사이클 (첫 웨이브만) | 웨이브 간 파이프라이닝; 프로듀서가 ~3K 사이클 앞서 프리페치(prefetch) |

**표 3**: 지속성 + 최적화 전후의 핵심 병목.
> **Table 3**: Key bottlenecks before and after persistence + optimizations.

핵심 요점: CTA 간 스톨이 지배적인 병목이며 — 텐서 코어 사용률이 아닙니다 — 이는 이런 작은 쿼리 시퀀스 길이에서 그렇습니다. 이 개선을 위해서는 지속성이 필수입니다. 지속성 이후의 프로파일링 결과와 그에 따른 지연 시간 변화는 그림 11C와 표 3에 제시됩니다.
> Key takeaway: cross-CTA stalls are the dominant bottleneck — not tensor core utilization – at these small query sequence lengths. Persistence is a must for this improvement. After persistence, the profiling results and its latency changes are presented in Fig. 11C and Table. 3.

**HBM2e 특화 최적화 / HBM2e-Specific Optimizations**

우리는 지속 커널을 H100 SXM5의 HBM2e 대역폭 제약에 맞춰 추가로 튜닝하여, 공유 메모리 용량을 로드/저장 블로킹(blocking) 감소와 맞바꿨습니다(표 4).
> We further tuned the persistent kernel for the H100 SXM5’s HBM2e bandwidth constraints, trading shared memory capacity for reduced load/store blocking. (Table 4).

| 커스텀 최적화/수정 / Customized optimization/fix | 이점 / Benefit |
|---|---|
| **파이프라인화된 TMA 비동기 저장과 함께 O의 SMEM 버퍼를 Q/V와 분리** | O를 Q/V SMEM 공유에서 분리하여 TMA 비동기 저장이 다음 웨이브 연산과 겹칠 수 있게 하고, 저장 블로킹 시간을 **1,300**에서 **400** 사이클/웨이브로 단축 |
| **Q₀와 Q₁ 버퍼 분리** | Q별 로딩 시간을 줄여, 한 컨슈머 그룹이 더 일찍 시작 — 웨이브 수가 K/V 시퀀스 반복 수를 크게 초과할 때(RecSys에서 흔함) 유리 |
| **명령어 캐시 미스(Instruction Cache Miss) 수정** | 분리된(peeled-out) 마지막 반복 코드 경로를 메인 루프에 다시 병합하여, 과도한 워프 특화 명령이 유발하는 icache 스래싱(thrashing) 제거(부록 12) |

**표 4:** HBM2e H100 SXM5를 위한 커스텀 최적화. 이들은 RecSys 경계 조건에서 사용 가능한 SMEM 예산 내에 여전히 들어맞습니다(부록 10).
> **Table 4:** Customized optimizations for the HBM2e H100 SXM5. These still fit within the available SMEM budget under RecSys boundary conditions (Appendix 10).

우리는 마스킹 로직을 단순화하기 위해 K 시퀀스의 끝에서 앞으로 반복하는(FA3/FA4-Hopper의 접근법과 일치) 지속 V2도 구현했습니다. 두 지속 변형 모두 표 4의 최적화를 적용합니다. 그림 12에서 보듯이, 낮은 시퀀스 길이(512–4,096)에서는 TLX FA3 지속 커널이 다른 모든 후보를 능가하며, 8K를 넘어가면 두 지속 변형이 수렴합니다.
> We also implemented persistent V2, which iterates from the end of the K sequence to the front (matching FA3/FA4-Hopper’s approach) to simplify masking logic. Both persistent variants apply the Table 4 optimizations. As shown in Fig. 12, at low sequence lengths (512–4,096) the TLX FA3 persistent kernel outperforms all other candidates; beyond 8K the two persistent variants converge.

![IKBO 구현 처리량 vs. 시퀀스 길이 / IKBO implementation throughput vs. sequence length](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/unnamed-2.png){:style="width:100%"}
*그림 12. IKBO 구현 처리량 vs. 시퀀스 길이 (B_candidate = 2,048; B_candidate : B_user = 64; num_head = 2; d_head = 128). 실용적인 RecSys 시퀀스 길이는 4K 미만이며 \[3\], 더 긴 길이는 LLM 사용 사례와의 비교를 위해 포함되었습니다. 일반화된(generalized) 버전은 사용자당 홀수 후보 확률 50%로 비균일 후보를 처리합니다. / Fig. 12: IKBO implementation throughput vs. sequence length (B\_candidate = 2,048; B\_candidate : B\_user = 64; num\_head = 2; d\_head = 128). Practical RecSys sequence lengths are under 4K \[3\]; longer lengths are included for comparison with LLM use cases. The generalized version handles non-even candidates per user with 50% odd-candidates per user probability*

**임의의 후보 배치 크기 랭킹을 위한 IKBO FA3 일반화 / Generalizing IKBO FA3 for ranking Arbitrary Candidate Batch Sizes**

우리의 IKBO FA3 커널은 WGMMA의 BLOCK_M ≥ 64 요구사항을 충족하기 위해 CTA당 두 개의 후보 배치를 함께 처리합니다. 사용자가 홀수 개의 후보를 가지면, 한 컨슈머 워프그룹은 짝을 이룰 상대가 없습니다. 우리는 이를 유휴(idling) 로직으로 처리합니다(그림 13, 좌; 알고리즘 1):
> Our IKBO FA3 kernel co-processes two candidate batches per CTA to meet WGMMA’s BLOCK\_M ≥ 64 requirement. When a user has an odd number of candidates, one consumer warpgroup has no pairing partner. We handle this with idling logic (Fig. 13, left; Algorithm 1):

- 유휴 워프그룹은 프로듀서 교착(deadlock)을 방지하기 위해 mbarrier 신호를 통해 K/V 버퍼를 비웁니다(drain).
- 활성 워프그룹은 핑퐁 동기화를 비활성화합니다(짝 워프그룹이 더 이상 명명된 배리어에 도착하지 않으므로).

> - The idle warpgroup drains K/V buffers via mbarrier signaling to prevent producer deadlock.
> - The active warpgroup disables ping-pong synchronization (its partner no longer arrives at the named barriers).

약 70 : 1의 후보-사용자 비율에서, 유휴 경로는 0.7% 미만으로 발동하며 오버헤드가 미미합니다(그림 12, IKBO TLX FA3 generalized). 이 접근법은 q_seq_len = 32로 일반화되며, 여기서는 유사한 유휴 및 마스킹 로직을 사용하여 CTA당 네 개의 후보 배치를 묶습니다.
> At a ~70 : 1 candidate-to-user ratio, the idle path triggers less than 0.7% of the time with negligible overhead (Fig. 12, IKBO TLX FA3 generalized). This approach generalizes to q\_seq\_len = 32, where four candidate batches are bundled per CTA using analogous idling and masking logic.

![일반화된 타깃 어텐션(좌)과 셀프 + 타깃 어텐션 융합(우)을 위한 CTA 배정 / CTA assignment for generalized target attention (left) and self + target attention fusion (right)](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/unnamed-1.png){:style="width:100%"}
*그림 13. 일반화된 타깃 어텐션(좌)과 셀프 + 타깃 어텐션 융합(우)을 위한 CTA 배정. 각 CTA는 동일한 사용자 K/V를 공유하는 두 개의 컨슈머 워프 그룹을 배정합니다. 후보 수가 홀수이면, 두 번째 컨슈머가 유휴 상태가 되어 배리어를 비웁니다. / Fig. 13: CTA assignment for generalized target attention (left) and self + target attention fusion (right). Each CTA assigns two consumer warp groups sharing the same user K/V. When the candidate count is odd, the 2nd consumer idles and drains barriers.*

![알고리즘 1 / Algorithm 1](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/unnamed.png){:style="width:100%"}
*알고리즘 1: 홀수 후보 처리를 포함한 IKBO 어텐션 순전파. / Algorithm 1: IKBO Attention Forward Pass with Odd Candidate Handling*

### 3.3 모델 공동 설계를 통한 셀프 + 타깃 어텐션 융합 / Self + Target Attention Fusion via Model Co-Design

앞선 절들은 타깃(크로스) 어텐션 최적화에 초점을 맞췄습니다. 자연스러운 질문이 떠오릅니다. 셀프 어텐션을 같은 커널에 접어 넣을 수 있을까요?
> The previous sections focused on optimizing target (cross) attention. A natural question arises: can we fold self-attention into the same kernel?

핵심 통찰은 두 어텐션 유형이 동일한 키-값(key-value) 소스, 즉 사용자 시퀀스를 공유한다는 것입니다. 유일한 차이는 쿼리입니다. 셀프 어텐션 쿼리는 사용자 측에서 오고, 타깃 어텐션 쿼리는 후보 측에서 옵니다. 둘 사이에 K/V 투영을 공유함으로써, 단일 실행 내에서 직접적인 수평 커널 융합(horizontal kernel fusion)을 가능하게 합니다. 그림 13(우)은 융합된 CTA 레이아웃을 보여줍니다. 첫 번째 CTA들은 셀프 어텐션 쿼리 블록을 처리하고, 나머지 CTA들은 타깃 어텐션 후보 쌍을 처리하며 — 모두 동일한 파이프라인화된 K/V 스트림에서 읽습니다.
> The key insight is that both attention types share the same key-value source — the user sequence. The only difference is the query: self-attention queries come from the user side, while target-attention queries come from the candidate side. By sharing K/V projections between the two, we enable direct horizontal kernel fusion within a single launch. Fig. 13 (right) illustrates the fused CTA layout: the first CTAs handle self-attention query blocks, while the remaining CTAs handle target-attention candidate pairs — all reading from the same pipelined K/V stream.

유사한 공동 설계 아이디어가 X의 오픈소스 추천 시스템인 XAI Phoenix에서 탐구된 바 있습니다 \[4\].
> Similar co-design ideas have been explored in XAI Phoenix, an open-source recommendation system from X \[4\].

우리는 K/V 투영 절감을 제외하고 융합 이점을 정량화하기 위해 융합 커널을 프로토타이핑했습니다(그림 13, 우):
> We prototyped a fused kernel to quantify the fusion benefit, excluding K/V projection savings (Fig. 13, right):

- seq_len = 512:    6.6% 개선 (514 vs. 482 TFLOPs/s)
- seq_len = 1,024:  4.1% 개선 (581 vs. 558 TFLOPs/s)
- seq_len = 2,048:  0.3% 개선 (612 vs. 610 TFLOPs/s) — 셀프 어텐션이 SM을 포화시킴

> - seq\_len = 512:    6.6% improvement (514 vs. 482 TFLOPs/s)
> - seq\_len = 1,024:  4.1% improvement (581 vs. 558 TFLOPs/s)
> - seq\_len = 2,048:  0.3% improvement (612 vs. 610 TFLOPs/s) — self-attention saturates the SMs

짧은 시퀀스에서의 이득은 커널 융합의 이점에서 비롯됩니다. 실행 오버헤드 감소, 공유 버퍼 할당 절감, 커널 간 파이프라이닝 기회, 그리고 웨이브 양자화 완화 — 메가커널(megakernel) 기법 \[12\] 이 LLM 추론에서 겨냥하는 것과 동일한 비효율성입니다. 프로덕션에서는 공유된 K/V 투영이 선형 투영 비용에 대한 추가 절감을 제공하며, 이는 KV 캐시 재사용과 유사합니다.
> The gains at short sequences stem from kernel fusion benefits: reduced launch overhead, shared buffer allocation savings, cross-kernel pipelining opportunities, and wave quantization mitigation — the same inefficiencies that megakernel techniques \[12\] target in LLM inference. In production, the shared K/V projections provide additional savings on linear projection cost, analogous to KV cache reuse.

## 4. 벤치마크 및 결과 요약 / Summary of Benchmarks and Results

이번 글에서 제시한 커널 수준 벤치마크를 종단 간 배포 성과와 함께 요약합니다. 아래의 모든 커널 벤치마크는 H100 SXM5에서 수행되었습니다(자세한 내용은 부록 1).
> We summarize the kernel-level benchmarks presented in this post alongside end-to-end deployment outcomes. All kernel benchmarks below are on H100 SXM5 (see details in Appendix 1).

- **선형 압축(2절).** 네 단계의 점진적 공동 설계 — 행렬곱 분해, 메모리 정렬, 브로드캐스트 융합, 그리고 TLX를 통한 워프 특화 다단계 융합 — 는 대표 설정에서 누적 약 4배의 속도 향상(1.944 ms → 0.482 ms)을 산출합니다. 이 이득은 배치 크기와 후보-사용자 비율 전반에서 견고하게 유지됩니다(그림 9).
- **Flash Attention(3절).** IKBO는 타깃 어텐션을 IO 바운드(~60 FLOPs/Byte)에서 연산 바운드(~833 FLOPs/Byte)로 전환하여, 621 BF16 TFLOPs로 공동 설계되지 않은 CuTeDSL FA4-Hopper 대비 2.4배/6.4배의 처리량을 달성합니다(커널만 / 커널 + 브로드캐스팅 기준).
- **종단 간 배포.** IKBO는 Meta의 RecSys 추론 스택 전반 — 초기 단계부터 후기 단계 랭킹 모델까지, GPU와 MTIA 가속기 양쪽에 — 폭넓게 배포되어, 공동 설계된 모델에서 연산 집약적인 순 지연 시간을 최대 2/3까지 감소시킵니다. IKBO는 약 10,000 : 1부터 약 10 : 1까지 이르는 후보-사용자 브로드캐스트 비율 전반에서 검증되어, 여러 작업 부하에 걸친 수치적 안정성과 확장성을 모두 확인했습니다.

> - **Linear Compression (Section 2).** Four progressive co-design stages — matmul decomposition, memory alignment, broadcast fusion, and warp-specialized multi-stage fusion via TLX — yield a cumulative ~4× speedup (1.944 ms → 0.482 ms) at representative settings. Gains remain robust across batch sizes and candidate-to-user ratios (Fig. 9).
> - **Flash Attention (Section 3).** IKBO shifts target attention from IO-bound (~60 FLOPs/Byte) to compute-bound (~833 FLOPs/Byte), achieving 2.4×/6.4× the throughput of non-co-designed CuTeDSL FA4-Hopper (kernel only / kernel + broadcasting) with 621 BF16 TFLOPs.
> - **End-to-end deployment.** IKBO has been deployed broadly across Meta’s RecSys inference stack — from early-stage to late-stage ranking models, on both GPU and MTIA accelerators — delivering up to 2/3 reduction in compute-intensive net latency on co-designed models. IKBO has been validated across candidate-to-user broadcast ratios spanning from ~10,000 : 1 down to ~10 : 1, confirming both numerical stability and scalability across workloads.

## 5. 결론 및 향후 방향 / Conclusion and Future Directions

IKBO는 브로드캐스트가 — 오랫동안 사용자-후보 상호작용의 피할 수 없는 비용으로 여겨져 온 것이 — 커널-모델-시스템 공동 설계를 통해 연산 프리미티브 계층에서 제거될 수 있음을 보여줍니다. 브로드캐스트 의미론을 커널에 직접 인코딩함으로써, 복제된 텐서가 결코 실체화되지 않으며, 절감액은 후보-사용자 비율에 따라 자연스럽게 확장됩니다.
> IKBO demonstrates that broadcast — long treated as an unavoidable cost of user-candidate interaction — can be eliminated at the computational primitive layer through kernel-model-system co-design. By encoding broadcast semantics directly into kernels, no replicated tensors ever materialize, and savings scale naturally with the candidate-to-user ratio.

이 작업에서 제시한 커널 구현은 Triton과 TLX를 통해 NVIDIA Hopper를 겨냥하지만, 핵심 아이디어 — 실체화된 브로드캐스트를 인덱스 기반 커널 내 조회로 대체하는 것 — 는 하드웨어 벤더에 독립적입니다. IKBO 커널을 CuTeDSL(고급 NVIDIA 백엔드 지원용)로 적응시키고 AMD CK 지원을 완성하는 것이 자연스러운 다음 단계입니다.
> While the kernel implementations presented in this work target NVIDIA Hopper via Triton and TLX, the core idea — replacing materialized broadcasts with index-driven in-kernel lookups — is hardware-vendor independent. Adapting the IKBO kernels to CuTeDSL (for advanced NVIDIA backend support) and completing the AMD CK support are natural next steps.

여기서 제시한 2단계 사용자-후보 계층을 넘어, 일부 RecSys 시나리오는 더 깊은 계층 구조를 수반합니다 — 예를 들어 사용자 → 광고 벤더(ads vendor) → 광고 항목(ads item)으로, 각 사용자가 여러 벤더를 보고 각 벤더가 여러 항목을 제공하는 경우입니다. 이는 독립적이고 비균일한 비율을 갖는 두 개의 중첩된 브로드캐스트 관계를 도입합니다. IKBO는 이를 우아하게 처리할 수 있으며, 이를 다중 계층(multi-level) 작업 부하에 적용하는 것은 프로덕션 RecSys 아키텍처에서 실체화 오버헤드를 더욱 줄이기 위한 자연스러운 방향입니다.
> Beyond the two-level user-candidate hierarchy presented here, some RecSys scenarios involve deeper hierarchies — for example, user → ads vendor → ads item, where each user sees multiple vendors and each vendor offers multiple items. This introduces two nested broadcast relationships with independent, non-uniform ratios. IKBO can handle this elegantly, and applying it to multi-level workloads is a natural direction for further reducing materialization overhead in production RecSys architectures.

## 감사의 글 / Acknowledgements

Triton 및 TLX 기반에 대한 강력한 내부 지원, 강력한 Triton 프로파일링 도구, 그리고 이 작업 전반에 걸쳐 Triton 관련 이슈를 신속하게 해결해 준 [Hongtao Yu](mailto:hoy@meta.com), [Yuanwei (Kevin) Fang](mailto:fywkevin@meta.com), [Daohang Shi](mailto:daohang@meta.com), [Yueming Hao](mailto:yhao@meta.com), [Srivatsan Ramesh](mailto:srir@meta.com), [Manman Ren](mailto:mren@meta.com) 에게 감사드립니다.
> We are grateful to [Hongtao Yu](mailto:hoy@meta.com), [Yuanwei (Kevin) Fang](mailto:fywkevin@meta.com), [Daohang Shi](mailto:daohang@meta.com), [Yueming Hao](mailto:yhao@meta.com), [Srivatsan Ramesh](mailto:srir@meta.com) and [Manman Ren](mailto:mren@meta.com) for their strong internal support of the Triton and TLX foundation, the powerful Triton profiling toolings, and for promptly resolving Triton-related issues throughout this work.

이 글의 명료성을 크게 개선해 준 통찰력 있는 피드백을 준 [Chris Gottbrath](mailto:gottbrath@meta.com) 에게 감사드립니다. 원활한 리뷰 과정을 도와준 데에도 크게 감사드립니다.
> Thanks [Chris Gottbrath](mailto:gottbrath@meta.com) for his insightful feedback, which significantly improved the clarity of this post. We also greatly appreciate his help in facilitating a smooth review process.

리더십 지원을 해준 [Santanu Kolay](mailto:skolay@meta.com), [Sandeep Pandey](mailto:sppandey@meta.com), [Matt Steiner](mailto:mattsteiner@meta.com), [GP Musumeci](mailto:gpmusumeci@meta.com), [Ashwin Kumar](mailto:ashwink3029@meta.com), [Ian Barber](mailto:ianbarber@meta.com), [Aparna Ramani](mailto:apr@meta.com), [CQ Tang](mailto:tang@meta.com) 에게 감사드립니다.
> Thanks [Santanu Kolay](mailto:skolay@meta.com), [Sandeep Pandey](mailto:sppandey@meta.com), [Matt Steiner](mailto:mattsteiner@meta.com), [GP Musumeci](mailto:gpmusumeci@meta.com), [Ashwin Kumar](mailto:ashwink3029@meta.com), [Ian Barber](mailto:ianbarber@meta.com), [Aparna Ramani](mailto:apr@meta.com), [CQ Tang](mailto:tang@meta.com) for leadership support.

## 참고문헌 / References

\[1\] Naumov, M., et al. “Deep Learning Recommendation Model for Personalization and Recommendation Systems,” arXiv:1906.00091, 2019.

\[2\] Wang, R., et al. “Deep & Cross Network for Ad Click Predictions,” ADKDD, 2017.

\[3\] Zhai, J., et al. “Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations,” ICML, 2024.

\[4\] xAI. “Phoenix: Recommendation System,” GitHub, 2026. https://github.com/xai-org/x-algorithm

\[5\] Guo, L., et al. “Request-Only Optimization for Recommendation Systems,” arXiv:2508.05640, 2025.

\[6\] Zhang, B., et al. “Wukong: Towards a Scaling Law for Large-Scale Recommendation,” ICML, 2024.

\[7\] Vaswani, A., et al. “Attention Is All You Need,” NeurIPS, 2017.

\[8\] Lee, J., et al. “Set Transformer: A Framework for Attention-based Permutation-Invariant Input,” ICML, 2019.

\[9\] Dao, T. “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning,” ICLR, 2024.

\[10\] Shah, J., et al. “FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision,” NeurIPS, 2024.

\[11\] Zadouri, T., et al. “FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling,” arXiv:2603.05451, 2026.

\[12\] Spector, B., et al. “Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B,” Hazy Research Blog, 2025. https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles

## 부록 / Appendix

### 부록 1. 벤치마크 설정 / Appendix 1. Benchmark Setup

모든 실험은 단일 NVIDIA H100 SXM5 GPU(700 W TDP, 96 GB HBM2e)에서 다음 소프트웨어 스택으로 수행되었습니다.
> All experiments are conducted on a single NVIDIA H100 SXM5 GPU (700 W TDP, 96 GB HBM2e) with the following software stack:

- CUDA: 12.4
- PyTorch: 2.11.0a0+fb (내부 빌드)
- Triton: facebookexperimental/triton@`4059e79bf` ([#831](https://github.com/facebookexperimental/triton/pull/831))

> - CUDA: 12.4
> - PyTorch: 2.11.0a0+fb (internal build)
> - Triton: facebookexperimental/triton@`4059e79bf` ([#831](https://github.com/facebookexperimental/triton/pull/831))

### 부록 2. 산술 강도 분석 / Appendix 2. Arithmetic Intensity Analysis

**2.1 H100 SXM5의 기계 균형점 (700 W TDP, 96 GB HBM2E) / Machine Balance Point of H100 SXM5**

![H100 SXM5의 기계 균형점 / Machine Balance Point of H100 SXM5](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/Screenshot-2026-05-02-at-10.58.29-AM.jpg){:style="width:100%"}

**2.2 기준선 LCE의 산술 강도 / Arithmetic Intensity of the Baseline LCE**

FP16의 배치 행렬곱 `(M, K) @ (B, K, N) → (B, M, N)` 에 대해, B=1024, M=433, K=2044, N=256:
> For a batched matmul `(M, K) @ (B, K, N) → (B, M, N)` in FP16, with B=1024, M=433, K=2044, N=256:

![기준선 LCE의 산술 강도 / Arithmetic Intensity of the Baseline LCE](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/Screenshot-2026-05-02-at-10.59.22-AM.jpg){:style="width:100%"}

### 부록 3. 2.1절 상세 결과 분석 / Appendix 3. Detailed Result Analysis for Section 2.1

*설정:* H100 SXM5(부록 1), PyTorch eager 모드(커널 융합 없음), 추론. 형태(shape)는 대표적인 구성에서 가져왔습니다.
> *Setup:* H100 SXM5 (Appendix 1), PyTorch eager mode (no kernel fusion), inference. Shapes from a representative configuration.

<table><tbody><tr><td><b>버전 / Version</b></td><td><b>총합 / Total (ms)</b></td><td><b>커널 / Kernels</b></td><td><b>지연 시간 / Latency (ms)</b></td><td><b>DRAM (GB)</b></td><td><b>L1/TEX 섹터 / Sectors (M)</b></td><td><b>연산 / Compute (GFLOPs)*</b></td><td><b>병목 / Bottleneck †</b></td></tr><tr><td>기준선 / Baseline</td><td>1.944</td><td>1 CUTLASS GEMM</td><td>1.944</td><td>1.31</td><td>798</td><td>460</td><td>L1/TEX (89%)</td></tr><tr><td rowspan="2">분해 / Decomposition</td><td rowspan="2">1.389</td><td>2 CUTLASS GEMM (사용자 + 후보 행렬곱)</td><td>0.984</td><td>0.68</td><td>351</td><td>200</td><td>L1/TEX (84%)</td></tr><tr><td>1 ATen Gather + 1 ATen add</td><td>0.405</td><td>0.87</td><td>36</td><td>0.11</td><td>DRAM (92%)</td></tr></tbody></table>

\*실행된 총 FLOPs이며, 처리량이 아닙니다.
†병목은 NCU Speed of Light 분석으로 식별했습니다. 방법론은 부록 4에 있습니다.
> \*Total FLOPs executed, not throughput.
> †Bottleneck identified via NCU Speed of Light analysis; methodology in Appendix 4.

중복 제거는 사용자 측 작업의 98% 이상을 제거하고(배치 1024 → 약 15), L1/TEX 섹터를 798M에서 351M으로, GEMM 지연 시간을 1.944 ms에서 0.984 ms로 줄입니다. GEMM 이후의 브로드캐스트와 덧셈은 0.405 ms(DRAM 바운드)가 소요되어, 0.555 ms의 순절감을 산출합니다.
> Deduplication eliminates >98% of user-side work (batch 1024 → ~15), cutting L1/TEX sectors from 798M to 351M and GEMM latency from 1.944 ms to 0.984 ms. The post-GEMM broadcast and addition costs 0.405 ms (DRAM-bound), yielding a net saving of 0.555 ms.

**정밀도 참고.** 기준선은 모든 K 곱을 단일 FP32/TF32 축약(reduction)으로 누적합니다. 분해는 K_user와 K_cand를 별도로 누적한 뒤, 부분 결과를 BF16/FP16에서 합산합니다. 학습도 동일한 분해를 사용하므로, 수치가 종단 간으로 일치합니다. 정확한 추론 동등성을 위해서는, 융합 커널(2.4절)이 최종 합산을 FP32에서 수행할 수 있습니다.
> **Precision note.** The baseline accumulates all K products in a single FP32/TF32 reduction. Decomposition accumulates K\_user and K\_cand separately, then sums the partial results in BF16/FP16. Training uses the same decomposition, so numerics match end-to-end. For exact inference parity, a fused kernel (Section 2.4) can perform the final summation in FP32.

### 부록 4. 병목 분석 방법론 / Appendix 4. Bottleneck Analysis Methodology

루프라인 분석 이후 더 자세히 들여다보기 위해, NCU의 Speed of Light 분석을 사용하여 하드웨어 서브시스템 병목을 식별합니다. 병목은 최대 지속 처리량(peak sustained throughput) 대비 사용률이 가장 높은 서브시스템입니다. 2.1절의 분석을 위해, 세 가지 지표를 모니터링합니다.
> For a closer look after roofline analysis, we use NCU’s Speed of Light analysis to identify hardware subsystem bottlenecks. The bottleneck is the subsystem with the highest utilization relative to its peak sustained throughput. For the analysis in Section 2.1, we monitor three metrics:

**연산(Compute)** 은 NCU가 직접 보고하는 최대 SM 파이프라인 사용률(`Compute (SM) Throughput`)입니다. 가장 활발한 실행 파이프라인(GEMM의 경우 텐서 코어)이 최대 명령 속도 대비 얼마나 바쁜지를 측정합니다.
> **Compute** is the peak SM pipeline utilization, reported directly by NCU (`Compute (SM) Throughput`). It measures how busy the most active execution pipeline (tensor cores for GEMMs) is relative to its peak instruction rate.

**L1/TEX** 사용률은 아래와 같이 L1/TEX 유닛이 처리해야 하는 총 섹터에서 유도되며, 여기서 num_L1_tex_sectors는 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` 과 `_st.sum` 카운터, SM_active_cycles는 `sm__cycles_active.avg` 카운터, num_SM은 132, 그리고 num_sustained_peak_sectors_per_sm_per_cycle는 H100에서 2.0입니다.
> **L1/TEX** utilization is derived from the total sectors the L1/TEX unit must process as below, where num\_L1\_tex\_sectors is `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` and `_st.sum` counter, is SM\_active\_cycles `sm__cycles_active.avg` counter, num\_SM is 132 and num\_sustained\_peak\_sectors\_per\_sm\_per\_cycle is 2.0 on H100.

![L1/TEX 사용률 유도 / L1/TEX utilization derivation](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/Screenshot-2026-05-02-at-11.02.44-AM.jpg){:style="width:100%"}

**DRAM** 사용률은 아래와 같이 전송된 총 HBM 바이트에서 유도되며, 여기서 dram_bytes_read_and_write는 `dram__bytes_read.sum` 과 `dram__bytes_write.sum` 카운터입니다. peak_bandwidth는 테스트 GPU 서버에서 2TB/s입니다.
> **DRAM** utilization is derived from total HBM bytes transferred as below, where dram\_bytes\_read\_and\_write is the dram\_\_bytes\_read.sum and `dram__bytes_write.sum` counter. peak\_bandwidth is 2TB/s on the testing GPU server.

![DRAM 사용률 유도 / DRAM utilization derivation](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/Screenshot-2026-05-02-at-11.03.36-AM.jpg){:style="width:100%"}

### 부록 5. 2.2절 상세 결과 분석 / Appendix 5. Detailed Result Analysis for Section 2.2

**결과.** 1.389 ms → 0.798 ms (**42.5% 감소**).
> **Result.** 1.389 ms → 0.798 ms (**42.5% reduction**).

<table><tbody><tr><td><b>버전 / Version</b></td><td><b>총 지연 시간 / Total Latency (ms)</b></td><td><b>커널 / Kernels</b></td><td><b>지연 시간 / Latency (ms)</b></td><td><b>DRAM 트래픽 / Traffic (GB)</b></td><td><b>연산 / Compute (GFLOPs) *속도 아님</b></td><td><b>L1/TEX 섹터 / Sectors (M)</b></td><td><b>병목 / Bottleneck †</b></td></tr><tr><td rowspan="2">분해 / Decomposition (패딩 없음 / unpadded)</td><td rowspan="2">1.386</td><td>2 CUTLASS GEMM – 사용자 & 후보 행렬곱</td><td>0.984</td><td>0.68</td><td>200</td><td>351</td><td>L1/TEX (84%)</td></tr><tr><td>1 ATen Gather – 브로드캐스트<br>1 ATen Elementwise – add</td><td>0.402</td><td>0.87</td><td>0.11</td><td>36</td><td>DRAM (92%)</td></tr><tr><td rowspan="2">분해 / Decomposition (패딩된 K / padded K)</td><td rowspan="2"><b>0.798</b></td><td>2 CUTLASS GEMM – 사용자 & 후보 행렬곱</td><td><b>0.400</b></td><td>0.69</td><td>200</td><td><b>0</b></td><td><b>균형 / Balanced</b></td></tr><tr><td>1 ATen Gather – 브로드캐스트<br>1 ATen Elementwise – add</td><td>0.398</td><td>0.87</td><td>0.11</td><td>36</td><td>DRAM (92%)</td></tr></tbody></table>

큰 속도 향상 뒤에는 두 가지 요인이 있습니다.
> Two factors behind the large speedup.

- **TMA.** 정렬된 행렬에서, CUTLASS는 TMA 기반 커널을 선택하여 L1/TEX를 완전히 우회합니다(섹터 → 0). 패딩되지 않은 커널은 또한 행렬 `B` 에 불필요한 페널티를 주었습니다. `B`(정렬된 `N` 포함)가 128비트 로드를 사용할 수 있었음에도 *두* 행렬 모두에 4바이트 로드를 적용했습니다.
- **뱅크 충돌(Bank conflicts).** 패딩되지 않은 커널은 또한 스위즐(swizzle) 패턴이 4바이트 cp.async 쓰기를 보호하지 못하는 sm80 MMA 경로를 사용하여, 많은 공유 메모리 뱅크 충돌을 일으킵니다. 패딩된 커널에는 이 문제가 없습니다.

> - **TMA.** With aligned matrices, CUTLASS selects a TMA-based kernel, bypassing L1/TEX entirely (sectors → 0). The unpadded kernel also penalized matrix `B` unnecessarily: it applied 4-byte loads to *both* matrices, even though `B` (with aligned `N`) could have used 128-bit loads.
> - **Bank conflicts.** The unpadded kernel also uses sm80 MMA path whose swizzle pattern doesn’t protect against 4-byte cp.async writes, causing many shared memory bank conflicts. The padded kernel doesn’t have this issue.

### 부록 6. 2.3절 상세 결과 분석 / Appendix 6. Detailed Result Analysis for Section 2.3

**결과.** 지연 시간: 0.798 ms → 0.580 ms (**27.4% 감소**).
> **Result.** Latency: 0.798 ms → 0.580 ms (**27.4% reduction**).

<table><tbody><tr><td>버전 / Version</td><td>총 지연 시간 / Total Latency (ms)</td><td>커널 / Kernels</td><td>지연 시간 / Latency (ms)</td><td>DRAM 트래픽 / Traffic (GB)</td></tr><tr><td rowspan="2">분해 / Decomposition (패딩된 K / padded K)</td><td rowspan="2">0.798</td><td>2 CUTLASS GEMM – 사용자 & 후보 행렬곱</td><td>0.400</td><td>0.68</td></tr><tr><td>1 ATen Gather – 브로드캐스트<br>1 ATen Elementwise – add</td><td>0.398</td><td>0.87</td></tr><tr><td>iKBO 융합 / iKBO Fusion</td><td>0.580</td><td>사용자 GEMM & 후보 iKBO 커널</td><td>0.580</td><td>0.68</td></tr></tbody></table>

0.87 GB의 중간 DRAM 트래픽이 예상대로 제거됩니다. NCU 프로파일링은 추가 기회를 드러냅니다. 점유율은 스케줄러당 워프 1개로 6.25%에 불과하며, PC 샘플링은 사이클의 23%만이 생산적임을 보여줍니다.
> The 0.87 GB of intermediate DRAM traffic is eliminated as expected. NCU profiling reveals further opportunity: occupancy is just 6.25% with 1 warp per scheduler, and PC sampling shows only 23% of cycles are productive:

| 스톨 원인 / Stall Reason | 비율 / Percentage | 커널에서 주로 가리키는 것 / What it mainly refers in the kernel |
|---|---|---|
| Stall long scoreboard | 41.8% | 전역 메모리 로드 / Global memory loads |
| Selected (executing) | 23.1% | 생산적 작업(좋음) – 실제로 발행된 명령 / Productive work (good) – instructions actually issued |
| Stall wait | 20.1% | WGMMA 대기 / Wait WGMMA |
| Stall barrier | 5.7% | 소프트웨어 파이프라인 스테이지 간 `bar.sync` / `bar.sync` between software-pipeline stages |

스케줄러당 워프 1개로는, 모든 스톨이 그대로 노출됩니다. 전환할 다른 워프가 없기 때문입니다. 파이프라인 깊이를 줄여 점유율을 높이면 K-루프 지연 은닉을 희생하게 됩니다. 이 커널에는 까다로운 상황입니다. 텐서 코어 처리량을 유지하려면 큰 타일과 깊은 파이프라인이 필요하지만, 이들은 공유 메모리 예산의 대부분을 소비하여 점유율을 통해 지연을 숨길 여지를 거의 남기지 않습니다.
> With 1 warp per scheduler, every stall is fully exposed: there is no other warp to switch to. Increasing occupancy by reducing pipeline depth would sacrifice K-loop latency hiding. This is a challenging situation for this kernel: large tiles and deep pipelines are needed to keep the tensor cores throughput, but they consume most of the shared memory budget, leaving little room to hide latency through occupancy.

### 부록 7. 릴리스-어콰이어 동기화 프로토콜 / Appendix 7. Release-Acquire Synchronization Protocol

**프로듀서 (사용자 CTA).** 사용자 타일을 전역 메모리에 저장한 후, CTA는 릴리스 의미론(release semantics)으로 타일별 플래그를 설정하여, 플래그 쓰기 이전에 데이터 가시성(visibility)을 보장합니다.
> **Producer (user CTA).** After storing a user tile to global memory, the CTA sets a per-tile flag with release semantics, ensuring data visibility before the flag write:

```python
tl.atomic_add(user_tile_flag_ptr, 1, sem="release", scope="gpu")
```

**컨슈머 (후보 CTA).** 워프 그룹당 단일 스레드가 스핀 중 메모리 트래픽을 최소화하기 위해 `ld.relaxed` 로 플래그를 폴링합니다. 플래그가 전환되면, 단일 `ld.acquire` 가 선행 발생 관계를 확립하고, 명명된 배리어가 워프 그룹의 128개 스레드 전체에 준비 완료를 브로드캐스트합니다.
> **Consumer (candidate CTA).** A single thread per warp group polls the flag with `ld.relaxed` to minimize memory traffic during the spin. Once the flag transitions, a single `ld.acquire` establishes the happens-before edge, and a named barrier broadcasts readiness to all 128 threads in the warp group:

```python
if tlx.thread_id(axis=0) % 128 == 0:  # 워프 그룹(4 워프)당 1 스레드
    ready = tl.inline_asm_elementwise(
        "ld.relaxed.gpu.global.b32 $0, [$1];", "=r,l",
        [user_tile_flag_ptr], dtype=tl.int32, is_pure=False, pack=1)
    while ready == 0:
        ready = tl.inline_asm_elementwise(
            "nanosleep.u32 50; ld.relaxed.gpu.global.b32 $0, [$1];", "=r,l",
            [user_tile_flag_ptr], dtype=tl.int32, is_pure=False, pack=1)
    tl.inline_asm_elementwise(
        "ld.acquire.gpu.global.b32 $0, [$1];", "=r,l",
        [user_tile_flag_ptr], dtype=tl.int32, is_pure=False, pack=1)
tlx.named_barrier_wait(12, 128)
```

### 부록 8. TLX vs. Triton NCU 프로파일링 지표 / Appendix 8. NCU Profiling Metrics for TLX vs. Triton

| 지표 / Metric | Triton | TLX | 비고 / Notes |
|---|---|---|---|
| 이론적 점유율 / Theoretical Occupancy | 6.25% | 18.75% | CTA당 워프 그룹 3개 vs. 1개 |
| DRAM 처리량 / Throughput (dram__cycles_active.avg.pct_of_peak_sustained_elapsed) | 38.51% | 52.39% | 지속적인 TMA 로드로 사용률 증가 |
| L2 캐시 처리량 / L2 Cache Throughput (lts__throughput.avg.pct_of_peak_sustained_elapsed) | 73.69% | 83.86% | 병목. TLX가 최대치에 더 가깝게 밀어올림 |

### 부록 9. 일반 flash attention vs IKBO flash attention 루프라인 분석 / Appendix 9. Roofline analysis of normal flash attention vs IKBO flash attention

산술 강도(AI)는 FP16/BF16 정밀도, user_seq_len = 1024, n_seed = 64, B_candidate(식에서 B) : B_user(식에서 B/num_cand_user) = 70: 1 조건으로 계산됩니다.
> Arithmetic intensity (AI) is calculated given FP16/BF16 precision, user\_seq\_len = 1024, n\_seed = 64, B\_candidate (B in eq) : B\_user (B/num\_cand\_user in eq) = 70: 1.

![일반 flash attention vs IKBO flash attention 루프라인 분석 / Roofline analysis of normal flash attention vs IKBO flash attention](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/Screenshot-2026-05-02-at-11.06.46-AM-scaled.jpg){:style="width:100%"}

### 부록 10. IKBO TLX FA3의 SMEM 소비량 / Appendix 10. SMEM consumption of IKBO TLX FA3

| SMEM 버퍼 / SMEM buffer | 개수 / Counts | 블록 차원 / Block dim | 총 크기 / Total size |
|---|---|---|---|
| Query | 2 (컨슈머 그룹당 1개) | 64 * 128 (2Bytes) | 32KB |
| Key | 2 | 128 * 128 (2Bytes) | 64KB |
| Value | 2 | 128 * 128 (2Bytes) | 64KB |
| Output | 2 (컨슈머 그룹당 1개) | 64 * 128 (2Bytes) | 32KB |
| 총합 / Total | | | 192KB |

### 부록 11. RecSys 경계 조건에서 IKBO FA vs CuTeDSL FA4 Hopper 및 TLX FA3 Hopper 커널 벤치마킹 / Appendix 11. Benchmarking IKBO FA vs CuTeDSL FA4 Hopper and TLX FA3 Hopper kernel under RecSys boundary condition

IKBO 커널은 기본적으로 GQA와 유사한 IO 및 연산 패턴을 공유하는 사용자-후보 상호작용 매핑 로직을 활성화합니다. 벤치마킹 동안, IKBO 커널에는 안정적인 B_candidate : B_user = 64 : 1을 적용하고, CuTeDSL FA4 Hopper GQA 버전에는 유사한 연산 패턴을 적용했습니다(2-컨슈머 워프그룹이 완벽하게 동작하도록 Q_seq_len = 128). 추가로 언급할 만한 점은, IKBO 커널은 실시간으로 랭킹될 다양한 수의 후보를 처리하기 위해 후보-사용자 매핑 텐서를 추가로 소비해야 한다는 것입니다.
> IKBO kernel is basically enabling the user-candidate interaction mapping logic which shares a similar IO and computation pattern as GQA. During benchmarking, a stable B\_candidate : B\_user = 64 : 1 is applied for IKBO kernel and similar compute patterns for CuTeDSL FA4 Hopper GQA version (Q\_seq\_len = 128 to make sure 2-consumer warpgroup to work perfectly). Worth additional mentioning, IKBO kernel still needs to extra consume the candidate-user mapping tensor to handle a varied number of candidates to be ranked in real time.

| 커널 유형 / Kernel type | 처리량 / Throughput (TFLOPs/s) | IO (GB/s) |
|---|---|---|
| Triton IKBO FA2 | 425 | 519 |
| TLX IKBO FA3 | 418 | 510 |
| TLX IKBO FA3 persistent | 592 | 723 |
| TLX IKBO FA3 persistent V2 (reverse k,v order) | 537 | 655 |
| CuTeDSL FA4 Hopper GQA | 518 | 633 |
| TLX FA3 GQA | 576 | 703 |

오픈소스 GQA 커널 대비 벤치마크한 IKBO FA. IKBO 커널의 Q, K, V 형태는 \[배치 크기, 헤드 수, 시퀀스, d_head\] 순서. Q_ikbo \[2048, 2, 64, 128\], K/V_ikbo \[32, 2, 1024, 128\]. GQA 커널의 Q, K, V 형태 Q_gqa \[1024, 2, 128, 128\], K/V_gqa \[32, 2, 1024, 128\].
> IKBO FA benchmarked vs open-source GQA kernel. Q, K, V shape for IKBO kernel in the sequence of \[Batch size, num head, seq, d\_head\] Q\_ikbo \[2048, 2, 64, 128\], K/V\_ikbo \[32, 2, 1024, 128\]. Q, K, V shape for GQA kernel Q\_gqa \[1024, 2, 128, 128\], K/V\_gqa \[32, 2, 1024, 128\]

| 커널 유형 / Kernel type | 처리량 / Throughput (TFLOPs/s) | IO (GB/s) |
|---|---|---|
| Triton IKBO FA2 | 449 | 329 |
| TLX IKBO FA3 | 470 | 345 |
| TLX IKBO FA3 persistent | 621 | 455 |
| TLX IKBO FA3 persistent V2 (reverse k,v order) | 587 | 430 |
| CuTeDSL FA4 Hopper GQA | 608 | 445 |
| TLX FA3 GQA | 628 | 460 |

오픈소스 GQA 커널 대비 벤치마크한 IKBO FA. Q_ikbo \[2048, 2, 64, 128\], K/V_ikbo \[32, 2, 2048, 128\]. GQA 커널의 Q, K, V 형태 Q_gqa \[1024, 2, 128, 128\], K/V_gqa \[32, 2, 2048, 128\].
> IKBO FA benchmarked vs open-source GQA kernel. Q\_ikbo \[2048, 2, 64, 128\], K/V\_ikbo \[32, 2, 2048, 128\]. Q, K, V shape for GQA kernel Q\_gqa \[1024, 2, 128, 128\], K/V\_gqa \[32, 2,2048, 128\]

**참고:** 표준 Flash Attention 커널은 IKBO 로직을 포함하지 않으므로, cuteDSL 버전의 처리량 결과를 시뮬레이션하기 위해 유사한 IO 비용과 FLOPs 소비를 갖는 GQA 구성을 사용합니다.
> **Note:** Since standard Flash Attention kernels do not incorporate IKBO logic, we use a GQA configuration with similar IO cost and FLOPs consumption to simulate throughput results for cuteDSL versions.

### 부록 12: 명령어 캐시 미스가 컨슈머-2 워프그룹에 상당한 지연을 유발 / Appendix 12: Instruction cache miss cause significant delay on the consumer-2 warpgroup

![명령어 캐시 미스 수정 전후 결과 / Instruction cache miss result before and after the fix](/assets/blog/2026-05-05-in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/Screenshot-2026-05-02-at-11.07.37-AM.jpg){:style="width:100%"}
*그림 A1. 명령어 캐시 미스 수정 전후 결과. / Fig. A1. Instruction cache miss result before and after the fix*

```
명령어 캐시 미스 수정 전:
    ---------------------------------------------------- ----------- ------------
    Metric Name                                          Metric Unit Metric Value
    ---------------------------------------------------- ----------- ------------
    gcc__cache_requests_type_instruction.sum                              319,394
    gcc__cache_requests_type_instruction_lookup_miss.sum                    7,234
    sm__icc_requests.sum                                       cycle    6,049,376
    sm__icc_requests_lookup_hit.sum                            cycle    5,438,421
    sm__icc_requests_lookup_miss.sum                           cycle      610,955
    ---------------------------------------------------- ----------- ------------

명령어 캐시 미스 수정 후:
    ---------------------------------------------------- ----------- ------------
    Metric Name                                          Metric Unit Metric Value
    ---------------------------------------------------- ----------- ------------
    gcc__cache_requests_type_instruction.sum                               33,008
    gcc__cache_requests_type_instruction_lookup_miss.sum                      769
    sm__icc_requests.sum                                       cycle      792,437
    sm__icc_requests_lookup_hit.sum                            cycle      722,244
    sm__icc_requests_lookup_miss.sum                           cycle       70,193
    ---------------------------------------------------- ----------- ------------
```
