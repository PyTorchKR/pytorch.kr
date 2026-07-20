---
layout: blog_detail
title: "비용 없는 정규화를 향해: 정규화를 GEMM과 어텐션 커널에 융합하기"
author: Jacky (Junqing) Zhou, Hongtao Yu, Jackie (Jiaqi) Xu, Menglu Yu, Ethan Che, Han Xu, Darren Liu, Peng Chen (Dev Infra), Daohang Shi, Max Leung
category: ["pytorch.org", "translation"]
org_title: "Towards Free Normalization: Fusing Normalization into GEMM and Attention Kernels"
org_link: https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/
---

## TL;DR

이번 글에서는 LayerNorm과 RMSNorm 같은 일반적인 정규화(normalization) 연산을 위한 다양하고 새로운 커널 융합(kernel fusion) 기법을 소개합니다. 이 기법들은 메모리 바운드(memory-bound) 성격이 매우 강한 정규화 커널의 메모리 IO 오버헤드를 줄여 상당한 속도 향상을 제공합니다. 먼저 LLM과 광고 추천 모델 양쪽에서 흔히 쓰이는 정규화 연산이 모델링에서 갖는 중요성과 성능 과제를 간략히 살펴본 뒤, Lazy Pre-Norm과 Multi-CTA Norm Fusion을 포함해 성능 병목을 공략하는 새로운 전략들을 소개합니다. 이러한 기법으로 정규화 커널 지연 시간의 최대 **90%** 를 GEMM과의 융합으로 숨길 수 있음을 보입니다. 마지막으로 GDPA \[1\] 같은 어텐션 커널 주변의 여러 정규화에 융합을 적용한 FlashNormAttention 알고리즘을 소개하며, 최대 **35%** 의 커널 속도 향상을 달성합니다.
> In this blog post, we present various novel kernel fusion techniques for common normalization ops like LayerNorm and RMSNorm, which provide significant speedup by reducing the memory-IO overhead of these highly memory-bound kernels. We start with a brief overview of the modeling importance as well as performance challenges of normalization ops common in both LLMs and ads recommendation models, then present the novel strategies in tackling the performance bottlenecks, including Lazy Pre-Norm and Multi-CTA Norm Fusion. We show that such techniques can hide as much as **90%** of a normalization kernel’s latency by fusing with GEMMs. In the end, we present the FlashNormAttention algorithm where we apply fusion for multiple normalizations around an attention kernel like GDPA \[1\], achieving up to **35%** kernel speedup.

이 작업은 주로 두 가지 커널 DSL로 수행되었습니다. 하나는 GPU 실행 제어를 위한 더 낮은 수준의 하드웨어 인지(hardware-aware) 지원을 갖춘 Triton DSL 확장 집합인 [TLX](https://arxiv.org/abs/2605.10905)이고, 다른 하나는 개발 속도, 이식성, 포괄적인 자동 튜닝(autotuning)에 강점이 있는 고수준 DSL인 [Helion](https://pytorch.kr/blog/2025/helion/)입니다. 벤치마크는 bfloat16 데이터 타입으로, 750W 전력 제한이 걸린 Meta 데이터센터의 NVIDIA B200 GPU에서 수행했습니다.
> This work is done with primarily two kernel DSLs: [TLX](https://arxiv.org/abs/2605.10905), a set of Triton DSL extensions with lower-level, hardware-aware support for GPU execution control; and [Helion](https://pytorch.org/blog/helion/), a high-level DSL that excels at developer velocity, portability, and comprehensive autotuning. Benchmarks are performed with data type bfloat16, on NVIDIA B200 GPUs in Meta’s data centers with a 750 W power cap.

코드는 다음에서 확인할 수 있습니다: [https://github.com/facebookresearch/ads\_model\_kernel\_library/tree/main/multi\_cta\_norm\_fusion](https://github.com/facebookresearch/ads_model_kernel_library/tree/main/multi_cta_norm_fusion) 및 [https://github.com/facebookresearch/ads\_model\_kernel\_library/tree/main/gdpa\_megakernel](https://github.com/facebookresearch/ads_model_kernel_library/tree/main/gdpa_megakernel)
> Code available at: [https://github.com/facebookresearch/ads\_model\_kernel\_library/tree/main/multi\_cta\_norm\_fusion](https://github.com/facebookresearch/ads_model_kernel_library/tree/main/multi_cta_norm_fusion) and [https://github.com/facebookresearch/ads\_model\_kernel\_library/tree/main/gdpa\_megakernel](https://github.com/facebookresearch/ads_model_kernel_library/tree/main/gdpa_megakernel)

## 소개 / Introduction

정규화 기법은 학습을 안정화하고 수렴을 가속하는 탁월한 효과 덕분에 대부분의 딥러닝 아키텍처에서 없어서는 안 될 요소가 되었습니다. 특히 가장 안쪽 임베딩 차원에 대한 전통적인 정규화(예: LayerNorm, RMSNorm)는 현대 대규모 언어 모델은 물론 Meta의 광고 모델 같은 추천 시스템(recsys) 모델에서도 가장 흔하고 어디에나 있는 유형입니다. 예를 들어 Meta의 최대 규모 추천 시스템 학습 파운데이션 모델인 [Generative Ads Model(GEM)](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/) \[3\] 에 배포된 [Kunlun](https://arxiv.org/abs/2602.10016) \[2\] 아키텍처에서는 Multi-Head Attention, Hierarchical Seed Pooling, GDPA 기반 \[1\] PFFN 등 거의 모든 핵심 구성 요소에 LayerNorm/RMSNorm이 존재합니다.
> Normalization techniques have become indispensable in most deep learning architectures due to their excellent effectiveness in stabilizing training and accelerating convergence. In particular, traditional normalization across the innermost embedding dimension (e.g. LayerNorm, RMSNorm) has been the most common and ubiquitous type in modern Large Language Models as well as recsys models like Meta’s ads models. For example, in the [Kunlun](https://arxiv.org/abs/2602.10016) \[2\] architecture deployed on Meta’s largest Recsys training foundation model, the [Generative Ads Model (GEM)](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/) \[3\], LayerNorm/RMSNorm exists in nearly all key components, such as Multi-Head Attention, Hierarchical Seed Pooling, and GDPA-enhanced \[1\] PFFN.

하지만 정규화가 어디에나 있다는 사실은 어려운 성능 과제도 함께 가져옵니다. 정규화는 텐서 코어(TensorCore)를 전혀 활용하지 못하는, 메모리 바운드 성격이 매우 강한 연산이기 때문입니다. 이는 모델 학습에서 하드웨어의 연산 능력을 최대로 끌어내는 데 걸림돌이 됩니다. Kunlun \[2\] 을 예로 들면, 정규화가 전체 학습 지연 시간의 약 20%를 차지합니다. 최적화하지 않으면 하드웨어 연산 처리량의 20%를 곧바로 잃는다는 뜻입니다. 연산 바운드(compute-bound) 성격이 더 강한 일반적인 LLM에서도 정규화는 여전히 전체 지연 시간의 약 10%를 차지할 수 있습니다.
> However, the ubiquity of normalization also brings a difficult performance challenge: it’s highly memory-bound with no TensorCore utilization. This hinders us from saturating hardware compute capabilities in model training. Using Kunlun \[2\] as an example, normalization takes up roughly 20% of the total training latency there. This means we immediately lose 20% of our hardware’s compute throughput without optimization. In a typical LLM which is more compute-bound, normalization could still take roughly 10% of total latency.

이를 해결하려면 정규화 커널을 IO를 인지하는(IO-aware) 방식으로 설계해야 합니다. 커널 융합으로 계산 정확도를 해치지 않으면서 메모리 IO 비용을 신중히 아끼고, 메모리와 CUDA 코어 부담이 큰 이 연산들을 텐서 코어 중심 연산과 겹쳐(overlap) 실행해야 합니다. 대부분의 정규화 연산은 행렬 곱셈(matmul) 연산(예: MLP, 어텐션)의 앞이나 뒤에 오므로, 이 작업은 정규화를 행렬 곱셈과 효율적으로 융합하는 방법에 집중합니다. 먼저 정규화를 단일 GEMM과 효율적으로 융합하는 여러 전략을 설명하고, 마지막으로 LayerNorm과 RMSNorm을 모두 어텐션에 융합하는 FlashNormAttention 알고리즘을 소개합니다.
> To address this, we must design our normalization kernels in an IO-aware way, carefully saving memory IO costs without compromising computation accuracy via kernel fusion, and also overlapping these memory-/CUDACore-heavy ops with TensorCore-intensive ops. Since most normalization operations follow or precede matmul operations (e.g. MLP, Attention), our work focuses on how to efficiently fuse norms with matmuls. We start by describing multiple strategies in efficiently fusing norms with single GEMMs, and in the end present the FlashNormAttention algorithm, fusing both a LayerNorm and an RMSNorm into attention.

**참고**: 아래 벤치마크 결과에서는 별도 언급이 없는 한 [원소별 아핀 변환(elementwise affine)](https://docs.pytorch.org/docs/2.12/generated/torch.nn.LayerNorm.html)을 비활성화했습니다. 우리 모델에서는 이 연산이 모델 품질에 미치는 효과는 미미한 반면 상당한 성능 오버헤드를 유발하는 것으로 확인되었기 때문입니다. 또한 별도 언급이 없는 한, 여기서 제시하는 핵심 최적화와 알고리즘 아이디어는 원소별 아핀 변환의 유무와 무관하게 적용할 수 있습니다. 다만 성능 결과는 달라질 수 있습니다.
> **Note**: In the following benchmark results, we disable [elementwise affines](https://docs.pytorch.org/docs/2.12/generated/torch.nn.LayerNorm.html) unless otherwise specified, as we find those to incur significant performance overhead with marginal model quality effects in our models. Also, unless otherwise specified, the core optimization and algorithmic ideas presented are applicable regardless of whether elementwise affines exist, although the performance results may differ.

## 1. 정규화 융합의 과제 / Challenges of Normalization Fusion

**개요:** 이 절에서는 정규화를 GEMM 같은 연산 집약적 커널과 일반적인 방식으로 융합할 때의 과제를 살펴봅니다. 이 과제는 근본적으로 서로 다른 타일링(tiling) 전략에서 비롯됩니다. 이어서 GEMM 알고리즘이 정규화 알고리즘과 같은 타일링을 따르도록 강제하는 "*나이브(naive)한*" 융합 해법을 제시하고, 이 방식이 아주 작은 N에서는 잘 동작하지만 N이 커질수록 타일링 제약과 비효율 때문에 준최적이 되거나 아예 실행조차 불가능해지는 것을 관찰합니다.
> **Overview:** In this section we discuss the challenges of fusing normalization with compute-intensive kernels like GEMMs in the typical way, which fundamentally stem from different tiling strategies. We then present a “naive” fusion solution that forces the GEMM algorithm to obey the same tiling as the normalization algorithm, and observe that it performs well for super small N, but becomes suboptimal or even infeasible as N increases due to tiling constraints and inefficiencies.

표준적인 활성화 함수 융합(예: GEMM+ReLU)과 비교할 때, 정규화 융합의 근본적인 어려움은 타일링의 차이에 있습니다. 정규화는 본질적으로 리덕션(reduction) 연산이라, 올바른 결과를 계산하려면 한 차원 전체의 데이터에 접근해야 합니다. 특히 LayerNorm과 RMSNorm의 경우 일반적인 커널은 입력을 바깥쪽 차원으로는 타일링하지만 안쪽 차원으로는 하지 않으므로, 각 CTA는 항상 데이터의 행 전체를 로드해야 합니다. 반면 일반적인 GEMM은 두 차원 모두 타일링하므로 각 타일이 행 전체에 걸치지 않아, 뒤따르는 행 단위 정규화가 불가능합니다.
> Compared with standard activation fusion (e.g. GEMM+ReLU), the fundamental challenge with normalization fusion is the difference in tiling. Normalization is by nature a reduction operation that requires access to data along an entire dimension to compute the correct result. In particular, for LayerNorm and RMSNorm, a typical kernel tiles the input along the outer dimension(s), but not the inner, meaning each CTA always needs to load entire rows of data. By comparison, a typical GEMM is tiled in both dimensions, meaning each tile does not span an entire row, making a following row-wise normalization impossible.

![행렬 곱셈 타일링과 정규화 타일링의 비교 / Comparison of matmul tiling and norm tiling](/assets/blog/2026-07-10-towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/1.png){:style="width:100%"}

가장 단순한 우회책은 GEMM의 타일 크기를 늘려 각 타일이 안쪽 차원 전체에 걸치게 하는 것입니다. 일반적인 (M×K) @ (K×N) GEMM이라면 N 차원의 타일 크기가 N보다 커야 한다는 뜻입니다(보통 N보다 큰 다음 2의 거듭제곱). 상위 수준의 알고리즘은 아래 다이어그램과 같습니다:
> The most straightforward workaround for this would be to stretch the tile size of the GEMM so that each tile spans the entire inner dimension. For a typical (MxK) @ (KxN) GEMM, this means that the tile size along the N dimension must be larger than N (usually the next power of 2 from N). The high-level algorithm is illustrated in the following diagram:

![GEMM 타일을 안쪽 차원 전체에 걸치도록 늘린 나이브한 융합 알고리즘 / Naive fusion algorithm that stretches GEMM tiles to span the entire inner dimension](/assets/blog/2026-07-10-towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/2-1-scaled.png){:style="width:100%"}

이 접근에는 두 가지 주요 문제가 있습니다:

- 순수 GEMM이었다면 최적이었을 타일링 전략에서 벗어나므로, 캐시 동작과 파이프라이닝(pipelining) 동작 등이 나빠져 GEMM 자체의 성능이 떨어집니다.
- 입력 형태(shape), 구체적으로는 N이 얼마나 커질 수 있는지에 강한 제한을 둡니다. N이 너무 크면 공유 메모리(shared memory)에 들어가지 못합니다.

> - It deviates from the would-be optimal tiling strategy for a pure GEMM, which would degrade the performance of the GEMM itself due to suboptimal cache behavior, pipelining behavior, etc.
> - It places a hard limit on the input shapes; specifically, on how large N could be. Too large an N would not be able to fit into shared memory.

N이 얼마나 커질 수 있는지 감을 잡기 위한 간단한 어림 계산을 해 봅시다. 공유 메모리 크기가 228KB인 Blackwell GPU, bfloat16 데이터 타입(dtype), 효율적인 파이프라이닝/중첩을 위한 최소 2단계의 파이프라이닝 스테이지를 가정합니다. 여기에 M 차원과 K 차원의 최소 타일 크기를 각각 32로 가정하면 다음과 같습니다:
> Here’s some napkin math to get a sense of how large N can be. Assume a Blackwell GPU with 228KB shared memory size, a dtype of bfloat16, and a minimum number of 2 pipelining stages for efficient pipelining/overlap. Further assume a minimum tile size of 32 for M dimension and 32 for K dimensions. Then we have:

```
2 stages x 2 bytes / element x (tile_m x tile_k + tile_k x tile_n + tile_m x tile_n) < 228KB
=> 32 x 32 + 32 x tile_n + 32 x tile_n < 228KB / 4
=> 512 < tile_n < 1024
```

타일 크기는 보통 2의 거듭제곱이어야 하므로, 이 커널이 실행이라도 되려면 tile\_n, 따라서 N은 최대 512로 제한됩니다.
> Since tile size should usually be a power of 2, this restricts tile\_n, and therefore N, to being at most 512 for this kernel to even be able to run.

이러한 제약을 우회하는 방법은 다음 절들에서 다룹니다. 하지만 제약에도 불구하고, 이런 융합 전략이 작은 N 값에서는 여전히 상당한 이득을 낼 수 있음을 확인했습니다. 이 실험에는 [Helion](https://pytorch.kr/blog/2025/helion/)을 사용했는데, 개발 효율이 높을 뿐 아니라, 이번처럼 타일 크기 하나가 강하게 제약된 비정형적 사례에서 도움이 되는 철저한 자동 튜닝을 제공하기 때문입니다.
> We will discuss ways to work around these limitations in the following sections. But despite them, we’ve found that such fusion strategies can still produce significant gains for small values of N. For this experiment, we used [Helion](https://pytorch.org/blog/helion/) because of its high developer efficiency and exhaustive autotuning which helps in uncanonical cases like this where one of the tile sizes is hard-constrained.

아래는 광고 모델에서 관찰되는 일반적인 입력 형태에 대한 벤치마크 결과입니다. 지연 시간 절감은 torch inductor의 정규화 커널 지연 시간 대비 백분율로 계산했습니다. 이렇게 하면 이 지표가 기반 GEMM 커널의 지연 시간(그리고 그것이 정규화 커널 지연 시간과 어떻게 비교되는지)과 무관해지고, 이런 유형의 융합 시도가 가진 여지(headroom)를 본질적으로 담아내게 됩니다(즉 100%가 우리가 할 수 있는 최선이며, 정규화를 GEMM과 완전히 겹쳐야 달성할 수 있습니다).
> Below are the benchmark results on the typical input shapes observed in ads models. Note that the latency saving is calculated as the percentage of the torch inductor’s normalization kernel’s latency. We do this so that the metric becomes independent of the latency of the base GEMM kernel (and how it compares to that of the normalization kernel), and inherently captures the headroom of fusion attempts of this sort (i.e. 100% is the best we could do and would require completely overlapping the normalization with the GEMM).

![나이브한 융합의 입력 형태별 지연 시간 절감 벤치마크 / Latency saving benchmark of the naive fusion across input shapes](/assets/blog/2026-07-10-towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/3.png){:style="width:100%"}

64와 128 같은 작은 형태에서는 이 융합 전략이 LayerNorm 커널 지연 시간을 17~32%나 절감할 수 있습니다. 하지만 K/N이 128을 넘어 커지면 이득이 사라지기 시작하고 심지어 큰 성능 저하로 돌변합니다. N이 커질수록 tile\_n = N이라는 강제가 융합하지 않은 GEMM 커널의 최적 타일 크기에서 점점 더 크게 벗어나게 되고, 메모리 IO를 아끼는 이득이 기반 GEMM 알고리즘을 왜곡하는 손해에 점차 가려지기 때문입니다.
> For small shapes like 64 and 128, this fusion strategy can yield a significant 17%-32% latency saving for the LayerNorm kernel. However, as K/N grows larger beyond 128, the gain starts disappearing and even turns into huge regression. This is because as N grows, the enforcement that tile\_n = N becomes a larger deviation from the would-be optimal tile size for an unfused GEMM kernel, and the benefit of saving memory IO gradually becomes overshadowed by the harm of distorting the base GEMM algorithm.

## 2. Lazy Pre-Norm: 사전 정규화를 선형 계층과 융합하는 새로운 기법 / Lazy Pre-Norm: A Novel Technique of Fusing Pre-Norm with Linear Layers

**개요:** 이 절에서는 사전 정규화(pre-norm)인 RMSNorm을 GEMM 커널에 융합하는 새로운 프롤로그 융합(prologue fusion) 기법을 소개합니다. 이러한 프롤로그 융합의 동기와 과제를 논의하고, 수학적 트릭으로 사전 정규화 계산의 일부를 GEMM 완료 이후로 전략적으로 미뤄 이 과제를 해결하며 좋은 성능 향상을 내는 **Lazy Pre-Norm** 이라는 새로운 알고리즘을 소개합니다.
> **Overview:** In this section we introduce a novel prologue fusion technique for fusing pre-RMSNorm into a GEMM kernel. We discuss the motivations as well as challenges of such prologue fusion, and present a novel algorithm named **Lazy Pre-Norm** that tackles these challenges via strategically delaying part of the pre-norm computation until after completing the GEMM using a mathematical trick, and yields good performance speedup.

첫 번째로 소개할 아이디어는 사전 정규화를 뒤따르는 GEMM과 프롤로그 융합으로 묶어 앞서 언급한 문제를 우회합니다. 프롤로그 융합은 일반적으로 피해야 하지만, 그럼에도 탐구할 가치가 있는 이유가 몇 가지 있습니다:

1. 프롤로그 융합은 에필로그 융합(epilogue fusion)에서 마주치는 타일링 문제, 즉 GEMM 커널의 각 CTA가 출력 텐서의 행 전체에 접근할 수 없다는 문제를 비켜갑니다. 반대로 각 CTA는 알고리즘 특성상 입력 텐서 A의 행 전체를 훑고 지나갑니다!
2. 현실적으로, 특히 대규모 언어 모델에서는 사전 정규화가 사후 정규화(post-norm)보다 훨씬 더 널리 쓰이게 되었습니다.

> The first idea we present gets around the aforementioned issue by fusing pre-norms with following GEMMs, as a prologue fusion. Although prologue fusion should generally be avoided, there are a few reasons it’s still worth exploring:
> 1. Prologue fusion bypasses the tiling issue encountered in epilogue fusion, where each CTA in a GEMM kernel doesn’t have access to entire rows of the output tensor. In contrast, each CTA does scan through entire rows of the input tensor A just by the algorithm!
> 2. Realistically, pre-norm has become much more prevalent in post-norm, especially in Large Language Models.

프롤로그 융합이 잘 동작하도록, 사전 정규화의 특수한 경우인 '원소별 아핀 변환이 없는 RMSNorm'을 위해 **Lazy Pre-Norm** 이라는 최적화 기법을 고안했습니다. 구체적으로 융합하려는 대상은 다음과 같습니다:
> For prologue fusion to work well, we devised an optimization technique called **Lazy Pre-Norm** for a special case of pre-norm: RMSNorm without elementwise affines. Specifically we are looking to fuse:

`C = rmsnorm(A) @ B`

`where rmsnorm(A) = A * rstd(A)[:, None]`
`and rstd(A) = rsqrt((A ** 2).sum(dim=-1) / A.shape[-1] + 1e-5)`

Lazy Pre-Norm이 해결하려는 이 융합의 핵심 난점은 이렇습니다. 일반적인 타일 GEMM에서는 결국에는 행 전체에 접근하게 되므로(덕분에 리덕션 결과인 rstd를 계산할 수 있습니다) 문제가 없어 보이지만, 실제로는 데이터를 타일 단위로 하나씩 접근하는데, 각 타일을 처리하려면 사실 rstd가 먼저 필요합니다! 이것이 순환 의존성을 만듭니다. rstd를 계산하려면 k-루프가 끝날 때까지 기다려야 하는데, 루프를 시작이라도 하려면 rstd가 필요한 것입니다!
> Here’s the key difficulty of this fusion that Lazy Pre-Norm aims to resolve: in a typical tiled GEMM, although we eventually have access to entire rows (which is what allows us to compute the reduction result rstd), we access them tile by tile, and we actually need rstd in order to process each tile! This creates a cyclic dependency: we need to wait until the end of the k-loop to be able to compute rstd, but we need rstd to even start working on the loop!

이를 해결하기 위한 첫 번째 핵심 관찰은, 상호 의존하는 두 부분이 본질적으로 서로 다른 유형의 계산이라는 점입니다. 하나는 리덕션이고 다른 하나는 원소별(elementwise) 적용입니다.

1. rstd 계산 부분은 행 전체에 대한 **리덕션** 입니다.
2. rstd를 사용해 정규화를 적용하는 것은 A의 개별 원소 각각에 대한 **원소별** 계산입니다.

> To resolve this, the first key observation here is that the two inter-dependent parts are in essence different types of computation: reduction and elementwise application.
> 1. The rstd computation part is a **reduction** over the entire rows.
> 2. Using rstd to apply normalization is an **elementwise** computation on each individual element of A

이 두 구성 요소를 따로 공략해 봅시다. 리덕션 부분에서 먼저 주목할 점은, 이 계산 자체는 아무것도 막고 있지 않다는 것입니다. 텐서 코어 계산과 병렬로 계산할 수 있다는 뜻이니 반가운 성질입니다. 각 CTA는 자연스럽게 A의 안쪽 차원을 따라 훑고 지나가므로, 행렬 곱셈과 나란히, 그리고 병렬로 A의 제곱합을 그냥 누산해 두면 됩니다.
> Let’s tackle these two components separately. For the reduction part, the first thing to notice is that it’s not blocking anything itself, which is a nice property because it means we can compute it in parallel to the TensorCore computation. Since each CTA naturally scans along the inner dimension of A, we can just accumulate the square sum of A alongside, and in parallel to, the matmuls.

원소별 부분은 더 까다롭습니다. 리덕션 결과에 의존하기 때문에 순환 의존성을 일으킵니다. 여기서 구원이 되어 주는 것이 수학적 트릭입니다. 핵심 관찰은, 아핀 변환이 없는 RMSNorm의 원소별 곱셈이 실제로는 **행 단위 곱셈(row-wise multiplication)** 이라는 점입니다. 즉 A의 같은 행에 있는 모든 원소가 같은 rstd와 곱해집니다. 여기서 다음의 핵심 성질이 나옵니다:
> The elementwise part is more problematic as it depends on the result of the reduction, which causes cyclical dependency. The rescue here is a mathematical trick based on the key observation that the elementwise multiplication in an affine-free RMSNorm is actually **row-wise multiplication**, where all elements in the same row of A are multiplied by the same rstd. This implies the following key property:

`(A * rstd[:, None]) @ B = (A @ B) * rstd[:, None]`

`Proof:   Row-wise multiplication is equivalent to M @ A where for some diagonal matrix M. So we have (A * rstd) @ B = (M @ A) @ B = M @ (A @ B) = (A @ B) * rstd`

이 성질은 아주 반갑습니다. 원소별 계산을 "*게으르게(lazily)*" 계산할 수 있다는, 즉 k-루프 전체가 끝난 뒤로 미룰 수 있고 사실상 에필로그가 된다는 뜻이기 때문입니다! 이를 모두 종합한 Lazy Pre-Norm 알고리즘의 커널 의사코드는 다음과 같습니다:
> This is great because it means the elementwise computation can be “lazily computed” and delayed until after the whole k-loop is done, and effectively becomes an epilogue! Putting it all together, below is the kernel pseudocode for the Lazy Pre-Norm algorithm:

```python
def GEMM_norm_fusion_kernel(A, B, C):
	compute the m_tile and n_tile of this CTA
	square_sum = zeros(m_tile)
	acc = zeros(m_tile, n_tile)
	for each k_tile:
		tile_A = A[m_tile][k_tile]
		tile_B = B[k_tile][n_tile]
		acc += tile_A @ tile_B
		square_sum += (tile_A * tile_A).sum(-1) # computed in parallel to the GEMM!
	rstd = rsqrt(square_sum / A.shape[-1] + 1e-5)
	acc *= rstd[:, None]
	C[m_tile][n_tile] = acc
```

각 k 반복(iteration)마다 약간의 추가 계산이 여전히 발생하지만, 행렬 곱셈 계산과 겹칠 수 있습니다. 워프 특화(warp specialization)를 적용하면 커널의 워프 분할과 실행은 다음과 같은 모습이 됩니다:
> Note that while some additional computation is still incurred in each k iteration, it can be overlapped with the matmul computation. With warp specialization, the kernel’s warp partitioning and execution would look like:

![Lazy Pre-Norm 커널의 워프 분할과 실행 / Warp partitioning and execution of the Lazy Pre-Norm kernel](/assets/blog/2026-07-10-towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/4.png){:style="width:100%"}

이 알고리즘에는 프롤로그 융합의 핵심 단점이 여전히 남아 있습니다. RMSNorm 계산이 여러 CTA에 걸쳐 중복 수행된다는 점입니다(출력 텐서의 같은 행, 다른 열을 계산하는 모든 CTA를 떠올려 보세요. 3절에서 더 다룹니다). 하지만 Lazy Pre-Norm은 이 계산 대부분이 텐서 코어와 완전히 겹치도록 보장하므로, 중복은 감수할 만하며 여전히 좋은 성능 이득을 냅니다.
> Notice that this algorithm still features a key disadvantage of prologue fusion: the RMSNorm computation is redundantly done across many CTAs (think about all CTAs computing the same rows but different columns of the output tensor; more on this in Section 3). However, since Lazy Pre-Norm makes sure that most of this computation is fully overlapped with TensorCore, the redundancy is acceptable and still yields good performance gains.

![Lazy Pre-Norm 융합의 벤치마크 결과 / Benchmark results of the Lazy Pre-Norm fusion](/assets/blog/2026-07-10-towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/5.png){:style="width:100%"}

Lazy Pre-Norm 알고리즘에서 유의할 몇 가지 한계는 다음과 같습니다:

1. 원소별 아핀 변환을 쉽게 지원할 수 없습니다. 아핀 변환은 열 단위 곱셈으로 동작하므로, 원소별 연산이 행 단위 곱셈이어야 한다는 전제 조건을 깨뜨립니다.
2. LayerNorm에는 적용되지 않습니다. LayerNorm의 원소별 부분은 뺄셈을 포함하므로 단순한 행 단위 곱셈이 아니기 때문입니다.
3. 이 융합의 역전파(backward) 구현은 까다로울 수 있습니다. 순전파(forward)에서 rmsnorm(A)를 어디에도 구체화(materialize)하지 않으므로, dA와 dB를 계산할 때 A와 rstd로부터 rmsnorm(A)를 즉석에서 재구성해야 합니다.

> A few limitations to note about the Lazy Pre-Norm algorithm:
> 1. It cannot easily support elementwise affines, which work as a column-wise multiplication. It would break our precondition that the elementwise operation must be a row-wise multiplication.
> 2. It does not work with LayerNorm, because the elementwise part of LayerNorm involves subtraction and is not a simple row-wise multiplication.
> 3. The backward implementation for this fusion would be tricky, because we never materialize rmsnorm(A) anywhere in forward. As such, we’d need to reconstruct rmsnorm(A) from A and rstd on the fly in computing both dA and dB.

## 3. Multi-CTA Norm: 사후 정규화를 선형 계층과 에필로그로 융합하기 / Multi-CTA Norm: Fusing Post-Norm with Linears as Epilogue

**개요:** 좋은 속도 향상에도 불구하고 Lazy Pre-Norm 프롤로그 융합에는 여전히 한계가 있으며, 대부분의 정규화 사용 사례로 일반화할 수 없습니다. 이 절에서는 사후 정규화를 GEMM과 융합하는 더 일반적인 기법을 논의합니다. 다시 에필로그 융합의 영역으로 돌아와, **CTA 클러스터(CTA cluster)** 와 **분산 공유 메모리(Distributed Shared Memory)** 를 사용해 1절에서 제시한 타일링 불일치 문제를 정면으로 공략합니다.
> **Overview:** Despite the good speedup, the Lazy Pre-Norm prologue fusion still has its limitations and cannot be generalized to most norm use cases. In this section we discuss a more general technique for fusing post-norms with GEMMs, and come back to the realm of epilogue fusion, directly tackling the tiling mismatch issue presented in Section 1, using **CTA clusters** and **Distributed Shared Memory**.

[Quack](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md)에서 아이디어를 빌려와, 독립 실행형 정규화 커널을 넘어 융합 커널로 확장합니다. Quack의 정규화 커널은 [CTA 클러스터](https://docs.nvidia.com/cuda/parallel-thread-execution/#cluster-of-cooperative-thread-arrays)를 활용해 큰 N을 같은 클러스터 안의 여러 CTA에 분할하고, 필요한 데이터를 **분산 공유 메모리** 로 서로 주고받으면서 N 전체에 걸친 단일 리덕션을 협력해 수행하게 합니다. 이렇게 하면 여러 CTA가 같은 행의 데이터를 나눠 맡아 협력해 처리하면서, 전역 메모리 IO 비용 없이 정규화에 필요한 만큼 서로 통신할 수 있습니다.
> We borrow an idea from [Quack](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md) and extend it beyond standalone norm kernels to the fused kernels. The Quack norm kernels leverage [CTA clusters](https://docs.nvidia.com/cuda/parallel-thread-execution/#cluster-of-cooperative-thread-arrays) to partition large N among different CTAs in the same cluster, and let them collaborate on a single reduction across N by communicating necessary data with each other via **distributed shared memory**. This allows us to have multiple CTAs collaboratively divide and work on the same rows of data, and communicate with each other as needed by normalization, without incurring the cost of global memory IO.

![CTA 클러스터와 분산 공유 메모리 구조 / Grid with CTA clusters and distributed shared memory](/assets/blog/2026-07-10-towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/6.png){:style="width:100%"}

앞서 언급했듯 대부분의 정규화 연산은 리덕션 부분(예: RMSNorm의 rstd, LayerNorm의 평균과 분산)과, 그 리덕션 결과를 활용하는 원소별 부분으로 분해할 수 있습니다. N 차원 전체를 훑어야 하는 것은 리덕션 부분뿐이므로, 이를 CTA 클러스터 안에서 분할 정복합니다. 리덕션 결과는 (말 그대로 리덕션이니) 대개 작기 때문에, 이를 다른 CTA와 주고받는 데 필요한 DSMEM 통신 오버헤드는 아주 작습니다.
> As mentioned above, most normalization ops can be decomposed into a reduction part (e.g. rstd for RMSNorm, mean and variance for LayerNorm), and a following elementwise part that utilizes the reduction result. Only the reduction part requires scanning through the entire N dimension, which we divide and conquer within a CTA cluster. Because the reduction result is usually small (because, well, it’s a reduction), a quite minimal DSMEM communication overhead is needed to send/receive it to/from other CTAs.

![단일 CTA 정규화와 multi-CTA 정규화의 비교 / Comparison of single-CTA and multi-CTA normalization](/assets/blog/2026-07-10-towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/7.png){:style="width:100%"}

이 아이디어는 정규화 융합에서 우리가 마주친 것과 정확히 같은 문제, 곧 N이 너무 크다는 문제를 공략합니다! (N이 너무 커지는 이유와 임계값은 다르지만요.) 즉 이 multi-CTA 알고리즘을 그대로 가져와 GEMM의 에필로그에 넣기만 하면 융합이 완성됩니다!
> Note that this idea tackles the exact same problem that we are facing with norm fusion – simply that N is too large! (although the reason and threshold for N being too large differ). This means that we can simply take this multi-CTA algorithm and put it in the epilogue of our GEMM, and the fusion is done!

```python
def GEMM_norm_fusion_kernel(A, B, C):
	compute the m_tile and n_tile of this CTA
	acc = zeros(m_tile, n_tile)
	for each k_tile:
		tile_A = A[m_tile][k_tile]
		tile_B = B[k_tile][n_tile]
		acc += tile_A @ tile_B
	acc = multi_cta_norm(acc) # where DSMEM communication happens
	C[m_tile][n_tile] = acc
```

이 융합은 결코 공짜가 아니며, DSMEM 오버헤드를 들이는 것 외에도 커널에 몇 가지 제약을 가합니다. 이는 정규화를 융합해 넣는 기반 GEMM 커널의 성능 저하로 이어질 수 있습니다:

1. CTA 스케줄링에 강한 제약을 둡니다. 특히 클러스터 안의 인접 CTA들은 같은 m\_tile을 공유하되 서로 다른 n\_tile을 맡아야 합니다.
2. #1 때문에 [paired-CTA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-cta-pair) 행렬 곱셈이 어렵습니다.
3. 마찬가지로 #1 때문에 [타일 슈퍼 그룹핑(tile super-grouping)](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations) \[5\] 을 할 수 없습니다.
4. 그래도 N을 무한정 키울 수 있게 되는 것은 아닙니다. 여전히 단일 CTA 한계(약 512)에 최대 클러스터 크기를 곱한 값에 묶입니다. Blackwell에서 이식성이 보장되는 최대 클러스터 크기는 [8](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#thread-block-clusters)이므로, N은 최대 4096으로 제한됩니다.

> Note that this fusion is by no means free and places a few constraints on the kernel other than introducing DSMEM overhead, which could potentially cause regression on the base GEMM kernel we are fusing norm into:
> 1. It puts strong restrictions on CTA scheduling. In particular, adjacent CTAs in a cluster must share the same m\_tile but different n\_tile’s.
> 2. Because of #1, [paired-CTA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-cta-pair) matmul is difficult
> 3. Similarly, because of #1, [tile super-grouping](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations) \[5\] cannot be done
> 4. This still doesn’t unlock N from being indefinitely large. We are still bounded by the single-CTA limit (around 512) multiplied by the max cluster size. On Blackwell, the portable max cluster size is [8](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#thread-block-clusters). This limits N to be at most 4096.

그럼에도 상당한 메모리 IO를 아끼는 이득이 이 제약들을 훨씬 능가합니다. 이 커널에는 [TLX](https://pytorch.org/blog/enabling-cluster-launch-control-with-tlx/)를 선택했는데, 유연성/개발 효율과 저수준 하드웨어 제어 사이의 균형이 좋기 때문입니다. 이 사례 연구에서는 둘 다 매우 중요합니다. 워프 특화가 적용된 [TLX GEMM 커널](https://github.com/facebookexperimental/triton/blob/main/third_party/tlx/tutorials/blackwell_gemm_ws.py) 위에 융합 커널을 구축했습니다. 광고 모델링에서 흔한 형태(M = 256k, K = O(512), N = O(512))를 대상으로 벤치마크하여 다음과 같은 성능 결과를 얻었습니다.
> Nonetheless, the benefit of saving significant memory IO still far outweighs the limitations. We chose [TLX](https://pytorch.org/blog/enabling-cluster-launch-control-with-tlx/) for this kernel which strikes a good balance between flexibility / dev efficiency and lower-level hardware control, both of which are critical in this case study. We built the fused kernel on top of the [TLX GEMM kernel](https://github.com/facebookexperimental/triton/blob/main/third_party/tlx/tutorials/blackwell_gemm_ws.py) with warp specialization. We benchmarked it against some common shapes in ads modelling (M = 256k, K = O(512), N = O(512)), and achieved the following performance result.

![Multi-CTA 정규화 융합의 순전파 벤치마크 결과 / Forward benchmark results of the multi-CTA norm fusion](/assets/blog/2026-07-10-towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/8.png){:style="width:100%"}

K와 N은 2048에서 상한을 두었습니다. 4096에 이르면 지연 시간이 GEMM에 완전히 지배되어 정규화가 전체 지연 시간의 5% 미만을 차지하기 때문입니다.
> Note that we capped K and N at 2048, because as they reached 4096, the latency became completely dominated by the GEMM and the normalization takes less than 5% of the total latency.

### 역전파는 어떻게 할까? 융합 재편성 / What about backward? Fusion regrouping.

**개요:** 이 소절에서는 같은 융합 아이디어를 역전파에 구현할 때의 추가 과제를 논의합니다. 바로 **순전파의 에필로그 융합이 역전파에서는 자연스럽게 프롤로그 융합이 된다** 는 점입니다. 프롤로그 융합의 핵심 문제를 논의하고, 순전파와 역전파에서 정규화를 서로 다른 GEMM과 융합해 **양쪽 모두 효율적인 에필로그 융합이 되게 하는** 새로운 우회책을 제시합니다.
> **Overview:** In this subsection, we discuss the additional challenge with implementing the same fusion idea for backward: **forward epilogue fusion naturally becomes prologue fusion in backward**. We discuss the key issues with prologue fusion, and present a novel workaround solution which fuses norm with different GEMMs in forward v.s. in backward, **resulting in efficient epilogue fusion in both**.

LayerNorm과 RMSNorm의 역전파에도 리덕션이 포함되지만, 순전파와 비슷한 방식으로 해결할 수 있으므로 큰 문제는 아닙니다. 효율적인 역전파 계산에는 순전파에서 저장해 둔 중간 리덕션 결과도 필요하지만, 리덕션 결과는 1차원이라 IO가 극히 적으므로 이 역시 큰 문제가 아닙니다. 진짜 문제는 순전파의 에필로그 융합이 역전파에서는 프롤로그 융합이 된다는 점입니다.
> The backward of LayerNorm and RMSNorm also involves reduction, which wouldn’t be a big issue as we can resolve it in a similar fashion as in forward. Efficient backward computation would also need the intermediate reduction result stored from forward, which also wouldn’t be a big issue since the reduction result is 1-dimensional and results in minimal IO. The real issue is that epilogue fusion in forward becomes prologue fusion in backward.

```python
# forward formula
C = norm(A @ B)

# backward formula
dA = norm_backward(dC) @ B.T
dB = A.T @ norm_backward(dC)
```

역전파에서는 norm\_backward 계산이 GEMM보다 먼저 일어나, 융합을 시도하면 프롤로그 융합이 된다는 점에 주목하세요. 프롤로그 융합의 일반적인 단점은 2절에서 논의했지만, 이 특정 사례에서 왜 문제가 되는지 더 이해하기 위해 가능한 프롤로그 융합 해법 하나를 살펴보겠습니다.
> Notice how in backward, the norm\_backward computation happens before the GEMM, making a potential fusion prologue fusion. We’ve discussed some general drawbacks of prologue fusion in Section 2, but in this specific case, let’s look at a potential prologue fusion solution to understand more why it’s an issue.

```python
def GEMM_norm_bwd_fusion_kernel(dC, BT, dA):
	compute the m_tile and n_tile of this CTA
	acc = zeros(m_tile, n_tile)
	for each k_tile:
		tile_dC = dC[m_tile][k_tile]
		tile_dC = multi_cta_norm_bwd(tile_dC) # where DSMEM communication happens
		tile_BT = BT[k_tile][n_tile]
		acc += tile_dC @ tile_BT
	dA[m_tile][n_tile] = acc
```

이 접근에는 몇 가지 성능 문제가 있습니다:

1. 정규화 역전파 계산이 임계 경로(critical path)에 놓여 매 반복마다 GEMM 계산을 막습니다! 메인 루프가 끝난 뒤 계산을 한 번만 수행하는 에필로그 융합과 비교해 보세요.
2. 정규화 역전파에 중복 계산이 발생합니다. dC의 각 행은, 같은 m\_tile을 공유하되 서로 다른 n\_tile을 맡아 dA의 서로 다른 타일을 계산하는 여러 CTA가 각각 따로 로드한다는 점을 기억하세요. 이 CTA들 각각이 같은 dC 타일에 대해 같은 정규화 역전파 계산을 수행해야 합니다.
3. 이 융합 커널은 dA만 계산하지만 dB도 계산해야 하는데, 거기서 norm\_backward(dC)를 다시 계산하게 되어 중복 계산이 더 늘어납니다.

> There are several performance issues with this approach:
> 1. The norm bwd computation is on the critical path blocking the GEMM computation for every iteration! Compare this to epilogue fusion where a single computation is done at the end of the main loop.
> 2. Redundant computation is being done for the norm backward. Remember that each row of dC is loaded separately by different CTAs that compute different tiles of dA that share the same m\_tile but different n\_tile’s. Each of those CTAs would need to do the same norm backward computation on the same dC tiles.
> 3. This fusion kernel only computes dA, but we’d also need to compute dB as well, where we’d compute norm\_backward(dC) again, leading to more redundant computation.

그렇다면 해법은 무엇일까요? 2절에서 논의한 것 같은 특수한 경우가 아니라면 딱히 할 수 있는 것이 없습니다. 그러니 프롤로그 융합을 그냥 피하면 됩니다. 그러려면 융합 전략을 조금 더 유연하게 가져가야 합니다. 정규화 계층은 보통 선형 계층들 사이에 놓이므로, 순전파와 역전파에서 정규화 연산을 서로 다른 선형 계층과 융합하면 어떨까요?
> What’s the solution here? Well, there isn’t much we can do unless it’s some specialized cases like the one discussed in Section 2. So just avoid prologue fusion. To do that we have to be a bit more flexible in our fusion strategy: since normalization layers usually stand between linear layers, what if we fuse the normalization op with different linears in forward v.s. in backward?

![순전파와 역전파에서 정규화를 서로 다른 선형 계층과 융합하는 융합 재편성 / Fusion regrouping that fuses the norm with different linears in forward vs. backward](/assets/blog/2026-07-10-towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/9.png){:style="width:100%"}

이제 역전파도 에필로그 융합이 되고, 정규화 역전파의 중복 계산도 더는 없다는 것을 쉽게 알 수 있습니다. 더 중요하게는, 융합이 순전파 융합과 구조적으로 동일해집니다! 순전파 커널에서 `multi_cta_norm` 을 `multi_cta_norm_bwd` 로 바꾸기만 하면 역전파 커널이 완성됩니다. 아래는 순전파와 같은 형태에 대한 벤치마크 결과입니다.
> It’s easy to see that now backward is also epilogue fusion, and that there’s no longer any redundant computation for the norm backward. Also more importantly, the fusion becomes structurally identical to the forward fusion! Just replace `multi_cta_norm` with `multi_cta_norm_bwd` in the forward kernel, and you’ve got your backward kernel. Below are the benchmark results on the same shapes as forward.

![융합 재편성을 적용한 역전파 벤치마크 결과 / Backward benchmark results with fusion regrouping](/assets/blog/2026-07-10-towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/10.png){:style="width:100%"}

이 아이디어가 동작하는 데 선형 계층이 꼭 엄밀한 의미의 선형 계층일 필요는 없습니다. 예를 들어 LLM 아키텍처에서는 어텐션 -> 정규화 -> 선형 계층 패턴이나 그 반대를 볼 수 있습니다. 정규화를 사이에 둔 것이 융합해 넣기에 이득이 되는(그리고 실현 가능한) 연산 집약적 연산이기만 하면, 이런 경우에도 같은 최적화 기법을 적용할 수 있습니다.
> Note that for this idea to work, the linears do not have to be linears exactly. For example, in LLM architectures, we might see a pattern like attention -> norm -> linear, or vice versa. These could still adopt the same optimization technique, as long as what sandwiches the norm is compute-intensive ops which are beneficial (and feasible) to fuse norms into.

다음 절에서는 정규화를 어텐션에 융합하는 예를 다룹니다.
> The next section discusses an example of fusing norms into attention.

## 4. FlashNormAttention: 사전·사후 정규화를 모두 FlashAttention 계열 커널에 융합하기 / FlashNormAttention: Fusing both Pre-Norm and Post-Norm into FlashAttention-variant kernels

**개요:** 이 절에서는 앞서 소개한 융합 아이디어들을 [GDPA 커널](https://pytorch.org/blog/generalized-dot-product-attention-tackling-real-world-challenges-in-gpu-training-kernels/) \[1\] 에 적용하는 사례 연구를 살펴보고 **FlashNormAttention** 알고리즘을 소개합니다. GDPA 커널은 Meta 광고 모델, 특히 [Kunlun](https://arxiv.org/abs/2602.10016) \[2\] 아키텍처에서 많이 쓰이며, [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute) \[6\] 을 재설계한 일반화된 어텐션 커널입니다. 따라서 아래에서 논의하는 최적화 아이디어 대부분은 FlashAttention 같은 다른 어텐션 커널에도 일반화할 수 있습니다. 이 알고리즘은 위의 multi-CTA GEMM+정규화 융합과 정확히 같은 알고리즘 트릭을 사용하지만, 복잡도는 한 단계 더 높습니다(GEMM이 아닌 어텐션, 정규화 1개가 아닌 2개 융합). 성능을 끌어올리기 위해 다음을 포함한 여러 최적화 기법이 적용되었으며 아래에서 논의합니다:

- 메모리 압력을 줄이는 SMEM/TMEM 재사용
- 레지스터 압력을 줄이는 레지스터 서브타일링(register subtiling)
- 무거운 CUDA 코어 연산을 병렬화하는 정밀하게 조율된 워프 특화
- 순전파에서 추가 텐서를 저장하는 IO 비용을 피하는, 역전파에서의 정규화 재계산
- 더 나은 파이프라인 효율을 위한 [TMA\_REDUCE\_ADD](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp#L1278) 와 텐서 코어 누산(TensorCore Accumulate) 같은 고급 하드웨어 기능 활용

> **Overview:** In this section we look at a case study of applying the aforementioned fusion ideas to the [GDPA kernel](https://pytorch.org/blog/generalized-dot-product-attention-tackling-real-world-challenges-in-gpu-training-kernels/) \[1\], and present the **FlashNormAttention** algorithm. The GDPA kernel is heavily used in Meta ads models and specifically the [Kunlun](https://arxiv.org/abs/2602.10016) \[2\] architecture, and is a generalized attention kernel redesigned from [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute) \[6\]. As such, most of the optimization ideas discussed below are generalizable to other attention kernels like FlashAttention. The algorithm uses the exact same algorithmic trick as the multi-CTA GEMM+norm fusion from above, but is at a higher level of complexity (attention v.s. GEMM, fusing two norms v.s. one). Many optimization techniques are adopted to make it performant and discussed below, including:
> - SMEM / TMEM reuse to reduce memory pressure
> - Register subtiling to reduce register pressure
> - Fine-tuned warp specialization to parallelize heavy CUDACore operations
> - Norm recomputation in backward to avoid IO costs of saving additional tensors in forward
> - Using advanced hardware features like [TMA\_REDUCE\_ADD](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp#L1278) and TensorCore Accumulate for better pipeline efficiency

Kunlun \[2\] 의 일반적인 PFFN 블록은 GDPA 커널을 뼈대로 사용하되, 그 주위를 몇 개의 정규화와 잔차 연결(residual connection)로 둘러쌉니다. 아래 다이어그램은 PFFN 블록 안에서 데이터가 어떻게 흐르는지 보여줍니다.
> The typical PFFN block in Kunlun \[2\] uses a GDPA kernel as the backbone, but also surrounds it with a couple of normalization and residual connections. The diagram below shows how data flows inside a PFFN block.

![PFFN 블록 내부의 데이터 흐름 / Data flow inside a PFFN block](/assets/blog/2026-07-10-towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/11-scaled.png){:style="width:100%"}

리덕션과 원소별 커널이 GDPA 커널 앞뒤로 흩어져 있어 IO 부담이 얼마나 큰지 주목하세요. 우리의 목표는 이 모든 연산을 하나의 커널로 융합하는 것이고, 이를 **FlashNormAttention** 이라고 부릅니다. 이는 한 모듈의 모든 연산을 수행한다는 점에서 "*메가커널(megakernel)*"과 비슷하지만, 의도와 의미에서 원래의 [메가커널](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) \[7\] 과는 다릅니다. 단지 커널 실행(launch) 비용을 아끼는 것이 아니라, 대개 더 큰 병목인 **HBM을 오가는 총 데이터 전송량을 줄이는 것** 이 목표이기 때문입니다.
> Notice how IO-heavy this is with reduction and elementwise kernels scattered both before and after the GDPA kernel. Our goal is to fuse all of these operations into one single kernel, that we call the **FlashNormAttention**. This is like a “megakernel” performing all operations in a module, but it differs, in intention and meaning, from the original [megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) \[7\] in that it aims not just to save kernel launch costs, but to **save the total amount of data transfer to/from HBM**, which is usually more of a bottleneck.

전체 융합 계획은 동일합니다. CTA 클러스터와 분산 공유 메모리를 활용해 정규화를 협력해 계산하는 것입니다. 여기서 GDPA는 멀티 헤드(multi-head)이고 정규화는 모든 헤드에 걸쳐 수행된다는 점에 유의하세요. 일반적인 GDPA/FA 커널에서는 단일 CTA가 헤드 차원 전체에 접근할 수 있지만, 정규화를 위해서는 여전히 다른 헤드의 데이터가 필요하므로 multi-CTA 정규화 알고리즘이 꼭 필요합니다.
> The overall fusion plan here stays the same: leveraging CTA clusters and distributed shared memory to collaboratively compute normalization. Note that the GDPA here is multi-head, and the norms are done across all heads. So even though in the typical GDPA/FA kernel, a single CTA has access to the full head dimension, it still needs data from the other heads for normalization, necessitating the multi-CTA norm algorithm.

먼저 원래의 GDPA 커널 알고리즘을 아래 의사코드로 설명합니다. 단순화를 위해 융합과 관련된 뼈대만 담았습니다. 상세한 최적화 알고리즘은 원래의 [GDPA 블로그 글](https://pytorch.org/blog/generalized-dot-product-attention-tackling-real-world-challenges-in-gpu-training-kernels/)을 참고하세요.
> Let’s start with the original GDPA kernel algorithm which we describe in pseudocode below. For simplicity only the bare bones relevant to fusion are included. For the detailed, optimized algorithm, refer to the original [GDPA blog](https://pytorch.org/blog/generalized-dot-product-attention-tackling-real-world-challenges-in-gpu-training-kernels/).

```python
# input: Q: [batch_size, seq_len_q, H, head_dim], K/V: [batch_size, seq_len_kv, H, head_dim]
# metaparam: BLOCK_M (tile size of m_tile on seq_len_q), BLOCK_N (tile size of n_tile on seq_len_kv)
# grid on (batch_size, seq_len_q // BLOCK_M, H)
def gdpa_fwd_kernel(Q, K, V, output):
	compute the batch_idx, m_tile and head_idx for this CTA
	q = Q[batch_idx, m_tile, head_idx, :] # [BLOCK_M, head_dim], B and H dimensions are indexed 
	acc = zeros(BLOCK_M, head_dim)
	for n_tile over entire seq_len_kv:
		k = K[batch_idx, n_tile, head_idx, :] # [BLOCK_N, head_dim], B and H dimensions are indexed 
		v = V[batch_idx, n_tile, head_idx, :] # [BLOCK_N, head_dim]
		p = elementwise_activation(q @ k.T) # [BLOCK_M, BLOCK_N]
		acc += p @ v # [BLOCK_M, head_dim]
	output[batch_idx, m_tile, head_idx, :] = acc
```

융합 작업에 들어가기 전에, 이 알고리즘에 대한 핵심 조정 하나부터 시작하겠습니다. Kunlun \[2\] 의 사용 사례에서는 seq\_len\_q는 대개 크고(O(1k)) seq\_len\_kv는 대개 작다는(O(128)) 것을 관찰했습니다. 이 때문에 안쪽 루프의 파이프라이닝이 매우 얕아져 프롤로그와 에필로그 오버헤드가 드러납니다. 이 경우의 성능을 개선하기 위해 커널에서 Q와 K/V의 역할을 맞바꿔, KV로 그리드(grid)를 잡고 Q를 순회합니다. 이 알고리즘은 seq\_len\_kv를 타일링하지 않을 때에만 수치적으로 정확하다는 점에 유의하세요.
> Even before getting into fusion work, let’s start with a key tweak on this algorithm. In Kunlun’s \[2\] use case, we observed that seq\_len\_q is usually large (O(1k)) while seq\_len\_kv is usually small (O(128)). This makes the inner loop’s pipelining very shallow and exposes prologue and epilogue overhead. To improve the performance for this case, we swap the role of Q and K/V in the kernel, gridding on KV and looping over Q. Notice that this algorithm is only numerically correct when we do not tile on seq\_len\_kv.

```python
# input: Q: [batch_size, seq_len_q, H, head_dim], K/V: [batch_size, seq_len_kv, H, head_dim]
# metaparam: BLOCK_M (tile size of m_tile on seq_len_q)
# grid on (batch_size, H)
def gdpa_fwd_kernel_short_kv(Q, K, V, output):
	compute the batch_idx and head_idx for this CTA
	k = K[batch_idx, :, head_idx, :] # [seq_len_kv, head_dim], B and H dimensions are indexed 
	v = V[batch_idx, :, head_idx, :] # [seq_len_kv, head_dim]
	for m_tile over entire seq_len_q:
		q = Q[batch_idx, m_tile, head_idx, :] # [BLOCK_M, head_dim], B and H dimensions are indexed
		p = elementwise_activation(q @ k.T) # [BLOCK_M, BLOCK_N]
		output[batch_idx, m_tile, head_idx, :] = p @ v # [BLOCK_M, head_dim]
```

이 최적화는 주제와 직접 관련은 없지만, 융합 커널을 이 버전 위에 구축했고 벤치마크도 이 버전을 기준으로 수행했기 때문에 여기에 함께 싣습니다. 원래 GDPA 블로그 글에서 언급되지 않았던 내용이라 완전성을 위해 포함합니다.
> Although this optimization is not directly related to our topic, we still include it here because we built our fusion kernel on top of this version, and benchmarks were done against this version. We include it for completeness as it was not mentioned in the original GDPA blog.

이제 정규화와 잔차를 커널에 융합합니다. 아이디어는 위의 multi-CTA 리덕션과 같습니다. 한 가지 유의할 점은, 여기서는 같은 클러스터의 CTA들이 같은 batch\_idx를 공유하고 서로 다른 head\_idx를 처리해야 한다는 것입니다. 또한 여기서 LayerNorm은 프롤로그 융합이라는 점에 주목하세요. 파이프라이닝과 이에 대한 행렬 곱셈의 의존성은 여전히 문제지만(아래에서 다룹니다), 다행히 K/V가 짧은 덕분에 LayerNorm 중복 계산 문제는 없습니다. K/V의 길이를 타일링하지 않으므로 Q의 각 타일은 단 하나의 CTA만 로드하고 처리합니다.
> Now we fuse the norms and residuals into the kernel. The idea is the same as above with multi-CTA reduction. The only caveat is that here CTAs in the same cluster should share the same batch\_idx and process different head\_idx. Also, notice that the layernorm here is a prologue fusion. While the pipelining and the matmuls’’ dependency on it are still an issue (which we address below), fortunately we don’t have the problem of redundant layernorm computation thanks to K/V being short. Since we don’t tile over K/V’s length, every tile of Q will only be loaded and processed by one single CTA.

```python
# input: Q: [batch_size, seq_len_q, H, head_dim], K/V: [batch_size, seq_len_kv, H, head_dim]
# metaparam: BLOCK_M (tile size of m_tile on seq_len_q)
# grid on (batch_size, H)
def gdpa_fwd_fusion_kernel_short_kv(Q, K, V, output):
	compute the batch_idx and head_idx for this CTA
	k = K[batch_idx, :, head_idx, :]
	v = V[batch_idx, :, head_idx, :]
	for m_tile over entire seq_len_q:
		q = Q[batch_idx, m_tile, head_idx, :]
		ln_q = multi_cta_layernorm(q) # multi-CTA norm 1
		p = elementwise_activation(ln_q @ k.T)
		gdpa_out = p @ v
		gdpa_out += ln_q # residual connection 1 
		out = multi_cta_rmsnorm(gdpa_out) # multi-CTA norm 2
		out += q # residual connection 2
		output[batch_idx, m_tile, head_idx, :] = out
```

여기까지는 모두 좋아 보이지만, 이 의사코드 뒤에는 두 가지 치명적인 문제가 숨어 있습니다:

1. **메모리 압력:** 이 대규모 융합은 레지스터와 공유 메모리 사용량에 큰 압력을 가합니다. ln\_q와 rmsnorm(gdpa\_out)처럼 이전에는 존재하지 않던 것들을 훨씬 더 많이 유지해야 합니다. 실행이 순수하게 순차적이라면 연산의 출력을 만드는 즉시 그 입력을 해제할 수 있으니 괜찮겠지만, 잔차 연결 때문에 여기서는 그렇지 않습니다. q와 ln\_q의 수명(lifetime)이 넓은 구간에 걸쳐 있는 것에 주목하세요. 나중의 잔차 연결 계산을 위해 붙들고 있어야 하기 때문입니다. 즉 이 변수들을 위한 공유 메모리를 따로 확보해야만 해서 필요한 총 메모리가 늘어납니다. 실제로 이 나이브한 알고리즘 버전에서는 공유 메모리 사용량이 두 배가 되어 한계를 크게 초과하는 것을 관찰했습니다.
2. **CUDA 코어 지배와 파이프라인 지연:** 융합으로 메모리 IO 대부분을 제거했더라도, 정규화와 잔차 연결을 위한 CUDA 코어 계산은 여전히 남아 텐서 코어 활용을 막습니다. CUDA 코어 지연 시간을 최대한 숨기려면 정밀하게 조율된 워프 특화와 파이프라이닝이 필요합니다.

> Although this all looks nice and great, there are two critical issues hidden behind this pseudocode:
> 1. **Memory pressure:** This massive fusion brings high pressure on registers and shared memory usage. We need to keep many more things that didn’t exist before, such as ln\_q and rmsnorm(gdpa\_out). If the execution is purely sequential, this would be fine because as we produce the output for an op, its input can immediately be freed. But this is not the case here due to residual connections. Notice how the lifetime of q and ln\_q spans across a large region because we need to keep them for later residual connection computation. This means we most definitely have to dedicate some shared memory for these variables, increasing the total memory needed. In fact, with this naive algorithm version, we observed the shared memory usage to double, significantly exceeding the limit.
> 2. **CUDA Core dominance & pipeline stalls:** Even though we eliminated most of the memory IO with the fusion, the CUDA core computation for the norms and residual connections still remains, and blocks Tensor Core utilization. Fine-tuned warp specialization and pipelining are needed to hide as much CUDA core latency as possible.

메모리 압력에는 3가지 주요 최적화 아이디어를 적용했습니다.

- **메모리 버퍼 재사용:** 메모리 사용량을 아끼는 핵심 기법은 겹치지 않는 데이터가 같은 메모리 버퍼를 공유하게 하는 것입니다. 우리 사례의 좋은 예가 out을 위한 공유 메모리 버퍼입니다. 위 코드의 마지막 줄에서는 out을 레지스터에서 HBM으로 바로 저장하는 것처럼 보이지만, 실제로는 먼저 SMEM 버퍼에 넣은 다음 TMA를 호출해 SMEM에서 HBM으로 비동기 저장을 수행합니다. 이 버퍼의 수명이 짧다는 것은 자명하므로, ln\_q를 계산한 뒤 두 번째 잔차 연결에 필요해지기 전까지 q를 임시로 보관하는 데 이 버퍼를 재사용합니다.
- **텐서 메모리와 텐서 코어의 누산 활용:** ln\_q가 행렬 곱셈 p @ v의 결과에 곧바로 더해지는 것에 주목하세요. ln\_q를 SMEM에 두었다가 행렬 곱셈이 끝난 뒤 읽어내는 대신, 이것이 정확히 tcgen05가 TMEM에서 지원하는 [MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma) 시맨틱임을 알 수 있습니다. 따라서 ln\_q를 p @ v에 할당된 TMEM 버퍼에 곧바로 두고, ln\_q 덧셈을 텐서 코어에 떠넘길 수 있습니다! 이는 SMEM 사용량과 계산 시간을 모두 아껴 줍니다.
- **레지스터 서브타일링:** 우리 커널처럼 CUDA 코어 부담이 큰 커널에서는 특히 레지스터가 SMEM/TMEM보다 더 희소한 자원입니다. 신중한 레지스터 할당 튜닝과 더불어, 레지스터 서브타일링으로 레지스터 압력을 완화합니다. SMEM/TMEM에 있는 텐서를 조각(chunk)으로 잘라, 정규화 계산(리덕션과 원소별 모두)을 위해 한 번에 한 조각씩 레지스터로 로드합니다. 이렇게 하면 심각한 성능 저하를 일으키는 레지스터 스필(spill)을 방지하는 데 도움이 됩니다.

> For memory pressure, we applied 3 main optimization ideas
> - **Memory buffer reuse:** A key technique for saving memory usage is to let non-overlapping data share the same memory buffers. A good example in our case is the shared memory buffer for out. In the last line of code above, it looks like we are directly storing out from registers to HBM, but in reality what happens is we first put it in a SMEM buffer, and then invoke TMA to asynchronously perform the store from SMEM to HBM. It’s obvious that this buffer has a short lifespan, and we reuse this buffer to temporarily store q after computing ln\_q but before needing it for the second residual connection.
> - **Leverage Tensor Memory and TensorCore’s accumulate:** Notice how ln\_q is added immediately to the result of a matmul p @ v. Instead of keeping ln\_q in SMEM and reading it out after the matmul is done, notice how this is exactly the [MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma) semantic supported by tcgen05 in TMEM. Therefore, we can directly keep ln\_q in the TMEM buffer allocated for p @ v, and offload the ln\_q addition to TensorCore! This helps save both SMEM footprint and computation time.
> - **Register subtiling:** Especially in a CUDA-core-heavy kernel like ours, registers are a scarcer resource than SMEM/TMEM. Besides careful register allocation tuning, we use register subtiling to mitigate the register pressure. We cut the tensor in SMEM/TMEM into chunks and load to registers one chunk at a time for normalization computation (both reduction and elementwise). This helps prevent register spilling which would cause significant performance degradation.

파이프라인 지연에는 다음 기법들을 사용해 파이프라인 효율을 개선했습니다:

- **워프 특화:** 원래 GDPA 설계에는 4개의 주요 특화 워프 파티션(load, mma, activation, epilogue)이 있습니다. FlashNormAttention에서는 RMSNorm 계산을 activation 워프에 두고, 프롤로그 LayerNorm 계산 전용의 다섯 번째 파티션을 추가해, 텐서 코어는 물론 다른 CUDA 코어 연산(예: 이전 반복의 RMSNorm 계산)과 더 잘 겹치게 했습니다. 실행 파이프라인은 다음과 같은 모습입니다. activation 파티션에 8개 워프(0-7)를, LayerNorm에 4개 워프(8-11)를 사용합니다. activation 워프에는 레지스터 할당을 최대화하고 LayerNorm 워프에는 제한을 두면서, 레지스터 서브타일링으로 레지스터-지연 시간 트레이드오프의 최적점을 찾습니다.
> - **Warp specialization:** In the original GDPA design, we have 4 main specialized warp partitions (load, mma, activation, and epilogue). In FlashNormAttention, we put the RMSNorm computation on the activation warp, while adding a fifth partition dedicated to prologue Layernorm computation, in order to better overlap it with TensorCore as well as other CUDA Core operations (e.g. RMSNorm computation for the previous iteration). The execution pipeline looks like the following. We use 8 warps (0-7) for the activation partition and 4 warps (8-11) for Layernorm. We maximize register allocation for the activation warps while limiting that for the Layernorm warps, using register subtiling to strike an optimal register-latency tradeoff.

![FlashNormAttention의 실행 파이프라인 / Execution pipeline of FlashNormAttention](/assets/blog/2026-07-10-towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/12-scaled.png){:style="width:100%"}

- **레지스터 프리로딩:** 실행 파이프라인을 지연시키는 핵심 요인은 잔차 연결이 만들어내는 복잡한 데이터 의존성입니다. q와 ln\_q 모두 오랫동안 메모리에 유지되어야 해서, 다음 반복을 위한 이 텐서들의 프리페치(prefetch)와 사전 계산을 막습니다. TMA를 통한 프리페치는 SMEM을 사용하므로, 위에서 논의한 메모리 압력 최적화 #2가 사실상 ln\_q를 임계 경로 밖으로 옮겨 줍니다. q의 경우, activation 워프가 q가 준비되는 즉시 SMEM에서 미리 로드해 두고, 두 번째 잔차 연결이 일어나는 맨 끝까지 붙들고 있게 합니다. 이렇게 하면 q가 차지하던 SMEM을 즉시 해제할 수 있어, 현재 반복을 처리하는 동안 다음 반복의 q를 로드할 수 있습니다. activation 워프에 최대 개수의 레지스터를 할당하므로 레지스터 압력은 여전히 괜찮습니다.
> - **Register pre-loading:** A key factor stalling the execution pipeline is the complex data dependency introduced by residual connections. Both q and ln\_q need to be held in memory for a long time, blocking the prefetching and precomputing of these tensors for the next iteration. Since prefetching with TMA utilizes SMEM, Optimization #2 for memory pressure discussed above effectively moves ln\_q off the critical path. As for q, we let the activation warps preload q from SMEM as soon as it’s ready, and hold on to it until the very end where the second residual connection happens. By doing this, we can immediately free up the SMEM for q so that the next iteration’s q can be loaded as we work on the current iteration. As we allocate a maximum number of registers to the activation warps, the register pressure is still fine.

아래는 우리가 사용하는 일반적인 GDPA 형태에 대한 벤치마크 결과입니다. K/V는 길이가 정확히 128인 밀집(dense) 시퀀스입니다. Q는 평균 희소성(sparsity) 0.5의 희소 시퀀스이며 최대 길이를 다양하게 바꿨습니다. 배치 크기는 768입니다. 헤드 차원은 128로 고정하고, 서로 다른 정규화 차원과 CTA 클러스터 크기에서의 성능 차이를 반영하기 위해 헤드 수도 다양하게 바꿨습니다. 융합이 복잡한 만큼, 지연 시간 절감을 정규화/원소별 커널만의 지연 시간이 아니라 전체 기준선(baseline) 지연 시간 대비 백분율로 제시합니다. 기준선은 inductor 컴파일로 얻었습니다.
> Below we present the benchmark results for the typical GDPA shapes we use. K/V are dense sequences with length exactly 128. Q is a sparse sequence with average sparsity 0.5 and varied max lengths. The batch size is 768. Head dimension is set to 128, and the number of heads is also varied to reflect the different performance behaviors under different normalization dimensions and different CTA cluster size. Due to the complex fusion, we present the latency saving as a percentage of the total baseline latency instead of just the normalization / elementwise kernels’ latency. The baseline is taken with inductor compilation.

![FlashNormAttention 순전파 벤치마크 결과 / Forward benchmark results of FlashNormAttention](/assets/blog/2026-07-10-towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/13.png){:style="width:100%"}

### 역전파 구현 / Backward implementation

역전파는 간결함을 위해 상세 알고리즘은 생략하고, 흥미로운 최적화와 순전파 대비 유사점/차이점만 짚어 보겠습니다.
> As for backward, we skip the detailed algorithm for brevity, and only note the interesting optimizations and similarities/differences compared to forward.

- **알고리즘:** 우선 역전파 융합은 순전파와 매우 닮았습니다. 정규화 같은 리덕션 연산의 역전파도 리덕션 연산이고, 잔차 연결의 역전파도 잔차 연결이기 때문입니다.
- **재계산:** 원래 GDPA 역전파 커널은 순전파의 메모리 IO를 아끼기 위해 q@k를 재계산합니다. 우리 융합도 같은 아이디어를 따릅니다. 첫째, 우리 커널에서는 q@k가 ln(q)@k가 되므로 먼저 ln(q)를 재계산합니다. 순전파에서 평균과 분산을 저장해 두어(1차원이라 저렴합니다) 역전파가 이를 이용해 ln(q)를 쉽게 유도할 수 있게 함으로써, 여기서 DSMEM 오버헤드가 발생하는 것을 피합니다. 둘째, RMSNorm의 역전파 계산에는 그 출력이 필요한데, 이는 rmsnorm\_out = kernel\_out – q 로 재계산합니다. 마지막으로 잔차 연결의 역전파에는 RMSNorm의 입력도 필요한데, 이는 rmsnorm\_in = rmsnorm\_out / rstd 로 재계산합니다.
- **워프 특화:** 원래 GDPA 역전파 커널에는 4개의 워프 파티션(mma, activation, load, 그리고 dQ를 원자적으로 더하는 reduction)이 있습니다. 같은 구조를 사용하되, LayerNorm 역전파를 제외한 모든 새 계산을 activation 파티션에 넣습니다. LayerNorm 역전파 계산은 더 나은 파이프라이닝을 위해 reduction 파티션에 넣습니다(순전파에서 프롤로그 LayerNorm을 별도 파티션에 둔 것과 비슷합니다).
- **메모리 압력:** 역전파는 저장하고 계산해야 할 데이터가 훨씬 많다는 이유만으로도 순전파보다 더 심한 메모리 압력을 받습니다. 완화를 위해 순전파와 비슷하게 SMEM/TMEM을 공격적으로 재사용했고, 타일 크기도 줄여야 했습니다.
- **파이프라인 효율:** 순전파와 마찬가지로 잔차의 긴 수명이 파이프라이닝을 막습니다. 안쪽 잔차에는 여전히 TMEM과 텐서 코어 누산을 사용해 수명을 끊을 수 있습니다. 하지만 바깥쪽 잔차는 역전파의 더 높은 압력 때문에 레지스터 프리로딩을 감당할 수 없습니다. 대신 [TMA\_REDUCE\_ADD](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp#L1278) 를 사용해 잔차가 준비되는 즉시 SMEM에서 HBM으로 곧바로 더하는 방식을 택했습니다.
  - FlashNormAttention의 핵심이 메모리 IO를 줄이는 것이므로 처음에는 직관에 어긋나 보일 수 있지만, 실제로는 합리적인 선택입니다. 융합 덕분에 메모리 IO는 더 이상 이 커널의 병목이 아니므로, 현재의 병목인 계산과 파이프라인 지연을 완화하기 위해 메모리 IO를 조금 더 지불하는 쪽으로 기꺼이 되돌아갈 수 있습니다.

> - **Algorithm:** First notice that the backward fusion very much resembles forward, because the backward of a reduction op like normalization is also a reduction op, and the backward of a residual connection is also a residual connection
> - **Recompute:** The original GDPA backward kernel recomputes q@k to save memory IO for forward. We follow the same idea for our fusion. First, the q@k becomes ln(q)@k in our kernel, so we recompute ln(q) first. We avoid incurring DSMEM overhead here by storing mean and variance in the forward (which is cheap because they are 1D) so that backward can use them to derive ln(q) easily. Second, we need the output of rmsnorm for its backward computation, which we recompute by rmsnorm\_out = kernel\_out – q. Lastly, we need the input of rmsnorm as well for residual connection backward, which we recompute by rmsnorm\_in = rmsnorm\_out / rstd.
> - **Warp specialization:** The original GDPA backward kernel has 4 warp partitions: mma, activation, load, and reduction (for atomic-adding dQ). We use the same structure and put all new computation except Layernorm backward in the activation partition. We put the Layernorm backward computation in the reduction partition to facilitate better pipelining (similar to how we put the prologue Layernorm in a separate partition in forward).
> - **Memory pressure:** Backward faces an even more severe memory pressure than forward, simply due to how much more data it needs to store and compute. For mitigation we applied aggressive SMEM/TMEM reuse similar to forward, but also had to shrink the tile sizes.
> - **Pipeline efficiency:** Similar to forward, the long lifecycle of the residuals blocks the pipelining. For the inner residual, we can still use TMEM and TensorCore accumulate to break the lifecycle. But for the outer residual, we can’t afford register preloading in backward due to the higher pressure. Instead our approach is to use [TMA\_REDUCE\_ADD](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp#L1278) to directly add the residual from SMEM to HBM as soon as it’s ready.
>   - It may seem counterintuitive at first to do this because the whole point of FlashNormAttention is to reduce memory IO, but this actually makes good sense. With the fusion the memory IO is no longer the bottleneck of the kernel, so we are happy to go back and trade a little more memory IO for mitigating the current bottleneck – the computation and pipeline stalls.

아래 차트는 역전파 융합의 성능 개선을 보여줍니다. 순전파와 정확히 같은 형태를 사용했습니다.
> The chart below shows the performance improvement of the backward fusion. The exact same shapes as in forward are used.

![FlashNormAttention 역전파 벤치마크 결과 / Backward benchmark results of FlashNormAttention](/assets/blog/2026-07-10-towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/14.png){:style="width:100%"}

## 감사의 글 / Acknowledgements

[Flash Attention](https://github.com/Dao-AILab/flash-attention)과 [Quack](https://github.com/Dao-AILab/quack)의 오픈소스 작업으로 이번 글에서 논의한 많은 최적화 기법의 기반을 놓고 영감을 준 Tri Dao, Markus Hoehnerbach, Jay Shah, Ted Zadouri, Vijay Thakkar, Wentao Guo에게 감사드립니다. 또한 여기서 제시한 아이디어들을 효율적이고 결실 있게 탐구할 수 있게 해 준 [Helion](https://github.com/pytorch/helion)과 [TLX](https://github.com/facebookexperimental/triton/tree/main)를 개발하고 유지보수하는 PyTorch 팀과 Triton 팀에도 감사드립니다.
> We thank Tri Dao, Markus Hoehnerbach, Jay Shah, Ted Zadouri, Vijay Thakkar, Wentao Guo for their open-source work in [Flash Attention](https://github.com/Dao-AILab/flash-attention) and [Quack](https://github.com/Dao-AILab/quack), which laid the foundation and provided inspiration for many optimization techniques discussed in this blog. We thank the Pytorch and Triton teams for their development and maintenance of [Helion](https://github.com/pytorch/helion) and [TLX](https://github.com/facebookexperimental/triton/tree/main), which made the exploration of the presented ideas efficient and fruitful.

## 참고 문헌 / References

\[1\] Generalized Dot-Product Attention: Tackling Real-World Challenges in GPU Training Kernels. [https://pytorch.org/blog/generalized-dot-product-attention-tackling-real-world-challenges-in-gpu-training-kernels/](https://pytorch.org/blog/generalized-dot-product-attention-tackling-real-world-challenges-in-gpu-training-kernels/)

\[2\] Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems through Unified Architecture Design. [https://arxiv.org/abs/2602.10016](https://arxiv.org/abs/2602.10016)

\[3\] Meta’s Generative Ads Model (GEM): The Central Brain Accelerating Ads Recommendation AI Innovation. [https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/)

\[4\] Quack: Getting Memory-bound Kernels to Speed-of-Light. [https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md)

\[5\] Triton Tutorials: 03 Matrix Multiplication. [https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)

\[6\] FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. [https://arxiv.org/pdf/2407.08608](https://arxiv.org/pdf/2407.08608)

\[7\] Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B. [https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
