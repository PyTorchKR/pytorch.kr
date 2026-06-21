---
layout: blog_detail
title: "DeepSpeed에서 Muon 옵티마이저 사용하기"
author: Zhipeng Wang, Guokai Ma, Peng Du and Chi McIsaac, DeepSpeed team
category: ["pytorch.org", "translation"]
org_title: "Using Muon Optimizer with DeepSpeed"
org_link: https://pytorch.org/blog/using-muon-optimizer-with-deepspeed/
---

## TL;DR

이제 DeepSpeed가 Muon 옵티마이저를 지원합니다! Muon 옵티마이저는 선도적인 AI 연구소들에서 상당한 채택이 이뤄지며 큰 탄력을 받고 있습니다. 그러한 AI 연구소 중 하나가 Moonshot AI로, Kimi-K2-Thinking과 같은 대규모 파운데이션 모델을 학습시키기 위해 Muon 옵티마이저를 채택했습니다. 이번 글에서는 Muon 옵티마이저가 무엇인지, 그리고 DeepSpeed에서 어떤 성능을 보이는지 자세히 살펴봅니다.
> DeepSpeed now supports Muon Optimizer! Muon Optimizer has gained great momentum with significant adoption from frontier AI Labs. One of those AI Labs is Moonshot AI, which has adopted Muon Optimizer to train its Large Foundation Model like Kimi-K2-Thinking. This post dives into what Muon Optimizer is and how it performs on DeepSpeed.

## Muon 옵티마이저란? / What is Muon Optimizer?

Muon은 신경망의 은닉 2D 가중치(hidden 2D weights)를 위해 설계된 옵티마이저입니다. 가중치의 변화도(gradient)를 받아 그 모멘텀을 계산하고, 모멘텀 행렬을 직교화(orthogonalize)하기 위해 Newton-Schulz 반복을 적용한 뒤, 이렇게 직교화된 행렬을 사용해 [가중치를 갱신](https://kellerjordan.github.io/posts/muon/)합니다. Muon은 (Adam의 두 개와 달리) 모멘텀 버퍼를 하나만 유지하기 때문에, 옵티마이저 상태에 더 적은 메모리를 사용합니다.
> Muon is an optimizer designed for hidden 2D weights of a neural network. It takes gradient of the weight, computes its momentum, and applies Newton-Schulz iterations to orthogonalize the momentum matrix, then uses this orthogonalized matrix to update [the weight](https://kellerjordan.github.io/posts/muon/). Because Muon only maintains one momentum buffer (versus Adam’s two), it uses less memory for optimizer states.

직교화 단계는 사전 학습(pretraining)에서 Muon이 갖는 수렴상의 이점에 핵심적인 역할을 합니다. 실제로 트랜스포머의 2D 가중치에 대한 변화도 갱신은 매우 높은 조건수(condition number)를 갖는 경향이 있습니다. 즉, 거의 낮은 랭크(low-rank)에 가까우며 몇 개의 큰 특이 방향(singular direction)에 의해 지배됩니다. 모멘텀 행렬을 직교화함으로써 Muon은 모든 특이값(singular value)을 균등하게 만들고, 그렇지 않았다면 가려졌을 드물지만 중요한 갱신 방향을 효과적으로 증폭시킵니다. 이는 더 나은 샘플 효율성(sample efficiency)으로 이어집니다. [NanoGPT 스피드러닝 벤치마크](https://github.com/KellerJordan/modded-nanogpt)에서 Muon은 AdamW 대비 학습 속도를 35% 향상시켰으며, 1.5B 매개변수 규모에서는 GPT-2 XL 수준의 성능에 [AdamW보다 약 25% 더 빠르게](https://kellerjordan.github.io/posts/muon/) 도달했습니다.
> The orthogonalization step is key to Muon’s convergence advantage in pretraining. In practice, gradient updates for 2D weights in transformers tend to have very high condition numbers — they are nearly low-rank, dominated by a few large singular directions. By orthogonalizing the momentum matrix, Muon equalizes all singular values, effectively amplifying rare but important update directions that would otherwise be overshadowed. This leads to better sample efficiency: in [NanoGPT speedrunning benchmarks](https://github.com/KellerJordan/modded-nanogpt), Muon improved training speed by 35% over AdamW, and at 1.5B parameter scale it reached GPT-2 XL level performance approximately [25% faster than AdamW](https://kellerjordan.github.io/posts/muon/).

각 매개변수마다 두 개의 모멘텀 버퍼를 필요로 하는 Adam 옵티마이저와 달리, Muon 옵티마이저는 모멘텀 버퍼를 하나만 필요로 합니다. 즉, Muon 옵티마이저를 사용하는 매개변수의 경우 모멘텀을 위한 버퍼를 하나만 할당하면 되므로 Adam에 비해 메모리를 절약할 수 있습니다.
> Unlike Adam optimizer that requires two momentum buffers for each parameter, Muon Optimizer only requires one momentum buffer. This means that for parameters using Muon Optimizer, we only need to allocate one buffer for momentum, which can save memory compared to Adam.

Muon은 Keller Jordan의 [NanoGPT](https://github.com/KellerJordan/modded-nanogpt) 변형판, Andrej Karpathy의 [nanochat](https://github.com/karpathy/nanochat)에서 사용되며, Muon의 변형(MuonClip)은 MoonShot의 프로덕션 수준 [LLM Kimi-K2](https://arxiv.org/pdf/2507.20534)에서도 사용됩니다. 보다 최근에는 Zhipu AI의 GLM-5(744B 매개변수)가 GLM-4.5와 GLM-5 사전 학습 모두에서 Muon 옵티마이저를 사용했음을 확인했으며, 여기에는 MLA 업프로젝션 행렬을 어텐션 헤드별로 분할하고 각 헤드를 독립적으로 직교화하는 "Muon Split" 기법이 함께 사용되어 [Muon](https://arxiv.org/abs/2602.15763) 사용 시 MLA와 GQA 사이의 성능 격차를 해소했습니다. DeepSeek-V4(1.6T 매개변수) 또한 [더 빠른 수렴과 더 큰 학습 안정성](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)을 위해 Muon 옵티마이저를 사용합니다.
> Muon is used by Keller Jordan’s mod of [NanoGPT](https://github.com/KellerJordan/modded-nanogpt), Andrej Karpathy’s [nanochat](https://github.com/karpathy/nanochat), and a variant of Muon (MuonClip) is also used by the production-level [LLM Kimi-K2 from MoonShot](https://arxiv.org/pdf/2507.20534). More recently, Zhipu AI’s GLM-5 (744B parameters) confirmed the use of Muon Optimizer in both GLM-4.5 and GLM-5 pretraining, along with a “Muon Split” technique that splits MLA up-projection matrices by attention head and orthogonalizes each head independently, addressing a performance gap between MLA and GQA when using [Muon](https://arxiv.org/abs/2602.15763) DeepSeek-V4 (1.6T parameters) also employs the Muon Optimizer for [faster convergence and greater training stability](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro).

## DeepSpeed의 Muon 옵티마이저 지원 / Muon Optimizer support in DeepSpeed

Muon 옵티마이저를 DeepSpeed에 적용할 때의 과제 중 하나는, 기존 옵티마이저(SGD, Adam)가 변화도를 평탄화된(flattened) 버퍼로 다룬다는 점입니다. 따라서 변화도 버퍼가 이미 평탄화되어 있어 같은 자리에 Muon 옵티마이저를 끼워 넣기가 어렵습니다. 이에 Muon 갱신을 stage 1 및 stage 2 `DeepSpeedZeroOptimizer` 의 `get_flat_partition` 함수로 옮겼는데, 이 단계에서는 매개변수별 변화도가 아직 평탄화되지 않은 상태이므로 Muon 갱신을 손쉽게 적용할 수 있습니다.
> One of the challenges of applying Muon optimizer to DeepSpeed is that previous optimizers (SGD, Adam) look at gradients as flattened buffers. Thus it is hard to swap in Muon Optimizer in the same place because the gradient buffers are already flattened. We move the Muon update to `get_flat_partition` function of stage 1 and 2 `DeepSpeedZeroOptimizer` in which per parameter gradients are still in unflattened stages, thus we can easily apply the Muon updates.

Muon 옵티마이저는 2D 가중치 행렬(어텐션 및 MLP 가중치)에서 동작합니다. 모멘텀 행렬에 Newton-Schulz 직교화를 적용하는데, 이를 위해서는 가중치가 2D여야 합니다. 2D가 아닌 매개변수(임베딩, 레이어 정규화, 편향, lm\_head)는 AdamW로 대체(fall back)됩니다. 모델 엔진 초기화 과정에서 파싱을 수행하여, 모델 매개변수가 2D이면서 은닉 계층에 속하는 경우에 한해 `use_muon` 으로 태그합니다. Muon 옵티마이저를 사용할 때, `use_muon` 으로 태그된 모든 매개변수는 Muon 옵티마이저로 가중치를 갱신합니다.
> Muon Optimizer works on 2D weight matrices (attention and MLP weights). It applies Newton-Schulz orthogonalization to the momentum matrix, which requires the weight to be 2D. Non-2D parameters (embeddings, layer norms, biases, lm\_head) fall back to AdamW. We apply a parse in model engine initializer to tag the model parameter with `use_muon`, if and only if the model parameter is 2D and belongs to hidden layers. When Muon Optimizer is used, any parameter tagged `use_muon` will use Muon Optimizer to update weight.

참고로 Muon은 하이브리드 옵티마이저입니다. 즉, 2D 은닉 가중치에만 Muon 갱신을 사용하고 그 외의 모든 매개변수(임베딩, 레이어 정규화, 편향, lm\_head)에는 Adam으로 대체됩니다. DeepSpeed 설정은 `muon_lr`(Muon 매개변수용)과 `adam_lr`(Adam 매개변수용)을 통해 별도의 학습률을 지원합니다.
> Note that Muon is a hybrid optimizer: it uses Muon updates only for 2D hidden weights and falls back to Adam for all other parameters (embeddings, layer norms, biases, lm\_head). The DeepSpeed config supports separate learning rates via `muon_lr`(for Muon parameters) and `adam_lr` (for Adam parameters).

## Muon 옵티마이저로 DeepSpeed 미세 조정 실행하기 / Running DeepSpeed finetune with Muon Optimizer

[DeepSpeed finetune 데모](https://github.com/delock/deepspeed_finetune_demo)는 다양한 DeepSpeed 학습 기능을 한곳에서 사용하고 그 성능을 비교해 볼 수 있는 데모입니다. 이를 사용해 Muon 옵티마이저로 LLM 모델을 미세 조정하는 것을 테스트할 수 있습니다.
> [Deepspeed finetune demo](https://github.com/delock/deepspeed_finetune_demo) is a demo to use different DeepSpeed training features and compare their performance in a single place. You can use it to test finetune LLM models with Muon Optimizer:

```sh
git clone https://github.com/delock/deepspeed_finetune_demo
cd deepspeed_finetune_demo
```

```sh
./finetune.sh z2_muon.json
```

## Muon 옵티마이저 수렴 실험 결과 / Muon Optimizer Convergence Experiment Result

Moonlight-16B-A3B(전체 16B, 활성 3B 매개변수를 갖는 전문가 혼합(Mixture-of-Experts) 모델)를 미세 조정하여 Muon 옵티마이저를 테스트했으며, 코드 생성(MBPP/MBPP+), 일반 지식(MMLU), 수학적 추론(GSM8K) 벤치마크에서 평가했습니다. 각 벤치마크는 자체적인 도메인 특화 학습 세트를 사용합니다.
> We tested Muon Optimizer by finetuning Moonlight-16B-A3B (a Mixture-of-Experts model with 16B total and 3B active parameters), and evaluated on code generation (MBPP/MBPP+), general knowledge (MMLU), and mathematical reasoning (GSM8K) benchmarks. Each benchmark uses its own domain-specific training set.

학습 구성:
> Training Configuration:

- 모델: Moonlight-16B-A3B (MoE, 전체 16B / 활성 3B)
- 학습 데이터셋: MBPP/MBPP+용 sahil2801/CodeAlpaca-20k, MMLU용 cais/mmlu (auxiliary\_train, 약 95k개 예시), GSM8K용 meta-math/MetaMathQA (sample\_rate=0.1, 약 39.5k개 예시)
- ZeRO Stage 2, bf16, 전문가 병렬화(Expert Parallelism, autoep\_size=4)
- 배치 크기: 16, 변화도 누적: 2, GPU 4개
- 1 에폭, 변화도 클리핑: 1.0

> - Model: Moonlight-16B-A3B (MoE, 16B total / 3B active)
> - Training datasets: sahil2801/CodeAlpaca-20k for MBPP/MBPP+, cais/mmlu (auxiliary\_train, ~95k examples) for MMLU, meta-math/MetaMathQA (sample\_rate=0.1, ~39.5k examples) for GSM8K
> - ZeRO Stage 2, bf16, Expert Parallelism (autoep\_size=4)
> - Batch size: 16, gradient accumulation: 2, 4 GPUs
> - 1 epoch, gradient clipping: 1.0

## 평가 결과 / Evaluation Results

| 옵티마이저 / Optimizer | 학습률 / Learning Rate | adam_lr (Muon용) / adam_lr (for Muon) | MBPP | MBPP+ | MMLU | GSM8K |
|---|---|---|---|---|---|---|
| baseline (미세 조정 전) / baseline (pre-finetune) | — | — | 0.495 | 0.431 | 0.401 | 0.526 |
| AdamW | 2e-6 | — | 0.661 | 0.534 | 0.660 | 0.805 |
| Muon | 1e-4 | 2e-6 | 0.646 | 0.548 | 0.678 | 0.810 |

Muon은 4개 지표 중 3개에서 AdamW를 앞섭니다. MBPP+(0.548 vs 0.534, +1.4%p), MMLU(0.678 vs 0.660, +1.8%p), GSM8K(0.810 vs 0.805, +0.5%p)입니다. MBPP 기본 테스트에서는 AdamW가 Muon을 근소하게 앞섰지만(0.661 vs 0.646, -1.5%p), 추가 테스트 케이스가 포함되어 더 엄격한 MBPP+에서는 Muon이 더 높은 점수를 얻어(0.548 vs 0.534) 더 나은 일반화 성능을 시사합니다.
> Muon outperforms AdamW on 3 out of 4 metrics: MBPP+ (0.548 vs 0.534, +1.4pp), MMLU (0.678 vs 0.660, +1.8pp), and GSM8K (0.810 vs 0.805, +0.5pp). On MBPP base tests, AdamW edges out Muon (0.661 vs 0.646, -1.5pp), though Muon achieves a higher score on the more rigorous MBPP+ with extra test cases (0.548 vs 0.534), suggesting better generalization.

## Muon 옵티마이저의 메모리 절감 / Muon Optimizer Memory Savings

Muon 옵티마이저는 매개변수당 (1차 및 2차 모멘트의) 두 개 대신 하나의 모멘텀 버퍼를 유지하기 때문에, 옵티마이저 상태에 Adam보다 더 적은 메모리를 사용합니다.
> Muon Optimizer uses less memory for optimizer states than Adam, because it maintains one momentum buffer per parameter instead of two (first and second moment).

메모리 사용량 비교
> Memory Usage Comparison

참고로 Muon은 하이브리드 옵티마이저입니다. 2D 은닉 가중치는 Muon(버퍼 1개)을 사용하고, 나머지 매개변수(임베딩, 레이어 정규화, lm\_head)는 여전히 Adam(버퍼 2개)을 사용합니다. 실제 메모리 절감량은 전체 매개변수 중 2D 은닉 가중치가 차지하는 비율에 따라 달라집니다. 일반적인 트랜스포머 모델에서는 매개변수의 약 90%가 2D 은닉 가중치이므로, 옵티마이저 상태 메모리가 약 45% 감소합니다. 다만 전체 GPU 메모리에는 모델 가중치, 변화도, 활성화도 포함되므로, 종단 간(end-to-end) 메모리 절감 폭은 이보다 작습니다(아래 측정 결과 참조).
> Note that Muon is a hybrid optimizer: 2D hidden weights use Muon (1 buffer), while remaining parameters (embeddings, layer norms, lm\_head) still use Adam (2 buffers). The actual memory savings depend on the fraction of parameters that are 2D hidden weights. For typical transformer models, approximately 90% of parameters are 2D hidden weights, so optimizer state memory is reduced by roughly 45%. However, because total GPU memory also includes model weights, gradients, and activations, the end-to-end memory reduction is smaller (see measured results below).

| 옵티마이저 / Optimizer | 매개변수당 상태 버퍼 / State Buffers per Param | 매개변수당 메모리 / Memory per Parameter |
|---|---|---|
| Adam | 2 (m, v) | 8 bytes |
| Muon | 1 (momentum) | 4 bytes |

## 측정된 GPU 메모리: Qwen2.5-3B 미세 조정 / Measured GPU Memory: Qwen2.5-3B Fine-tuning

위에서 설명한 것과 동일한 8xA100(40GB) 구성(배치 크기 32, ZeRO Stage 2, bf16)으로 tatsu-lab/alpaca에서 Qwen2.5-3B를 미세 조정하는 동안의 최대(peak) GPU 메모리를 측정했습니다.
> We measured peak GPU memory during fine-tuning Qwen2.5-3B on tatsu-lab/alpaca using the same 8xA100 (40GB) configuration described above (batch size 32, ZeRO Stage 2, bf16).

| 옵티마이저 / Optimizer | GPU당 최대 메모리 / Peak Memory per GPU | AdamW 대비 절감 / Savings vs AdamW |
|---|---|---|
| AdamW | 34.5 GiB | — |
| Muon | 31.4 GiB | 9% |

Muon은 AdamW에 비해 GPU당 메모리를 약 3 GiB(9%) 줄입니다. 이 절감은 전적으로 옵티마이저 상태에서 비롯됩니다. Muon 매개변수는 Adam의 두 개(8 bytes) 대신 하나의 모멘텀 버퍼(4 bytes)를 저장합니다. 다만 옵티마이저 상태는 (모델 가중치, 변화도, 활성화와 함께) 전체 GPU 메모리의 한 구성 요소에 불과하므로, 종단 간 절감 폭은 크지 않습니다. 더 큰 모델이나 더 빠듯한 메모리 예산에서는, 이 9%의 절감이 워크로드를 온디바이스에 올릴 수 있느냐, 아니면 CPU 오프로딩이 필요하냐를 가르는 차이가 될 수 있습니다.
> Muon reduces per-GPU memory by approximately 3 GiB (9%) compared to AdamW. The savings come entirely from optimizer states: Muon parameters store one momentum buffer (4 bytes) instead of Adam’s two (8 bytes). However, because optimizer states are only one component of total GPU memory (alongside model weights, gradients, and activations), the end-to-end reduction is modest. For larger models or tighter memory budgets, this 9% savings could make the difference between fitting a workload on-device versus requiring CPU offloading.

## 다음 단계 / What’s Next

Muon은 커뮤니티에서 빠르게 입지를 넓혀 가고 있으며, Kimi-K2(1T 매개변수)와 GLM-5(744B 매개변수)의 프로덕션 수준 채택은 Muon이 대규모 학습의 기본 옵티마이저로서 Adam을 대체할 유력한 후보임을 보여줍니다. 현재 DeepSpeed에서 완전한 Muon 지원을 적극적으로 구축하고 있으며, 일련의 개선 작업이 이미 진행 중입니다.
> Muon is rapidly gaining traction in the community, and production-level adoption by Kimi-K2 (1T parameters) and GLM-5 (744B parameters) signals that it is a serious contender to replace Adam as the default optimizer for large-scale training. We are actively building out full Muon support in DeepSpeed, with a series of improvements already in flight:

- ZeRO Stage 2 지원 — 병합 완료(merged)
- ZeRO Stage 3 지원 — 병합 완료(merged)
- Gram-Schmidt 기반 Newton-Schulz 반복 — 더 빠른 직교화 커널, 리뷰 중
- CPU 오프로딩 — 진행 중
- MuonClip — Kimi-K2가 사용하는 변형, 계획 중

> - ZeRO Stage 2 support — merged
> - ZeRO Stage 3 support — merged
> - Gram-Schmidt based Newton-Schulz iteration — a faster orthogonalization kernel, in review
> - CPU Offloading — in progress
> - MuonClip — the variant used by Kimi-K2, planned

DeepSpeed의 Muon 옵티마이저 지원과 관련된 어떤 의견, 피드백, 기여도 환영합니다. 논의를 위해 이슈를 열거나 DeepSpeed에 PR을 제출해 주세요. DeepSpeed에서 Muon을 견고하고 빠르게 만들어 봅시다!
> We welcome any thoughts, feedback and contributions related to Muon Optimizer support on DeepSpeed – please start an issue for discussion or submit a PR to DeepSpeed. Let’s make Muon rock solid and lightning fast in DeepSpeed!
