---
layout: blog_detail
title: "베이지안 최적화로 Helion의 자동 튜닝 가속하기"
author: Ethan Che, Oguz Ulgen, Max Balandat, Jongsok Choi, Jason Ansel
ext_author: Junghwan Park (박정환)
category: ["pytorch.org", "translation"]
date: 2026-02-24 12:00:00
org_title: "Accelerating Autotuning in Helion with Bayesian Optimization"
org_link: https://pytorch.org/blog/accelerating-autotuning-in-helion/
---

## 들어가며 / Introduction

[이전 블로그 글](https://pytorch.org/blog/helion/)에서 소개했듯이, Helion은 익숙한 PyTorch 스타일의 문법으로 고성능 ML 커널을 작성할 수 있게 해주는 고수준 DSL이며, 복잡한 최적화 작업은 자동 튜닝(autotuning) 엔진에 위임합니다. 이 자동 튜너(autotuner)는 블록 크기(block size), 루프 순서(loop order), 메모리 접근 패턴 등 구현 선택지로 이루어진 방대한 고차원 공간을 탐색하여 대상 하드웨어에서 성능을 극대화하는 구성(configuration)을 찾아냅니다. 그 결과 Helion은 `torch.compile`은 물론, Triton이나 CuTe DSL로 정교하게 손으로 작성한 커널보다도 상당한 속도 향상을 달성할 수 있습니다.
> As introduced in a [previous blog post](https://pytorch.org/blog/helion/), Helion is a high-level DSL that empowers developers to write high-performance ML kernels using a familiar PyTorch-like syntax, delegating the complex task of optimization to its autotuning engine. This autotuner explores a vast, high-dimensional space of implementation choices—block sizes, loop orders, memory access patterns—to discover configurations that maximize performance on the target hardware. As a result, Helion can achieve significant speedups over torch.compile and even highly-optimized, hand-written kernels in Triton or CuTe DSL.

그러나 자동 튜닝으로 얻는 성능 향상에는 대가가 따릅니다. 바로 **긴 실제 소요 시간(wall-clock time)** 입니다. 일반적인 자동 튜닝 세션은 수천 개의 후보 구성을 평가하면서 10분 이상이 걸리며, 복잡한 커널의 경우 수 시간 단위까지 늘어나기도 합니다. 출시 이후 긴 자동 튜닝 시간은 사용자 불만으로 꾸준히 제기되어 왔으며, 커널 개발 주기에서 가장 큰 고충 중 하나였습니다. Helion은 탐색 단계 수를 줄이는 등 자동 튜닝 과정을 단축할 수 있는 선택지를 제공하지만, 이는 보통 커널 성능 저하로 이어져 바람직하지 않은 절충을 강요합니다.
> However, the performance gains from auto-tuning comes with a cost: **long wall-clock times**. A typical autotuning session can take 10+ minutes, evaluating thousands of candidate configurations, and can even take on the order of hours for complex kernels. Since its launch, long autotuning times have consistently surfaced as a user complaint and one of the biggest pain points in the kernel development cycle. While Helion provides developers options to shorten the auto-tuning process, e.g. by reducing the number of search steps, this typically leads to a loss in kernel performance, forcing an undesirable trade-off.

이번 글에서는 자동 튜닝 경험을 개선하기 위한 진행 중인 노력을 다룹니다. 특히 이러한 문제를 해결하기 위해 개발한 새로운 탐색 알고리즘 [LFBO Pattern Search](https://helionlang.com/api/autotuner.html#module-helion.autotuner.surrogate_pattern_search)를 소개합니다. 이 알고리즘은 머신러닝(ML) 기법을 활용하여 자동 튜닝 엔진의 효율을 높입니다. 탐색 알고리즘이 ML 모델을 학습시켜 후보 구성을 지능적으로 걸러냄으로써 평가하는 후보의 수를 크게 줄입니다. 중요한 점은, 이 모델이 탐색 과정에서 수집된 데이터만 사용하며 사용자가 별도의 데이터를 제공할 필요가 없다는 것입니다.
> In this blog post, we discuss our ongoing efforts to improve the autotuning experience. In particular, we discuss a new search algorithm [LFBO Pattern Search](https://helionlang.com/api/autotuner.html#module-helion.autotuner.surrogate_pattern_search) we developed to address these issues, which employs techniques from machine learning (ML) to improve efficiency of the autotuning engine. The search algorithm trains an ML model to intelligently filter candidate configurations, substantially reducing the number of candidates evaluated. Importantly, the model only uses data collected during the search process, and doesn't need the user to provide any additional data.

ML을 활용하면 성능을 희생하지 않으면서도 자동 튜닝 시간을 상당히 줄일 수 있습니다.
> Using ML, we can reduce autotuning time substantially without sacrificing performance:

- 벤치마크용 NVIDIA B200 커널 모음에서, 자동 튜닝 시간을 **36.5%** 줄이는 동시에 커널 지연 시간(latency)을 평균 **2.6%** 개선했습니다.
- AMD MI350 커널에서는 자동 튜닝 시간을 **25.9%** 줄이면서 커널 지연 시간을 **1.7%** 개선했습니다.

> -   On our set of benchmark NVIDIA B200 kernels, we reduce autotuning time by **36.5%** while improving kernel latency by **2.6%** on average.
> -   On AMD MI350 kernels, we reduce autotuning time by **25.9%** while improving kernel latency by **1.7%**.

![NVIDIA B200 커널에서의 결과 / Results on NVIDIA B200 kernels](/assets/blog/2026-02-24-accelerating-autotuning-in-helion/1-3.png){:style="width:100%"}

![AMD MI350 커널에서의 결과 / Results on AMD MI350 kernels](/assets/blog/2026-02-24-accelerating-autotuning-in-helion/2-4.png){:style="width:100%"}

일부 커널에서는 개선 효과가 특히 두드러집니다. B200 layer-norm 커널에서는 실제 소요 시간이 최대 **50%** 감소했고, B200 Helion FlashAttention 커널에서는 커널 지연 시간이 **15% 이상** 개선되기도 했습니다. 이처럼 향상된 성능 덕분에, 이 알고리즘은 이 글을 쓰는 현재 기본 탐색 알고리즘입니다.
> For some kernels the improvements are especially significant: we see up to a **50% reduction** in wall-clock time for B200 layer-norm kernels, and even a **\>15% improvement** in kernel latency for B200 Helion FlashAttention kernels. Due to its enhanced performance, it is the default search algorithm at the time of writing.

## 커널 자동 튜닝의 어려움 / The Challenges of Kernel Autotuning

자동 튜닝 엔진은 커널 구성들을 탐색하면서 그 지연 시간을 벤치마크하고, 그 결과를 바탕으로 다음에 벤치마크할 구성 집합을 결정합니다. 단일 구성을 컴파일하고 지연 시간을 측정하는 데는 수 초 정도가 걸리지만, 자동 튜닝 엔진은 가능한 최고의 성능을 얻기 위해 보통 수천 개의 구성을 탐색합니다. 최적의 커널 구성을 찾는 일은 설계 공간에 내재된 여러 요인으로 인해 어려운 최적화 문제입니다.
> The autotuning engine searches through kernel configurations, benchmarking their latency and using the outcomes to determine the next set of configs to benchmark. While compiling and measuring the latency of a single configuration takes on the order of seconds, the autotuning engine typically searches through thousands of configurations to achieve the best possible performance. Finding the optimal kernel configuration is a challenging optimization problem due to several factors inherent to the design space:

- **고차원의 조합 공간(High-Dimensional, Combinatorial Space)**: 블록 크기, 언롤 팩터(unroll factor) 등 가능한 모든 조합의 공간은 고차원이며 방대합니다. LayerNorm처럼 단순한 커널조차 8천조(8 quadrillion, 10^16) 개가 넘는 구성을 가질 수 있습니다. 다만 탐색 공간은 거대하지만, 좋은 성능을 내는 구성은 극히 일부에 불과합니다.
- **긴 컴파일 시간(Long Compile Times)**: 어떤 커널 구성은 컴파일에 상당한 시간이 걸려, 자동 튜닝 과정의 실제 소요 시간을 불필요하게 늘립니다.
- **구성 오류와 타임아웃(Config Errors and Timeouts)**: 탐색 공간에는 컴파일 오류가 나거나, 부정확한 결과를 내거나, 컴파일에 너무 오래 걸리는 구성도 포함될 수 있습니다.

> -   **High-Dimensional, Combinatorial Space:** The space of all possible combinations of block sizes, unroll factors, etc. is high-dimensional and vast. Even a simple kernel like LayerNorm has more than 8 quadrillion (10^16) possible configurations. However, while the search space is large, only a small fraction of configs have good performance.
> -   **Long Compile Times:** Certain kernel configurations can take a significant amount of time to compile, unnecessarily extending the autotuning process's wall-clock time.
> -   **Config Errors and Timeouts:** The search space can also include configs that have compilation errors, produce inaccurate results, or take too long to compile.

이전의 기본 탐색 전략([Pattern Search](https://helionlang.com/api/autotuner.html#module-helion.autotuner.pattern_search))은 여러 개의 유망한 구성('탐색 사본(search copies)')에서 시작하여, 단일 매개변수를 변형한 모든 경우를 빠짐없이 평가하는 방식으로 이웃 구성들을 탐색합니다. 철저하긴 하지만 이 방식은 비효율적입니다. 이웃 구성의 대부분은 성능을 전혀 개선하지 못하는데도 하나하나 컴파일하고 벤치마크하기 때문입니다. 게다가 이동을 단일 매개변수 변경으로 제한하면, 고차원 탐색 공간을 빠르게 가로지르는 능력이 떨어집니다.
> The previous default search strategy ([Pattern Search](https://helionlang.com/api/autotuner.html#module-helion.autotuner.pattern_search)) starts from multiple promising configurations ('search copies') and explores neighboring configs by exhaustively evaluating all single-parameter perturbations. While thorough, this approach is inefficient: the vast majority of neighbors offer no performance improvement, yet each is compiled and benchmarked. Furthermore, restricting moves to single-parameter changes limits the algorithm's ability to traverse the high-dimensional search space quickly.

## 가능도 없는 베이지안 최적화 패턴 탐색 / Likelihood-Free Bayesian Optimization Pattern Search

이러한 비효율을 해결하기 위해, 다음에 평가할 점을 지능적으로 선택하는 확률적 대리 모델(surrogate model, 예: 가우시안 프로세스(Gaussian Process))을 활용하는 머신러닝의 한 분야인 [베이지안 최적화(Bayesian Optimization)](https://arxiv.org/abs/1807.02811)에서 영감을 얻었습니다([botorch](https://github.com/meta-pytorch/botorch)나 [Ax](https://ax.dev/) 같은 라이브러리에서 사용할 수 있습니다). 추가되는 실제 소요 시간을 최소화하기 위해, 더 가벼운 분류(classification) 모델을 대리 모델로 사용하는 [가능도 없는 베이지안 최적화(Likelihood-Free Bayesian Optimization)](https://arxiv.org/abs/2206.13035)(LFBO)를 도입했습니다. Pattern Search의 지역 탐색 휴리스틱과 LFBO 분류기 모델을 결합하여, 완전 탐색 대신 가장 유망한 후보만 걸러서 벤치마크합니다.
> To address these inefficiencies, we take inspiration from [Bayesian Optimization](https://arxiv.org/abs/1807.02811), a sub-domain of machine learning which utilizes a probabilistic surrogate model (e.g. a Gaussian Process) to intelligently select which points to evaluate next (available in libraries such as [botorch](https://github.com/meta-pytorch/botorch) and [Ax](https://ax.dev/)). To minimize additional wall-clock time, we adapt [Likelihood-Free Bayesian Optimization](https://arxiv.org/abs/2206.13035) (LFBO), which uses a lighter-weight classification model as a surrogate. We combine the local search heuristic of Pattern Search with the LFBO classifier model to filter only the most promising candidates to benchmark, instead of exhaustive search.

LFBOPatternSearch 알고리즘은 다음과 같습니다.
> The LFBOPatternSearch algorithm is as follows:

1. PatternSearch와 마찬가지로, 먼저 무작위로 생성한 구성 집합을 벤치마크하여 가장 유망한 소수의 구성('탐색 사본(search copies)')을 식별합니다.
2. 탐색 사본으로부터 여러 매개변수에 걸쳐 무작위 변형을 가해 후보를 생성하며, PatternSearch보다 더 넓게 탐색합니다.
3. 지금까지 수집한 지연 시간 데이터로 분류 모델(랜덤 포레스트(RandomForest))을 학습시킵니다. 지연 시간을 직접 예측하는 대신, 해당 구성이 지연 시간 기준 상위 10%에 드는지를 나타내는 이진 레이블을 예측합니다.
4. ML 모델의 예측을 바탕으로 후보의 순위를 매깁니다. 일반적인 LFBO와 달리, 탐색을 장려하기 위해 이미 순위가 매겨진 후보와의 유사도에 대한 패널티도 추가합니다.
5. 그중 상위 10%를 선택해 컴파일하고 벤치마크합니다. 성능이 가장 좋은 구성을 바탕으로 탐색 사본을 갱신하고, 측정된 지연 시간을 데이터셋에 추가합니다.

> 1.  Similar to PatternSearch, we first benchmark a set of randomly generated configs, and identify a small set of the most promising configurations ('search copies').
> 2.  We generate candidates from the search copies, by making random perturbations across multiple parameters, exploring more widely than PatternSearch.
> 3.  We train a classification model (RandomForest) on latency data collected so far. Instead of predicting latency directly, we predict a binary label indicating whether the config is in the top 10% in terms of latency.
> 4.  We rank the candidates based on ML model predictions. Unlike typical LFBO, we also add a penalty for similarity to previously ranked candidates to encourage exploration.
> 5.  We select the top 10% of them to compile and benchmark. We update the search copies based on the best performing configs and add the latencies to the dataset.

![LFBO Pattern Search 알고리즘 개요 / Overview of the LFBO Pattern Search algorithm](/assets/blog/2026-02-24-accelerating-autotuning-in-helion/3-3.png){:style="width:100%"}

관찰된 실제 소요 시간과 지연 시간 개선을 달성하는 데 결정적인 역할을 한 몇 가지 핵심 설계 결정을 살펴보겠습니다.
> We discuss some key design decisions, which are critical for achieving the improvements in wall-clock time and latency we observed:

**분류 대 회귀(Classification vs Regression)**: 모델이 지연 시간을 직접 예측하도록 학습시키는 회귀(regression) 기반 방법은 시스템/컴파일러 연구에서 비용 모델링(cost modeling)의 사실상 표준입니다. 그러나 좋든 나쁘든 모든 구성의 지연 시간을 학습하려고 애쓰는 대신, 분류 기반 접근이 가장 성능이 좋은 구성에 모델의 역량을 더 잘 집중시킨다는 점을 확인했습니다. 둘째로, 분류 손실(classification loss)은 오류가 나거나 컴파일 타임아웃이 발생하는 구성(이들에는 음의 레이블이 부여됩니다)을 피하도록 모델이 학습하게 해줍니다. 반면 이런 점들은 유효한 지연 시간 데이터가 없으므로 회귀 기반 접근이 학습할 거리가 없습니다.
> **Classification vs Regression:** Regression-based methods, i.e. training the model to predict latency directly, is the de-facto approach for cost modeling in systems / compiler research. However, we find that a classification-based approach better focuses model capacity on the most performant configs instead of trying to learn the latency of all configs, good or bad. Second, the classification loss enables the model to learn to avoid configs that error out or suffer compile timeouts (as these are assigned negative labels). However, these points do not have any valid latency data for a regression-based approach to learn from.

**다양성 장려(Encouraging Diversity)**: 보통 구성들은 병렬 사전 컴파일(pre-compilation)을 활용하기 위해 배치(batch) 단위로 컴파일됩니다. 랜덤 포레스트 분류기는 서로 뭉쳐 있는 유사한 구성을 반복적으로 선택할 수 있는데, 이는 새로운 정보를 거의 주지 못하는 중복 샘플에 배치 예산을 낭비하게 만듭니다. 이를 완화하기 위해, 랜덤 포레스트 모델의 리프 노드 동시 출현(leaf node co-occurrence)을 바탕으로 유사도 점수를 계산하고, 이미 순위가 매겨진 구성과의 유사도에 패널티를 부여합니다.
> **Encouraging Diversity:** Typically configs are compiled in batch, to take advantage of parallelized pre-compilation. The Random Forest classifier may repeatedly select similar configurations that cluster, which can waste the batch budget on redundant samples that provide little new information. To mitigate this, we compute a similarity score based on leaf node co-occurrence from the Random Forest model, and penalize similarity to previously ranked configs.

LFBO Pattern Search의 동작을 살펴보면, 커널·하드웨어 종류·형상(shape) 전반에 걸친 성능 개선이 더 적은 평가 횟수로 더 나은 런타임을 가진 구성을 찾아내는 능력에서 비롯된다는 것을 실제로 확인할 수 있습니다. 아래는 B200 layer-norm 커널에 대한 자동 튜닝 트레이스 예시로, 시간 경과에 따라 자동 튜너가 얻은 최적 구성의 지연 시간을 보여줍니다. LFBO가 자동 튜닝을 더 일찍(약 9분 대신 약 5분) 완료할 뿐만 아니라, Pattern Search에 비해 훨씬 큰 폭의 성능 도약을 이루며 더 나은 구성을 더 빠르게 찾아낸다는 것을 볼 수 있습니다.
> When we investigate the behavior of LFBO Pattern Search, we see indeed that improvements in performance across kernels, hardware types, and shapes are due its ability to find configurations with better runtime using fewer evaluations. Below is a plot of example auto-tuning traces for a B200 layer-norm kernel, displaying the latency of the best configuration obtained by the autotuner over time. We see not only that LFBO completes auto-tuning earlier (~5 min instead of ~9 min), it finds better configurations faster with much larger jumps in performance compared to Pattern Search.

![B200 layer-norm 커널의 자동 튜닝 트레이스 / Auto-tuning traces for a B200 layer-norm kernel](/assets/blog/2026-02-24-accelerating-autotuning-in-helion/4-2.png){:style="width:100%"}

LFBO가 이를 달성하는 방식은 Pattern Search보다 더 넓게 탐색하는 것입니다. 아래는 동일한 B200 layer-norm 커널에 대해 LFBO Pattern Search와 Pattern Search가 샘플링한 구성을 보여주는 그래프로, (구성이 고차원이므로) 시각화를 위해 주성분 분석(Principal Component Analysis, PCA)을 적용했습니다. LFBO Pattern Search는 구성을 절반도 안 되는 수만큼 평가하면서도, 그 샘플링한 구성들이 Pattern Search의 것보다 더 넓게 퍼져 있음을 볼 수 있습니다. Pattern Search의 구성들은 단일 매개변수 변형만 하기 때문에 한곳에 심하게 뭉쳐 있습니다. 분류기의 안내를 받아 LFBO Pattern Search는 더 크면서도 더 표적화된 도약을 할 수 있습니다.
> We see that LFBO accomplishes this by exploring more widely than Pattern Search. Below is a plot of configs sampled by LFBO Pattern Search and Pattern Search for the same B200 layer-norm kernel, where we apply Principal Component Analysis (PCA) for visualization (as configs are high-dimensional). We see that while LFBO Pattern Search evaluates less than half of the number of configs, its sampled configs are more spread out than Pattern Search's which are highly clumped together due to Pattern Search making only single parameter perturbations. Guided by the classifier, the LFBO Pattern Search is able to make larger, but more targeted jumps.

![표집된 구성의 PCA 시각화 / PCA visualization of sampled configurations](/assets/blog/2026-02-24-accelerating-autotuning-in-helion/5-2.png){:style="width:100%"}

마지막으로, 다른 대리 모델들, 특히 랜덤 포레스트, 그래디언트 부스팅 트리(Gradient-Boosting Tree), 다층 퍼셉트론(Multi-Layer Perceptron, MLP)을 사용하는 회귀 기반 접근들과 함께 애블레이션(ablation) 실험을 수행했습니다. 이때 (PatternSearch에서 수집한) 자동 튜너 로그 데이터셋을 사용했습니다. 자동 튜너 성능과 가장 직접적으로 연관된 지표인, 대리 모델을 사용해 다음 후보 배치를 걸러낼 때 기대되는 커널 지연 시간 개선치를 계산합니다. 아래에서는 대리 모델이 선택하도록 허용된 후보 비율 대비 기대 개선치(지연 시간의 상대적 % 개선)를 그래프로 나타냈습니다. LFBO 기반 방법이 가장 큰 기대 개선치를 제공하며, 다양성을 고려한 선택이 의미 있는 개선을 더한다는 것을 확인했습니다. 특히 후보의 10%만 선택할 수 있을 때, 회귀 기반 방법은 단순 무작위 선택과 동등하거나 오히려 더 나쁜 성능을 보였습니다. 회귀가 항상 순위 매기기(ranking) 성능과 일치하지는 않기 때문입니다.
> Finally, we perform an ablation with other surrogate models, in particular regression-based approaches involving a Random Forest, Gradient-Boosting Tree, and Multi-Layer Perceptron (MLP), using a dataset of autotuner logs (collected from PatternSearch). We compute a metric that is most directly correlated with autotuner performance: the expected improvement in kernel latency when using the surrogate to filter the next batch of candidates. Below we plot the expected improvement (in terms of relative % improvements in latency) compared to the percent of candidates the surrogate is allowed to select. We find that the LFBO-based methods deliver the largest expected improvement, with meaningful improvements from diverse selection. Notably, when we only can select 10% of candidates, the regression-based methods perform equivalent or even worse than simple random selection, as regression is not always aligned with ranking performance.

![대리 모델별 기대 개선치 애블레이션 / Ablation of expected improvement across surrogate models](/assets/blog/2026-02-24-accelerating-autotuning-in-helion/6-2.png){:style="width:100%"}

## 맺음말 / Conclusion

이번 글에서는 머신러닝(ML)이 어떻게 자동 튜닝 엔진을 가속하고 Helion에서의 커널 작성 경험을 개선할 수 있는지를 보여주었습니다. 탐색 과정에서 수집한 지연 시간 데이터를 활용하면, 자동 튜너를 더 유망한 구성에 집중시켜 시간을 절약하고 더 빠른 커널 구성을 발견할 수 있습니다. 우리는 강화 학습(reinforcement learning, RL)과 대규모 언어 모델(large language models, LLMs)의 기법을 포함하여, 자동 튜너를 강화하기 위한 추가적인 ML 기법을 적용하는 데 적극적인 관심을 갖고 있으며, 어떤 형태의 기여든 환영합니다.
> In this blog post, we illustrate how machine learning (ML) can accelerate the autotuning engine and improve the kernel authoring experience in Helion. By using the latency data collected during the search process, we can focus the autotuner on more promising configurations, saving time and discovering faster kernel configs. We are actively interested in applying additional ML techniques to enhance the auto-tuner, including methods from reinforcement learning (RL) and large language models (LLMs), and welcome any contributions.
