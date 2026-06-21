---
layout: blog_detail
title: "분 단위에서 초 단위로: Helion 커널을 위한 LLM 기반 자동 튜닝"
author: Jongsok Choi, Ethan Che, Jason Ansel, Oguz Ulgen
ext_author: Junghwan Park (박정환)
category: ["pytorch.org", "translation"]
date: 2026-06-18 12:00:00
org_title: "From Minutes to Seconds: LLM-Guided Autotuning for Helion Kernels"
org_link: https://pytorch.org/blog/from-minutes-to-seconds-llm-guided-autotuning-for-helion-kernels/
---

## TL;DR

PyTorch의 성능 이식 가능한(performance portable) 머신러닝 커널용 도메인 특화 언어(domain-specific language, DSL)인 Helion은 성능을 위해 자동 튜닝(autotuning)에 크게 의존합니다. 현재 Helion의 탐색은 가능도 없는 베이지안 최적화(Likelihood-Free Bayesian Optimization, LFBO)를 활용해 가장 성능이 좋은 구성(config)을 찾습니다. LFBO는 잘 동작하는 강력한 기준선(baseline)이지만, 커널마다 여전히 수백 번의 컴파일·벤치마크 주기를 거쳐야 합니다. 이를 해결하기 위해, LFBO 수준의 커널 성능(기하평균 1.009배)을 달성하면서도 약 10배 더 적은 구성을 벤치마크하고 약 6.7배 짧은 실제 소요 시간(wall-clock time)에 끝내는 LLM 기반 자동 튜너를 소개합니다. LLM이 LFBO보다 5% 이상 뒤처지는 소수의 커널에 대해서는, 하이브리드 전략(LLM으로 초기값을 잡은 뒤 LFBO로 정교화)이 그 격차를 메우면서도 전체 LFBO 탐색보다 약 3배 저렴하게 동작합니다. 마지막으로, 이 결과는 대체로 LLM 모델에 독립적입니다 — Opus-4.8, gpt-5.5, Sonnet-4.6이 서로 몇 퍼센트 이내의 성능을 보이며, 이는 LLM 기반 자동 튜닝이 프로덕션 품질을 유지하면서 커널 튜닝을 획기적으로 빠르게 만드는 실용적인 접근임을 보여줍니다.
> Helion, PyTorch's domain-specific language (DSL) for performance portable machine learning kernels, heavily relies on autotuning for performance. Currently Helion searches utilize the Likelihood-Free Bayesian Optimization (LFBO) to find the most performant configs. LFBO is a strong baseline which works well, but it still grinds through hundreds of compile-and-benchmark cycles per kernel. To this end, we introduce an LLM-guided autotuner that matches LFBO-level kernel performance (geomean 1.009X) while benchmarking ~10X fewer configurations in ~6.7X less wall-clock time. For the handful of kernels where the LLM trails by >5%, a hybrid strategy (LLM seeding followed by LFBO refinement) closes the gap while remaining ~3X cheaper than the full LFBO search. Finally, the result is largely LLM model-independent — Opus-4.8, gpt-5.5, and Sonnet-4.6 perform within a couple percent of each other — showing that LLM-guided autotuning is a practical approach to dramatically faster kernel tuning at production quality.

## 들어가며 / Introduction

자동 튜닝은 성능이 뛰어나고 이식 가능한 ML 커널을 작성하기 위한 PyTorch의 DSL인 Helion의 근간입니다. 모든 Helion 커널은 방대한 고차원 구성 공간(타일 크기, 블록 크기, num\_warps, num\_stages 등, 자세한 내용은 [문서](https://helionlang.com/api/autotuner.html) 참조)을 대상으로 튜닝되어 대상 하드웨어에서 최고 성능에 도달합니다. 튜닝 시간을 줄이는 것 또한 개발 속도와 프로덕션 배포에 중요하며, 이는 Helion 채택에 영향을 줍니다.
> Autotuning is the backbone of Helion – PyTorch's DSL for authoring performant and portable ML kernels. Every Helion kernel is tuned across a vast, high-dimensional configuration space (tile sizes, block sizes, num\_warps, num\_stages, see [documentation](https://helionlang.com/api/autotuner.html) for more) to reach peak performance on the target hardware. Reducing the tuning time is also critical for developer velocity and production deployment, which impacts Helion adoption.

자동 튜너는 조합 공간을 탐색하여 구성을 찾고, 각 구성을 벤치마크한 뒤 가장 좋은 것을 보존합니다. 이 탐색을 더 빠르고 똑똑하게 만드는 것은 활발히 진행 중인 작업입니다. Helion의 현재 기본 자동 튜너는 LFBO(가능도 없는 베이지안 최적화)를 사용하는데, 여기서는 가벼운 랜덤 포레스트(Random Forest) 분류기를 탐색 도중 벤치마크된 데이터로 즉석에서 학습시켜, 어떤 구성이 유망한 후보인지 예측하는 법을 배웁니다. 이 예측을 이용해 가장 중요한 매개변수에 집중하여 공간을 표적 지향적으로 도약하며 탐색합니다. LFBO 탐색은 NVIDIA와 AMD GPU 모두에서 커널 성능과 튜닝 시간을 모두 크게 개선했기에 현재 기본값으로 채택되어 있습니다. 자세한 내용은 PyTorch 블로그 "[베이지안 최적화로 Helion의 자동 튜닝 가속하기](https://pytorch.org/blog/accelerating-autotuning-in-helion/)"를 참조하세요.
> The autotuner searches the combinatorial space to find configurations, benchmarks each configuration, and keeps the best. Making that search faster and smarter is an active area of work. Helion's current default autotuner uses LFBO (Likelihood-Free Bayesian Optimization), where a lightweight Random Forest classifier is trained during the search on the fly on the benchmarked data, learning to predict which configurations are promising candidates. It uses the prediction to focus on the parameters that matter the most to take targeted jumps through the space. LFBO search is now the default, as it showed substantial improvements in both kernel performance and tuning time on NVIDIA and AMD GPUs. See our PyTorch blog "[Accelerating Autotuning in Helion with Bayesian Optimization](https://pytorch.org/blog/accelerating-autotuning-in-helion/)" for more details.

LFBO는 잘 동작하는 강력한 기준선이지만, 커널마다 여전히 수백 번의 컴파일·벤치마크 주기를 거쳐야 합니다. 만약 탐색을 무작정 시작하는 대신, LLM에게 커널을 추론하게 하여 구성을 제안하도록 요청할 수 있다면 어떨까요? 바로 LLM 기반 자동 튜너입니다 — 자동 튜닝의 각 라운드마다 LLM에게 커널, 워크로드, 그리고 지금까지 가장 좋았던 구성들을 보여주고 시도해 볼 새로운 구성을 제안하게 합니다. 이번 글에서는 LLM 기반 자동 튜너가 어떻게 동작하는지 설명하고, B200에서 33개(커널 11개 × 형태(shape) 3가지) 사례에 대해 LLM 기반 탐색과 LFBO 탐색을 비교한 벤치마크 결과를 보여줍니다. 그 결과, 새로운 LLM 기반 접근법은 10배 더 적은 구성을 컴파일·벤치마크하면서도 LFBO 수준의 커널 성능에 도달하며, 이는 6.7배 짧은 실제 소요 시간으로 이어집니다. 또한 두 방식의 장점을 결합한 하이브리드 탐색도 소개하는데, 이는 LLM으로 성능이 좋은 구성에 빠르게 도달한 뒤 LFBO로 세밀한 탐색을 수행합니다.
> LFBO is a strong baseline which works well, but it still grinds through hundreds of compile-and-benchmark cycles per kernel. What if, instead of starting the search blindly, you could ask an LLM to reason about the kernel and propose configurations? That's the LLM-guided autotuner – for each round of autotuning, an LLM is shown the kernel, the workload, and the best-so-far configs to propose new configs to try. In this blog, we describe how the LLM-guided autotuner works and show benchmarking results comparing the LLM-guided search to LFBO search on 33 (11 kernels x 3 shapes) cases on B200. Results show that the new LLM-based approach reaches LFBO-level kernel performance while compiling/benchmarking 10X less configs, leading to 6.7X less wall-clock time. We also introduce a hybrid search to combine the best of both worlds, which uses an LLM to quickly get to a performant configuration, followed by LFBO for fine-grained search.

## LLM 기반 자동 튜너의 동작 방식 / How the LLM-Guided Autotuner Works

새로운 LLM 기반 자동 튜너는 프롬프트와 피드백의 여러 주기를 거치며 동작하는 모집단 기반(population-based) 탐색을 실행합니다. 초기 단계에서 Helion은 커널과 관련 세부 정보를 LLM에 제공하여 후보 구성 집합을 요청합니다. LLM이 응답하면, Helion은 그 구성들을 컴파일·벤치마크하여 가장 성능이 좋은 구성들을 보존합니다. 이어지는 정교화 라운드(refinement round)에서는 가장 성공적이었던 구성들과 그 성능 지표, 그리고 성공 패턴 분석을 LLM에 제공하여 구체적인 변형(mutation)을 유도합니다. 유의미한 성능 향상이 감지되지 않으면 과정을 조기에 종료합니다. B200 GPU에서 실행되는 rms\_norm 커널을 예로 든 전체 과정은 다음과 같습니다.
> Operating through multiple cycles of prompts and feedback, the new LLM-based autotuner executes a population-based search. In the initial phase, Helion provides the kernel and the associated details to the LLM to ask for a set of candidate configurations. Once LLM responds, Helion compiles and benchmarks the configs, retaining the top-performing configurations. Subsequent refinement rounds then occur, where the LLM is given the most successful configs, their performance metrics, and an analysis of successful patterns to guide specific mutations. If no significant performance gains are detected, the process terminates early. An example process is shown for an rms\_norm kernel running on a B200 GPU.

### 초기 프롬프트 / The Initial Prompt

초기 프롬프트는 LLM의 역할을 설정하고, 조정 가능한 노브(knob)를 제공하며, 출력 계약(output contract)을 명시합니다.
> The initial prompt sets the role of the LLM, provides the knobs, and gives the output contract:

```
You are an expert GPU kernel autotuner for Helion/Triton kernels.

Use the provided Configuration Space and Default Configuration as the source of truth for allowed field names, scalar-vs-list, required list lengths, valid ranges and defaults.

Output contract:
- Return minified JSON on a single line: {"configs":[...]}. No markdown/fences/comments.
- Only specify fields you want to change; unspecified = default.
- For list-valued fields, emit a JSON array of the exact required length shown in the space.
- If unsure about a field's structure, length, or allowed values, omit it instead of guessing.
```

또한 Helion은 커널, 대상 하드웨어, 구성 공간을 함께 제공합니다. rms\_norm 커널 프롬프트에는 다음이 담깁니다.

- 커널 소스(Kernel source): 실제 `@helion.kernel` 소스 코드
- 입력 텐서(Input Tensors): 예: `arg[0]: shape=[4096, 1024], dtype=torch.float16, …`
- GPU 하드웨어(GPU Hardware): 예: `NVIDIA B200, 148 SMs, 178.4 GB, 2048 threads/SM`
- 구성 공간(Configuration Space): 타입/범위가 명시된 모든 조정 가능 필드
- 기본 구성(Default Configuration): 기준이 되는 baseline 구성

> Helion also provides the kernel, the target hardware, and the configuration space. The rms\_norm kernel prompt has:
> -   Kernel source: The actual @helion.kernel source code
> -   Input Tensors: e.g.: arg\[0\]: shape=\[4096, 1024\], dtype=torch.float16, …
> -   GPU Hardware: e.g.: NVIDIA B200, 148 SMs, 178.4 GB, 2048 threads/SM
> -   Configuration Space: Every tunable field with type/range
> -   Default Configuration: The baseline config.

Helion 컴파일러는 또한 커널을 분석하여 프롬프트에 휴리스틱을 추가합니다. rms\_norm의 경우 다음과 같습니다.
> The Helion compiler also analyzes the kernel to add heuristics to the prompt. For rms\_norm:

```
## Compiler Analysis

Helion's compiler statically analyzed this kernel's structure and derived the following structural priors. Treat them as strong starting points.

**Compiler-derived seed config(s):**
{"block_sizes":[1],"load_eviction_policies":["last","last","last","last","last"],"reduction_loops":[null]}
```

## 모델이 반환하는 것 / What the Model Returns

15개의 구성이 담긴 압축(minified) JSON입니다. rms\_norm에 대한 Opus-4.8의 첫 응답은 다음과 같습니다.
> A minified JSON with 15 configs. Opus-4.8's first reply for rms\_norm:

```json
{"configs":[
{"block_sizes":[1],"load_eviction_policies":["last","last","last","last","last"]},
{"block_sizes":[1]},
{"block_sizes":[4],"load_eviction_policies":["last", "..."],"num_warps":8},
{"block_sizes":[8],"load_eviction_policies":["last", "..."],"num_warps":8,"num_stages":2},
{"block_sizes":[16],"load_eviction_policies":["last", "..."],"num_warps":8,"num_stages":4},
{"block_sizes":[1],"load_eviction_policies":["last", "..."],"pid_type":"persistent_blocked"}
….
]}
```

Helion의 하네스(harness)는 이 응답을 파싱하고, 형식이 잘못되었거나 중복된 구성을 걸러낸 뒤 이를 컴파일·벤치마크합니다.
> Helion's harness parses this, drops malformed and duplicate configs to compile and benchmark them.

## 정교화 라운드 / Refinement Rounds

벤치마크 후, 이어지는 각 라운드는 탐색 상태로부터 구성된 정교화 프롬프트를 전송합니다.

- 탐색 상태(Search state): 라운드 번호, 모집단 크기, 지금까지의 최고 성능
- 앵커 구성(Anchor configs): 변형의 기준이 될 상위 구성들
- 결과(Results): 벤치마크된 구성들의 측정 성능
- 상위/실패 구성 패턴(Top/failed config patterns): 어떤 필드 값이 빠른 구성과 실패하거나 느린 구성에 상관되는지
- 다음 단계(Next Step): 구성의 실패율에 기반한 권장 사항

> After benchmarking, each subsequent round sends a refinement prompt built from the search state:
> -   Search state: Round number, population size, best perf so far.
> -   Anchor configs: Top configs to mutate around.
> -   Results: Measured performance for the benchmarked configs.
> -   Top/failed config patterns: Which field values correlate with fast vs. failed/slow configs.
> -   Next Step: Recommendations based on failure-rate of configs

따라서 모델은 가장 좋았던 구성에 앵커링하고 실패한 패턴은 피합니다. 이 피드백 루프는 각 라운드의 상대적 개선이 약 0.5% 미만으로 떨어지면 조기에 멈춥니다.
> So the model anchors on the best and avoids the patterns that failed. The feedback loop stops early if relative improvement from each round drops below ~0.5%.

## LLM으로 초기값을 잡은 LFBO: 두 방식의 장점을 모두 / LLM-Seeded LFBO: The Best of Both Worlds

LLM이 미세 아키텍처 노브(micro-architectural knob)를 탐색하지 않고 남겨두는 경향을 해결하기 위해, 하이브리드 전략(LLM으로 초기값을 잡은 LFBO 탐색, LLM-Seeded LFBO Search)도 탐구합니다. 이 접근법은 LLM과 LFBO의 상호 보완적인 장점을 결합합니다. LLM은 강력한 시작점을 제공하고, LFBO는 지역 탐색(local search)에 뛰어납니다.
> We also explore a hybrid strategy (LLM-Seeded LFBO Search) to address LLM's tendency to leave micro-architectural knobs unexplored. This approach merges the complementary advantages of both LLM and LFBO: the LLM provides a strong starting point, while LFBO excels at local search.

### 하이브리드 워크플로우 / The Hybrid Workflow

1. **1단계 – LLM 초기값 생성(LLM Seeding)**: 앞의 "초기 프롬프트"에서 설명한 대로, 한 라운드의 LLM 기반 탐색으로 과정을 시작합니다. Helion은 이를 검증하고 상위 구성들을 보존하기 위해 벤치마크합니다.
2. **인수인계(Handoff)**: 가장 성공적이었던 LLM 생성 구성들이 다음 단계에서 LFBO의 대리 모델(surrogate model)을 학습시키기 위한 시작점이 됩니다. 이로써 LFBO는 빈 상태에서 시작하는 대신 유망한 영역에 대한 사전 지식을 가지고 출발할 수 있습니다.
3. **2단계 – LFBO 정교화(LFBO Refinement)**: 초기 모집단이 채워진 상태로 LFBO 탐색을 실행합니다. LFBO는 반복 루프를 수행합니다. 즉, 랜덤 포레스트 분류기를 갱신하고, 상위 후보를 예측하며, 특징 중요도(feature importance)에 기반해 핵심 매개변수를 변형합니다. 이 주기는 성능 향상이 정체되거나 최대 반복 횟수(20회)에 도달할 때까지 계속됩니다.

> 1.  **Stage 1 – LLM Seeding**: The process begins with a single round of LLM-Guided Search, as described "The Initial Prompt" above. Helion benchmarks to validate and retain the top configs.
> 2.  **Handoff:** The most successful LLM-generated configs serve as the starting point for the next phase to train LFBO's surrogate model. This allows LFBO to begin with immediate knowledge of promising regions rather than starting from a blank slate.
> 3.  **Stage 2 – LFBO Refinement**: LFBO Search is executed with its initial population seeded. LFBO performs its iterative loop: updating the Random-Forest classifier, predicting top candidates, and mutating critical parameters based on feature importance. This cycle continues until performance gains stall or reaches the maximum number (20) of iterations.

시스템은 두 단계에 걸쳐 찾아낸 최적의 구성을 반환합니다. 고품질의 시작점과 정보에 기반한 대리 모델을 활용함으로써, 하이브리드 탐색은 차가운 상태에서 시작하는(cold) LFBO 탐색보다 훨씬 빠르게 수렴합니다. 이 효율성 덕분에 탐색 예산을 LLM이 찾지 못할 수 있는 특정 미세 아키텍처 노브의 미세 조정에 집중할 수 있습니다.
> The system returns the optimal configuration found across both stages. By leveraging a high-quality starting point and an informed surrogate, the hybrid search converges significantly faster than a cold LFBO search. This efficiency allows the search budget to be focused on fine-tuning the specific micro-architectural knobs that the LLM may not find.

## 벤치마크 결과 / Benchmarking Results

### 방법론 / The Methodology

NVIDIA B200에서 작은(small), 중간(medium), 큰(large) 형태에 대해 11개 커널 — matmul(정방 + split-K), grouped-GEMM, attention, fp8-attention, softmax, rms\_norm, rope, swiglu, mamba2, gated-delta-net — 을 대상으로, LFBO(full effort의 LFBOTreeSearch)와 Opus 4.8을 사용한 LLM 기반 탐색을 비교합니다.
> We compare LFBO (LFBOTreeSearch with full effort) to LLM-Guided Search using Opus 4.8 across 11 kernels — matmul (square + split-K), grouped-GEMM, attention, fp8-attention, softmax, rms\_norm, rope, swiglu, mamba2, and gated-delta-net — on small, medium, and large shapes on NVIDIA B200.

### 결과 1: 효율성의 승리 / Result 1: The Efficiency Win

바로 이 지점에서 LLM이 빛을 발합니다. 훨씬 적은 탐색 비용으로 LFBO의 품질에 필적합니다.
> This is where the LLM shines, matching LFBO's quality with significantly smaller search cost.

- **벤치마크된 구성의 기하평균: LLM 기반 자동 튜너가 9.8배 더 적은 구성(커널당 약 546개 대비 약 55개)을 사용**. 이는 기계에 독립적인 지표로, 새로운 접근법의 효능을 입증합니다.
- **실제 소요 시간의 기하평균: 384스레드 호스트에서 측정한 종단 간(end-to-end) 튜닝 시간이 6.7배 짧음(261초 대비 39초)**. 종단 간 튜닝 시간은 구성 생성(LLM의 경우 API 왕복 시간 포함), 모든 후보의 Triton/ptxas 컴파일, 그리고 모든 후보의 GPU 벤치마크로 이루어집니다.

> -   **Geomean configs benchmarked: 9.8X fewer configs (~55 vs ~546 per kernel) for LLM-guided autotuner**. This is a machine-independent metric that demonstrates the efficacy of the new approach.
> -   **Geomean wall-clock time: 6.7X less end-to-end tuning time (39 s vs 261 s), measured on a 384-thread host**. The end-to-end tuning time consists of config generation (for the LLM, including its API round-trips), Triton/ptxas compilation of every candidate, and GPU benchmarking of every candidate.

![커널당 탐색 효율: 평가된 구성 수 / Search efficiency: configs evaluated per kernel](/assets/blog/2026-06-18-from-minutes-to-seconds-llm-guided-autotuning-for-helion-kernels/search-efficiency.png){:style="width:100%"}

![커널당 자동 튜닝 비용 / Autotuning cost per kernel](/assets/blog/2026-06-18-from-minutes-to-seconds-llm-guided-autotuning-for-helion-kernels/autotuning-cost.png){:style="width:100%"}

종단 간 튜닝 시간은 후보 구성을 컴파일하는 데 지배되는데, 여기서 Helion은 CPU 코어에 걸쳐 후보들을 병렬로 사전 컴파일합니다. 이 호스트는 수백 개의 스레드를 가지고 있어 컴파일이 크게 병렬화됩니다. 코어 수가 더 적은 머신에서는 LLM의 약 10배 적은 구성이 비례적으로 더 큰 실제 소요 시간 감소로 이어질 것입니다. 기계에 독립적인 지표는 벤치마크된 구성의 수와, 최선의 구성이 최적 결과에 얼마나 빠르게 수렴하는지입니다(아래 참조).
> End-to-end tuning time is dominated by compiling candidate configs, where Helion precompiles them in parallel across CPU cores. As this host has 100s of threads, compilation is heavily parallelized. On a machine with fewer cores, the LLM's ~10X fewer configs would translate into a proportionally larger wall-clock time reduction. The machine-independent metrics are the number of configs benchmarked and how fast the best configs converge to their optimal results (shown below).

## 결과 2: LLM은 LFBO 예산의 첫 약 7% 안에서 수렴한다 / Result 2: LLM Converges in the First ~7% of LFBO Budget

탐색 노력 대비 지금까지의 최선 구성(best-config-so-far)을 그려보면 그 차이가 선명하게 드러납니다. LLM은 수십 개의 구성만으로 자신의 정체 구간(plateau)에 도달합니다.
> Plotting best-config-so-far against search effort makes the difference vivid. The LLM drops to its plateau in a few dozen configs.

![탐색 노력 대비 수렴 / Convergence vs Search Effort](/assets/blog/2026-06-18-from-minutes-to-seconds-llm-guided-autotuning-for-helion-kernels/convergence.png){:style="width:100%"}

수렴이 일어나는 12개 커널 전반에서, LLM은 LFBO 예산의 대략 첫 7% 안에서 자신의 정체 구간에 도달합니다. grouped GEMM(g=4, m=512)의 경우, 같은 커널 성능에서 LFBO보다 18배 더 적은 구성을 사용합니다.
> Across all 12 convergence kernels, the LLM drops to its plateau inside roughly the first ~7% of LFBO's budget. On grouped GEMM (g=4, m=512), that's 18X fewer configs than LFBO at the same kernel performance.

## 결과 3: LLM은 LFBO 수준의 성능을 낸다 / Result 3: LLM Delivers LFBO-level Performance

커널 성능 면에서 LLM은 LFBO와 대략 동등하며, LLM 커널/LFBO 커널 지연 시간의 기하평균 성능은 1.009배입니다.
> On kernel performance, the LLM is roughly on-par with LFBO, with the geomean performance of LLM kernel/LFBO kernel latency being 1.009X.

![커널별 성능 / Per Kernel Performance](/assets/blog/2026-06-18-from-minutes-to-seconds-llm-guided-autotuning-for-helion-kernels/per-kernel-performance.png){:style="width:100%"}

따라서 LLM은 좋은 구성을 빠르게 제공하는 반면, LFBO는 더 많은 구성 탐색과 튜닝 시간을 대가로 더 나은 성능을 낼 수 있습니다. LLM이 커널 성능에서 LFBO에 5% 넘게 뒤지는 사례는 8건 있습니다.
> Hence LLM gives you a good config fast, while LFBO can outperform at the cost of more config exploration and tuning time. There are 8 cases where LLM loses to LFBO by more than 5% in kernel performance.

## 하이브리드 탐색이 격차를 메울 수 있을까? / Can the Hybrid Search Close the Gap?

LLM의 약점이 노브의 미세 조정을 남겨두는 것이라면, LLM으로 초기값을 잡은 LFBO TreeSearch 하이브리드 탐색 전략을 사용할 수 있습니다.
> If LLM's weakness is leaving fine tuning the knobs, we can use our hybrid search strategy with LLM-Seeded LFBO TreeSearch.

LLM이 LFBO에 5% 넘게 뒤지는 8건의 사례에 대해 하이브리드 탐색을 실행했습니다.
> We ran the hybrid search on the 8 cases where the LLM trails LFBO by more than 5%.

![하이브리드가 격차를 메우는가? / Does the Hybrid Close the Gap?](/assets/blog/2026-06-18-from-minutes-to-seconds-llm-guided-autotuning-for-helion-kernels/hybrid-gap.png){:style="width:100%"}

하이브리드 전략은 모든 사례에서 커널 성능을 개선하며, 8건 중 6건에서 LFBO와의 격차를 메웁니다. mamba2 계열은 여전히 LFBO보다 성능이 낮으며, 이 격차를 줄이기 위해 LLM 휴리스틱 개선을 연구하고 있습니다.
> The hybrid strategy improves kernel performance in all cases and closes the gap to LFBO in 6/8 cases. The mamba2 family still does worse than LFBO and we are investigating improving the LLM heuristics to close this gap.

자동 튜닝 시간 측면에서, 하이브리드 탐색은 LFBO보다 훨씬 효율적입니다. 8개 커널 전반에서 LFBO보다 4배 적은 구성을 탐색하여, 종단 간 자동 튜닝 시간이 3배 빨라집니다. LLM 단독, 하이브리드, LFBO를 비교한 기하평균 결과는 아래와 같습니다.
> In terms of autotuning time, hybrid search is significantly more efficient than LFBO. Across the 8 kernels, it explores 4X fewer configs LFBO, leading to 3X faster end-to-end autotuning time. The geomean results comparing LLM-only, hybrid, and LFBO are shown below.

| | **LLM 단독 / LLM-only** | **하이브리드 / Hybrid** | **전체 LFBO / Full LFBO** |
|---|---|---|---|
| **자동 튜닝 시간 / Autotuning Time** | 44초 | 111초 | 328초 |
| **탐색한 구성 수 / Explored Configs** | 59 | 186 | 686 |

아래에는 탐색한 구성 수와 튜닝 시간을 비교한 개별 커널 결과도 함께 제시합니다.
> We also present the individual kernel results below comparing the number of explored configs as well as their tuning times:

![LLM 단독 vs 하이브리드 vs LFBO 자동 튜닝 / LLM-only vs Hybrid vs LFBO-Autotuning](/assets/blog/2026-06-18-from-minutes-to-seconds-llm-guided-autotuning-for-helion-kernels/llm-vs-hybrid-vs-lfbo.png){:style="width:100%"}

## 모델이 중요할까? / Does the Model Matter?

위의 모든 내용은 단일 모델인 Claude Opus-4.8을 사용했습니다. 작업을 수행하는 모델이 결정적인 역할을 하는지, 아니면 어떤 유능한 LLM이든 같은 결과에 도달하는지 궁금할 수 있습니다. 이를 확인하기 위해, OpenAI gpt-5.5와 Claude Sonnet-4.6 두 모델을 추가로 사용하여 전체 33개 커널 인스턴스에 대해 LLM 단독 탐색(LLM 기반 탐색)을 벤치마크하고 Opus 4.8 기준선과 비교했습니다.
> Everything above used one model: Claude Opus-4.8. One may ask whether the model doing the work is load-bearing, or whether any capable LLM gets you the same place. To this end, we benchmark the LLM-only search (LLM-Guided Search) across the full 33-kernel instances with two more models, OpenAI gpt-5.5 and Claude Sonnet-4.6, to compare to the Opus 4.8 baseline.

| **모델 / model** | **Opus-4.8 대비 기하평균 성능 / geomean perf vs Opus-4.8** | **탐색한 구성의 기하평균 / Geomean configs explored** |
|---|---|---|
| **Opus-4.8** | 1.00 (기준선 / baseline) | 55 |
| **gpt-5.5** | 0.98 | 61 |
| **Sonnet-4-6** | 1.03 | 51 |

기하평균으로 보면 세 모델 모두 매우 비슷하게 동작했으며, 흥미롭게도 Sonnet-4.6은 가장 적은 수의 구성으로 이를 해냈습니다.
> In geomean, all 3 models performed very similarly and interestingly, Sonnet-4.6 did it with the fewest number of configs.

## 결론 / Conclusions

우리가 답하고자 했던 질문은 "LLM이 LFBO 탐색만큼 잘, 그러나 훨씬 저렴하게 Helion 커널을 자동 튜닝할 수 있는가?"였습니다. B200에서 벤치마크한 33개 커널 모음 전반에서, 그 답은 '그렇다'입니다.
> The question we set out to answer was, "Can an LLM autotune Helion kernels as well as the LFBO search, but far more cheaply? Across a 33-kernel suite, benchmarked on B200, the answer is yes.

**효율성 향상이 상당하다**: LLM 기반 자동 튜너는 **LFBO 예산의 7%** 안에서 LFBO 품질의 결과로 수렴하며, **약 10배 더 적은 구성을 탐색하고, 실제 소요 시간을 약 6.7배 줄여**, 개발 속도를 대폭 끌어올립니다.
> **The efficiency gain is substantial:** The LLM-guided autotuner converges to LFBO-quality results in **7% of LFBO's budget, explores ~10X fewer configurations, with ~6.7X reduction in wall-clock time**, offering a massive boost in developer velocity.

**LLM이 LFBO 수준의 성능에 도달한다**: LLM 기반 자동 튜너는 대부분의 커널에서 LFBO 탐색과 동등하며 일부에서는 앞서기까지 합니다. 더 높은 자동 튜닝 시간을 대가로 LFBO가 이기는 사례도 있습니다.
> **LLM reaches LFBO-level performance**: The LLM-guided autotuner ties the LFBO search on most kernels and even wins on some. There are cases that LFBO wins at the cost of higher autotuning time.

**하이브리드 전략이 격차를 메운다**: 하이브리드 접근법(LLM 초기값 생성 후 LFBO 정교화)은 LFBO 탐색보다 약 3배 저렴하게 유지하면서도 남은 성능을 회복할 수 있습니다.
> **Hybrid strategy bridges the gap**: The hybrid approach (LLM seeding followed by LFBO refinement) can recover remaining performance while remaining ~3X cheaper than a LFBO search.

**실용적인 처방**: 간소화된 워크플로우를 위해, 먼저 LLM 단독 탐색을 시도하여 고성능 커널을 빠르게 찾아내는 것을 권합니다. 성능을 극대화하려면 하이브리드 탐색을 적용하여 마지막 성능 향상까지 정교화하고 확보할 수 있습니다. 앞으로는 LLM 기반 자동 튜너와 하이브리드 자동 튜너 모두의 효과를 더욱 끌어올리도록 휴리스틱을 개선할 계획입니다.
> **The Practical Recipe**: For a streamlined workflow, we suggest trying the LLM-only search to rapidly identify a high-performance kernel. To maximize performance, users can apply the hybrid search to refine and capture the final performance gains. Moving forward, we plan to enhance the heuristics to further boost the effectiveness of both the LLM-guided and hybrid autotuners.
