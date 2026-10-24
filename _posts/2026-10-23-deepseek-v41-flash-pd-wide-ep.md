---
layout: blog_detail
title: "DeepSeek-V4.1-Flash 대규모 서빙: vLLM이 DeepInfer와 대등한 성능을 내고 GB300 NVL72까지 확장하다"
author: Inferact and the vLLM Team
ext_author: Junghwan Park (박정환)
category: ["vllm.ai", "translation"]
date: 2026-10-23 12:00:00
org_title: "Serving DeepSeek-V4.1-Flash at Scale: vLLM Matches DeepInfer and Scales to GB300 NVL72"
org_link: https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep
---

![GB300 NVL72 랙 한 대에서 P/D 분리와 Wide-EP를 적용해 SemiAnalysis AgentX로 측정한 DeepSeek-V4.1-Flash 처리량 7.9배 향상, DeepInfer와 대등한 성능 / 7.9x throughput for DeepSeek-V4.1-Flash with P/D and Wide-EP on a GB300 NVL72 rack, measured on SemiAnalysis AgentX, and on par with DeepInfer](/assets/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep/hero.png){:style="width:100%"}

**TL;DR:** NVIDIA GB300 NVL72 랙 한 대에서, 프리필/디코드 분리(P/D disaggregation)와 Wide-EP(전문가 병렬화를 데이터 병렬화와 결합해 많은 GPU로 넓힌 방식)를 적용한 vLLM은 SemiAnalysis AgentX에서 DeepSeek-V4.1-Flash를 **GPU당 954K tokens/s** 로 서빙합니다. 이때 사용자당 P90 속도는 196 tokens/s입니다. 이는 [첫 번째 글](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash)의 단일 노드 결과보다 **7.9배** 높은 수치이며, DeepSeek의 비혼잡 시간대(off-peak) 가격으로 환산하면 **GPU 시간당 $38.96** 입니다. 특화된 디코드 경로는 단일 사용자 기준 **826 tokens/s** 에 도달합니다. DeepSeek의 자체 엔진인 DeepInfer와 비교하면, vLLM의 디코드 처리량은 배치 크기 8~256 전 구간에서 **6%** 이내이고, 프리필도 DeepInfer가 공개한 곡선을 따라갑니다.
> **TL;DR:** On one NVIDIA GB300 NVL72 rack, vLLM with P/D disaggregation and Wide-EP serves DeepSeek-V4.1-Flash on SemiAnalysis AgentX at **954K tokens/s per GPU** at a P90 of 196 tokens/s per user, **7.9×** our single-node result from the [first post](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash) and **$38.96 per GPU-hour** at DeepSeek's off-peak prices. A specialized decode path reaches **826 tokens/s** for a single user. Against DeepInfer, DeepSeek's own engine, vLLM's decode throughput is within **6%** across batch sizes 8–256, and its prefill tracks DeepInfer's published curve.

이 효율은 vLLM에 이미 있는 구성 요소를 공들여 엔지니어링한 결과이며, DeepSeek이 오픈소스로 공개한 커널 위에 만들어졌습니다. 이번 글에서는 성능 연구 과정을 따라가며 각 최적화를 하나씩 분석하고, 그 과정에서 얻은 교훈을 공유합니다:

- **범용 엔진도 특정 모델 전용 엔진과 대등한 성능을 낼 수 있습니다**
- **최적화하기 전에 모델을 이해합니다**
- **프리필에는 작은 DEP 인스턴스가 가장 좋습니다**[^terms]
- **Wide-EP 디코드는 EPLB와 함께 잘 확장됩니다**
- **NVL72 규모에서는 CPU 오버헤드가 결정적인 문제가 됩니다**
- **특화된 디코드 경로가 낮은 지연 시간을 가능하게 합니다**

> The efficiency comes from deliberate engineering on components that vLLM already has, built on the kernels DeepSeek open-sourced. This post walks through our performance study, breaks down each optimization, and shares the lessons we learned:
>
> - **A general-purpose engine can match a model-specific one**
> - **Understand the model before optimizing**
> - **Small DEP instances work best for prefill**[^terms]
> - **Wide-EP decode scales well with EPLB**
> - **CPU overhead becomes critical at NVL72 scale**
> - **A specialized decode path unlocks low latency**

## 성능 개요 / Performance overview

<iframe class="vllm-embed" src="/assets/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep/dsv41-pd-agentx-curve.html" title="NVIDIA GB300 NVL72의 SemiAnalysis AgentX, P/D 분리와 Wide-EP 대 단일 노드 통합 결과 / SemiAnalysis AgentX on NVIDIA GB300 NVL72, P/D and Wide-EP vs. the aggregated single-node result" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

*그림 1: NVIDIA GB300 NVL72에서 측정한 SemiAnalysis AgentX 결과. P/D 분리와 Wide-EP를 적용한 vLLM(금색)은 [첫 번째 글](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash)의 단일 노드 통합(aggregated) 결과보다 처리량을 7.9배 더 높였습니다. 처리량에는 캐시된 입력 토큰, 캐시되지 않은 입력 토큰, 출력 토큰이 모두 포함됩니다([출처](https://inferencex.semianalysis.com/inference/deepseek-v41-flash?i_xmode=interactivity&i_seq=agentic-traces&g_model=DeepSeek-V4.1-Flash&i_best=0&i_active=gb300_dynamo-vllm)). / Figure 1: SemiAnalysis AgentX on NVIDIA GB300 NVL72: vLLM with P/D disaggregation and Wide-EP (gold) achieves a further 7.9× throughput increase over our aggregated single-node result from the [first post](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash). Throughput counts cached input, uncached input and output tokens ([source](https://inferencex.semianalysis.com/inference/deepseek-v41-flash?i_xmode=interactivity&i_seq=agentic-traces&g_model=DeepSeek-V4.1-Flash&i_best=0&i_active=gb300_dynamo-vllm)).*

### DeepInfer와 비교 / Compared to DeepInfer

[DeepInfer](https://github.com/deepseek-ai/DeepInfer)는 DeepSeek이 DeepSeek-V4.1-Flash를 위해 만든 자체 추론 엔진으로, 이번 주에 프리필 곡선과 디코드 곡선을 함께 공개한 오프라인 엔진입니다. 이 모델에 가장 적합한 비교 대상이므로, 최적화 내용을 다루기 전에 vLLM이 DeepInfer와 비교해 어디쯤에 있는지 먼저 보여 드립니다.
> [DeepInfer](https://github.com/deepseek-ai/DeepInfer) is DeepSeek's own inference engine for DeepSeek-V4.1-Flash, released this week as an offline engine with published prefill and decode curves. It is the most relevant reference for this model, so before going into our optimizations we show where vLLM stands against it.

**디코드.** DeepSeek의 디코드 구성(131K 컨텍스트, nextn=3)에서, GB200 위의 vLLM은 배치 크기 8~256 전 구간에서 DeepInfer가 공개한 결과의 6% 이내 처리량을 달성합니다(그림 2).
> **Decode.** Under DeepSeek's decode configuration (131K context, nextn=3), vLLM on GB200 achieves throughput within 6% of DeepInfer's published results across batch sizes 8 to 256 (Figure 2).

<iframe class="vllm-embed" src="/assets/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep/dsv41-pd-decode-vs-deepinfer.html" title="GB200 DEP32 위의 vLLM이 DeepSeek의 디코드 곡선을 따라감 / vLLM on GB200 DEP32 tracks DeepSeek's decode curve" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

*그림 2: 131K 컨텍스트, nextn=3에서 배치 크기에 따른 GPU당 디코드 출력 처리량. GB200 위의 vLLM과 DeepInfer의 공개 결과를 비교했으며, DeepInfer 값은 README의 차트에서 읽었습니다([출처](https://github.com/deepseek-ai/DeepInfer)). / Figure 2: Decode output throughput per GPU vs. batch size at 131K context, nextn=3. vLLM on GB200 vs. DeepInfer's published results, read from the charts in its README ([source](https://github.com/deepseek-ai/DeepInfer)).*

**프리필.** DeepInfer가 공개한 프리필 곡선은 중복 전문가(redundant expert) 32개와 140토큰 제한된 리플레이(bounded replay)를 쓴 EP4DP4 구성으로 측정한 것이며, 차트에는 하드웨어가 적혀 있지 않습니다. 그 값은 차트에서 읽었습니다. 이번 실행에서는 GB300에서 중복 전문가 없이 DEP2, 128토큰 리플레이, 자체 프롬프트를 사용했습니다. 구성이 다르므로, 두 곡선 사이의 차이를 엔진 간 속도 향상으로 해석하지 않습니다. 그림 3이 보여 주는 것은 vLLM의 DEP2 프리필이 DeepSeek이 공개한 범위 전반에서 DeepSeek 자체 엔진의 곡선을 따라간다는 점입니다.
> **Prefill.** DeepInfer's published prefill curve uses EP4DP4 with 32 redundant experts and a 140-token bounded replay, on hardware the chart does not state, and we read its values off the chart. Our run uses DEP2 on GB300 with no redundant experts, a 128-token replay and our own prompts. Because the setups differ, we do not read the gap between the two curves as an engine-to-engine speedup. What Figure 3 shows is that vLLM's DEP2 prefill tracks DeepSeek's own engine across the range they published.

<iframe class="vllm-embed" src="/assets/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep/dsv41-pd-prefill-vs-deepinfer.html" title="프리필 곡선: vLLM DEP2 대 DeepInfer EP4DP4 / Prefill curve: vLLM DEP2 vs. DeepInfer EP4DP4" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

*그림 3: 새 토큰 수에 따른 프리필 처리량. GB300 위의 vLLM DEP2와 DeepInfer가 공개한 곡선(EP4DP4, 중복 전문가 32개, 140토큰 리플레이, 하드웨어 미기재)을 비교했으며, DeepInfer 값은 README의 차트에서 읽었습니다([출처](https://github.com/deepseek-ai/DeepInfer)). / Figure 3: Prefill throughput vs. new tokens. vLLM DEP2 on GB300 vs. DeepInfer's published curve (EP4DP4, 32 redundant experts, 140-token replay, hardware not stated), read from the chart in its README ([source](https://github.com/deepseek-ai/DeepInfer)).*

두 엔진의 결과가 가까운 이유는 양방향으로 계보를 공유하기 때문입니다. DeepInfer는 원래 vLLM을 기반으로 만들어졌습니다. vLLM 역시 MegaAttn, MegaMoE, Mega-mHC를 포함한 DeepSeek 라이브러리의 커널을 꾸준히 통합해 왔고, 그 커널을 중심으로 스택의 나머지 부분을 폭넓게 최적화했습니다. DeepInfer는 오프라인 모드에서 단일 모델만을 대상으로 합니다. 반면 같은 vLLM 빌드는 NVL72 랙 전체에 걸쳐 P/D 분리와 Wide-EP를 적용해 이 모델을 온라인으로도 서빙하며, 이 글의 나머지 부분에서 측정하는 것이 바로 이 구성입니다. 이어지는 섹션에서 그 작업을 설명합니다.
> The two engines land close because they share a lineage in both directions. DeepInfer was originally built on vLLM, and vLLM in turn has continuously integrated kernels from DeepSeek's libraries, including MegaAttn, MegaMoE and Mega-mHC, and has extensively optimized the rest of the stack around them. DeepInfer targets a single model in offline mode; the same vLLM build also serves this model online with P/D disaggregation and Wide-EP across a full NVL72 rack, which is what the rest of this post measures. The sections that follow describe that work.

## 내부 구조: 속도는 어디에서 오는가 / Under the hood: where the speed comes from

### 먼저 모델을 측정합니다 / Measure the model first

튜닝에 앞서, DeepSeek-V4.1-Flash의 효율적인 아키텍처 설계를 수치로 파악하는 일부터 시작했습니다:

- **작은 KV 캐시 사용량.** 레이어 간 KV 공유(cross-layer KV sharing)와 FP4 압축 덕분에 전역(global) KV 캐시는 토큰당 890바이트에 불과합니다([DeepSeek-V4.1-Flash 기술 보고서(Technical Report)](https://arxiv.org/abs/2609.19969) 참고).
- **가벼운 어텐션.** 레이어 2, 8, 14는 전체 컨텍스트 인덱싱에 2배 압축을 사용합니다. 제한된 리플레이를 쓰면 레이어 20은 모든 토큰의 전역 KV를 기록하지만, 인덱싱과 어텐션은 마지막 128개 토큰에 대해서만 실행합니다. 레이어 21~39는 그 토큰만 처리합니다.
- **긴 컨텍스트에서도 디코드의 병목은 MoE입니다(MoE-bound).** 아래 그림 5를 참고하세요. 128K 컨텍스트에서 랭크(rank)당 요청 64개로 DEP4 디코딩을 하면, MoE가 타깃 스텝 시간의 62%를 차지하는 반면 어텐션과 인덱서(indexer)를 합친 비중은 13%에 불과합니다.

> Before tuning, we started by quantifying DeepSeek-V4.1-Flash's efficient architectural designs:
>
> - **Small KV cache footprint.** Only 890 bytes per token for global KV cache, thanks to cross-layer KV sharing and FP4 compression (see the [DeepSeek-V4.1-Flash Technical Report](https://arxiv.org/abs/2609.19969)).
> - **Lightweight attention.** Layers 2, 8 and 14 use 2× compression for full-context indexing. With bounded replay, layer 20 writes global KV for all tokens but runs indexing and attention only on the last 128; layers 21–39 process only those tokens.
> - **Decode is MoE-bound even in a long context.** See Figure 5 below. With DEP4 decoding at 128K context and 64 requests per rank, MoE takes 62% of the target step, while attention and the indexer together take only 13%.

[^terms]: DEP*n* 은 *n* 개 GPU에 걸쳐 데이터 병렬(data-parallel) 어텐션과 전문가 병렬화(expert parallelism)를 함께 쓰는 구성입니다. 타깃 순전파(target forward)는 각 디코드 스텝에서 메인 모델이 수행하는 전체 순전파로, 드래프트 토큰을 검증하고 출력을 만들어 냅니다. 드래프트 순전파(draft forward)는 DSpark 드래프트 모델의 전체 순전파로, 다음 스텝에 쓸 토큰을 제안합니다.
    > DEP*n* is data-parallel attention plus expert parallelism over *n* GPUs. Target forward is the main model's full forward pass at each decode step, which verifies the draft tokens and produces the output; draft forward is the DSpark draft model's full forward pass, which proposes the tokens for the next step.

아래 표는 GB300에서 DEP32, 랭크당 요청 16개, 요청당 토큰 6개(토큰 1개와 DSpark 드래프트 토큰 5개), 모든 요청의 컨텍스트 길이가 같은 조건에서 측정한 타깃 모델의 순전파 시간입니다. 128K에서 855K로 컨텍스트가 6.7배 늘어나도 타깃 모델의 디코드 지연 시간은 0.6 ms(+7%)만 늘어납니다.
> The table below shows the target model's forward time at DEP32, with 16 requests per rank, 6 tokens per request (one token plus five DSpark draft tokens), and every request at the same context length, on GB300. Going from 128K to 855K, 6.7× the context only adds 0.6 ms (+7%) to the target model decode latency.

| 요청당 컨텍스트 / Context per request | 64K | 128K | 256K | 512K | 855K |
|---|---:|---:|---:|---:|---:|
| 타깃 순전파 (ms) / Target forward (ms) | 8.86 | 8.93 | 9.03 | 9.25 | 9.53 |
| 128K 대비 / vs. 128K | −0.8% | – | +1.1% | +3.6% | +6.7% |

이러한 특성이 이후 최적화 전략의 방향을 정합니다:

- **프리필은 작은 DEP 크기로 실행해야 합니다.** B300이나 GB300에서는 DEP2 인스턴스 하나만으로도 모델 가중치와 많은 동시 세션의 KV 캐시를 모두 담을 수 있습니다. 월드 크기(world size)가 작으면 DP/EP 부하 분산(load balancing)이 훨씬 단순해집니다.
- **KV 용량이 배치 크기를 제한하는 경우는 드뭅니다.** KV 캐시가 HBM에 넉넉하게 들어가므로, KV 용량 문제를 우회하는 대신 각 연산을 연산 한계나 메모리 대역폭 한계까지 밀어붙일 수 있습니다. 또한 선점(preemption)이나 KV 캐시 압박이 없어서 측정 스윕(sweep)도 일관되고 노이즈가 적습니다.
- **Wide-EP는 디코드에 효과적입니다.** MegaMoE 디코드의 병목은 가중치 읽기이며, Wide-EP는 가장 큰 비중을 차지하는 이 MoE 비용을 직접 줄입니다.
- **디코드의 DP 부하 분산은 더 단순합니다.** 이 모델의 디코드 스텝 시간은 컨텍스트 길이에 따라 거의 달라지지 않으므로, DP 부하 분산을 랭크당 요청 수 기준의 최소 부하(least-load) 방식으로 단순화할 수 있습니다.
- **합성(synthetic) 워크로드도 여전히 디코드 성능을 잘 대표합니다.** 평균 입력 길이와 출력 길이를 사용하면, 전체 에이전트 트레이스(agentic trace)를 재실행하지 않고도 성능을 잘 추정할 수 있습니다.

> These characteristics shape the rest of our optimization strategy:
>
> - **Prefill should run at a small DEP size.** Even a DEP2 instance on B300 or GB300 can hold both the model weights and the KV cache for many concurrent sessions. A smaller world size greatly simplifies DP/EP load balancing.
> - **KV capacity rarely limits batch size.** The KV cache fits comfortably in HBM, allowing us to push each operation toward its compute or memory bandwidth limits instead of working around KV capacity. This also keeps our sweep consistent and low-noise with no preemption or KV cache pressure.
> - **Wide-EP is effective for decode.** MegaMoE decode is bound by weight reads, and Wide-EP directly cuts this dominating MoE cost.
> - **Decode DP load balancing is simpler.** This model's decode step time varies little with context length, so DP load balancing can be simplified to least-load by request count per rank.
> - **Synthetic workloads remain representative for decode.** Using average input and output lengths gives us a good estimate of the performance without replaying the full agentic trace.

#### 측정 방법 / How we measured

방법론을 지나치게 복잡하게 만들지 않으면서 성능을 정확하게 측정하는 일은 간단하지 않습니다. 이 문제를 두 가지 질문으로 나눴습니다. 프리필 스텝이나 디코드 스텝에 드는 비용은 얼마여야 하는가, 그리고 DP와 EP의 불균형(imbalance)으로 인한 추가 오버헤드는 얼마나 되는가입니다.

- **프리필과 디코드의 스텝 시간.** `vllm bench forward`([#61339](https://github.com/vllm-project/vllm/pull/61339))는 정확히 지정한 프리필 배치나 디코드 배치에 대해, DP 랭크와 EP 랭크 전반에서 실제 `execute_model` 경로를 거치는 모델 순전파 시간을 측정합니다. 이 도구는 GPU 시간만 측정하므로 깔끔한 트레이스를 얻을 수 있습니다. 또한 vLLM의 `DecodeBenchConnector`는 주어진 동시성(concurrency)에서 엔진 수준의 엔드투엔드(end-to-end) 디코드 성능을 측정합니다. GPU만 측정한 스텝 시간과 서빙 중 스텝 시간을 비교하면 예상하지 못한 오버헤드가 드러납니다.
- **프리필의 DP 불균형: 미러 재실행과 지연 재실행.** 에이전트 트레이스는 실행할 때마다 달라지므로 DP 불균형만 따로 떼어 보기 어렵습니다. 그래서 모든 DP 랭크에서 똑같은 요청을 재실행합니다. *미러(mirror)* 실행에서는 모든 랭크가 동시에 요청을 받습니다. *지연(lag)* 실행에서는 랭크 *r* 이 같은 요청을 *r* × 5초 늦게 받습니다. 따라서 랭크들의 진행 시점은 어긋나지만 전체 작업량은 같습니다. 두 실행의 차이가 DP 불균형의 비용입니다.
- **디코드의 EP 불균형: 실제 라우팅과 균등 라우팅.** 각 MoE 레이어에 대해 균형도(balancedness)를 랭크 부하의 평균을 랭크 부하의 최댓값으로 나눈 값으로 정의하고, 스텝마다 그리고 EPLB의 윈도우 전체에 대해 계산합니다. 균형 잡힌 기준선으로는, 같은 디코드 배치를 토큰이 균등하게 무작위로 라우팅되도록 실행합니다. 이 기준선에도 샘플링 노이즈는 있습니다. DEP32에서 GPU당 요청 72개일 때, 균등 라우팅의 균형도는 타깃(top-6 라우팅)이 약 0.96, 드래프트(top-3)가 0.94입니다. 실제 라우팅과 균등 라우팅 사이의 차이가 전문가 균형 조정으로 회복할 수 있는 최대치이며, EPLB 실행 결과는 그 차이 가운데 실제로 얼마나 회복하는지 보여 줍니다.

> Measuring performance accurately without overcomplicating the methodology is not trivial. We break it into two questions: what should a prefill or decode step cost, and how much additional overhead comes from DP and EP imbalance?
>
> - **Prefill and decode step time.** `vllm bench forward` ([#61339](https://github.com/vllm-project/vllm/pull/61339)) measures the model-forward time for an exact prefill or decode batch through the actual `execute_model` path across DP and EP ranks. This measures GPU time only and gives a clean trace. Further, vLLM's `DecodeBenchConnector` measures engine-level end-to-end decode performance at a given concurrency. The gap between GPU-only and serving step times reveals unexpected overhead.
> - **Prefill DP imbalance: mirror vs. lag replays.** Agentic traces vary from run to run, so DP imbalance is hard to isolate. We therefore replay identical requests on every DP rank. In a *mirror* run, all ranks receive their requests at the same time. In a *lag* run, rank *r* receives the same requests *r* × 5 s later, so the ranks are out of step but do the same total work. The difference between the two runs is the cost of DP imbalance.
> - **Decode EP imbalance: real vs. uniform routing.** For each MoE layer, we define balancedness as the mean rank load divided by the max rank load, per step and over EPLB's window. As the balanced reference, we run the same decode batch with tokens routed uniformly at random. This reference still has sampling noise: at DEP32 with 72 requests per GPU, uniform routing reaches about 0.96 balancedness for the target (top-6 routing) and 0.94 for the draft (top-3). The gap between real and uniform routing is the most that expert balancing can recover, and the EPLB run shows how much of that gap it actually recovers.

### 프리필: DEP2가 가장 좋습니다 / Prefill: DEP2 works best

앞에서 설명했듯이, 프리필에는 작은 DEP 크기가 유리합니다. 동시성 256의 AgentX 프리필 전용 실행에서 DEP4의 GPU당 처리량은 DEP2의 83%에 그쳤습니다(53.3K 대 64.4K tokens/s). 프리픽스 캐싱(prefix caching)을 끄고 미러 재실행과 지연 재실행을 비교한 결과는 DP 불균형을 원인으로 가리킵니다. 랭크들의 진행 시점이 어긋나면 DEP4는 처리량의 13.6%를 잃습니다.[^earlier-build] 그래서 DEP2를 간단한 기본값으로 삼았습니다.
> As discussed above, a small DEP size is appealing for prefill. In an AgentX prefill-only run at concurrency 256, DEP4 gave only 83% of DEP2's per-GPU throughput (53.3K vs. 64.4K tokens/s). Mirror and lag replays with prefix caching off point to DP imbalance: when the ranks fall out of step, DEP4 loses 13.6% of its throughput.[^earlier-build] This makes DEP2 a simple default for us.

| 인스턴스당 프리필 처리량 (tokens/s) / Prefill throughput per instance (tokens/s) | 미러 (균형) / mirror (balanced) | 지연 (불균형) / lag (imbalanced) | 손실 / loss |
|---|---:|---:|---:|
| DEP2 | 138.7K | 130.5K | 5.9% |
| DEP4 | 267.6K | 231.1K | 13.6% |

[^earlier-build]: 이전 빌드에서 측정했습니다. 불균형 비용은 커널이 아니라 DEP 크기에서 비롯되는 특성입니다.
    > Measured on an earlier build; the imbalance cost is a property of DEP size, not of the kernels.

#### 트레이스 뒤에 숨어 있던 최적화 / Optimizations hidden behind the trace

[첫 번째 글](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash)에서는 폭넓은 프리필 최적화를 다뤘습니다. 트레이스가 깔끔해지자 남아 있던 병목, 특히 컨텍스트 길이에 따라 커지는 병목을 찾을 수 있었습니다. 예를 들어 GB300 GPU 2개에서 8K 토큰 프리필은 **이전 컨텍스트가 없으면 87.9 ms** 가 걸리지만, **512K 컨텍스트가 캐시되어 있으면 189.4 ms** 가 걸립니다. 다음 세 가지 병목을 추가로 해결했습니다:
> Our [first post](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash) covered a broad set of prefill optimizations. Cleaner traces helped us identify the remaining bottlenecks, particularly those that grow with context length. For example, on two GB300 GPUs, an 8K-token prefill takes **87.9 ms without prior context**, but **189.4 ms with 512K cached context**. We addressed three additional bottlenecks:

<details markdown="1">
<summary><strong>SWA 제한된 리플레이를 레이어 20까지 확장</strong> (<a href="https://github.com/vllm-project/vllm/pull/59894">#59894</a>): 32K 새 토큰에서 +19.9% / <strong>Extend SWA bounded replay to layer 20</strong> (<a href="https://github.com/vllm-project/vllm/pull/59894">#59894</a>): +19.9% at 32K new tokens</summary>

SWA(슬라이딩 윈도우 어텐션) 제한된 리플레이는 이미 레이어 21~39가 마지막 128개 토큰만 처리하도록 제한하고 있었습니다. 이전에는 레이어 20이 여전히 모든 새 토큰에 대해 인덱서 연산을 수행했습니다. 이제 레이어 20은 모든 새 토큰의 KV를 기록하되, 나머지 연산은 128토큰 리플레이 배치에서만 실행합니다. 이 변경으로 **32K 새 토큰에서 프리필 처리량이 19.9%** 향상되었고, 향상 폭은 4K에서 10.1%, 64K에서 23.1%로 커집니다.
> SWA bounded replay already limits layers 21–39 to the last 128 tokens. Previously, layer 20 still performed indexer computation across all new tokens. Now it writes KV for all new tokens but runs its remaining computation only on the 128-token replay batch. This improves prefill throughput by **19.9% at 32K new tokens**, with gains increasing from 10.1% at 4K to 23.1% at 64K.

</details>

<details markdown="1">
<summary><strong>공유 Engram 호스트 테이블에 2 MB 페이지 사용</strong> (<a href="https://github.com/vllm-project/vllm/pull/62359">#62359</a>): 32K에서 추가로 +2.8% / <strong>Use 2 MB pages for shared Engram host tables</strong> (<a href="https://github.com/vllm-project/vllm/pull/62359">#62359</a>): another +2.8% at 32K</summary>

Engram은 DP 랭크들이 공유하는 호스트 테이블에서 무작위 읽기를 수행합니다. 이전의 `/dev/shm` 매핑은 64 KB GPU 페이지를 사용해서 GPU TLB 미스가 자주 일어났습니다. 테이블을 호스트 NUMA cuMem 할당으로 옮기면 2 MB GPU 페이지를 쓸 수 있게 되어, **32K 새 토큰에서 프리필 처리량이 2.8%** 추가로 향상됩니다.
> Engram performs random reads from host tables shared across DP ranks. The previous `/dev/shm` mapping used 64 KB GPU pages, causing frequent GPU TLB misses. Moving the tables to host-NUMA cuMem allocations enables 2 MB GPU pages, improving prefill throughput by another **2.8% at 32K new tokens**.

</details>

<details markdown="1">
<summary><strong>프리필 인덱서에 DeepSelect top-k 사용</strong> (<a href="https://github.com/vllm-project/vllm/pull/62240">#62240</a>): 32K에서 추가로 +15.6% / <strong>Use DeepSelect top-k in the prefill indexer</strong> (<a href="https://github.com/vllm-project/vllm/pull/62240">#62240</a>): another +15.6% at 32K</summary>

이제 프리필 인덱서도 디코드 경로와 마찬가지로 DeepSelect를 사용해 후보 블록의 순위를 매깁니다. 순위를 매길 키가 많을수록 이점이 커지며, 프리필 처리량이 **32K 새 토큰에서 15.6%**, 64K에서 15.2% 추가로 향상됩니다.
> The prefill indexer now uses DeepSelect to rank candidate blocks, matching the decode path. Its benefit grows with the number of keys being ranked, improving prefill throughput by another **15.6% at 32K new tokens** and 15.2% at 64K.

</details>

이 변경들을 모두 합치면, 128K 프리픽스가 캐시된 상태에서 4K~64K 새 토큰 구간의 **새 토큰 프리필 처리량이 35.6~43.9%** 향상됩니다. 32K에서 합산 향상 폭은 **42.5%** 이고, 64K에서는 처리량이 **71.2K에서 102.5K tokens/s/GPU로(+43.9%)** 늘어납니다.
> Together, these changes improve **new-token prefill throughput by 35.6–43.9%** across 4K–64K new tokens on top of a 128K cached prefix. At 32K, the combined gain is **42.5%**; at 64K, throughput increases from **71.2K to 102.5K tokens/s/GPU (+43.9%)**.

<iframe class="vllm-embed" src="/assets/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep/dsv41-pd-prefill-ablation.html" title="128K 캐시 적중 이후 DEP2의 프리필 향상 / Prefill gains on DEP2 after a 128K cache hit" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

*그림 4: 128K 캐시 적중(cache hit) 이후 DEP2에서 측정한 GPU당 새 토큰 프리필 처리량. 레이어 20 리플레이, Engram 2 MB 페이지, DeepSelect top-k를 하나씩 추가했습니다. / Figure 4: New-token prefill throughput per GPU on DEP2 after a 128K cache hit, adding layer 20 replay, Engram 2 MB pages and DeepSelect top-k one at a time.*

### 디코드: Wide-EP 확장과 EPLB로 전문가 균형 맞추기 / Decode: scaling Wide-EP and balancing experts with EPLB

디코드 쪽에서 EPLB를 함께 쓰는 Wide-EP는 이미 자리 잡은 방식으로, [DeepSeek의 추론 시스템 개요](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)에 설명되어 있습니다. vLLM은 두 기능을 한동안 지원해 왔고([대규모 서빙 글](https://vllm.ai/blog/2025-12-17-large-scale-serving#wide-ep) 참고), 이번에도 두 기능이 여전히 잘 작동한다는 것을 확인했습니다. 이 섹션에서는 새로 발견한 점과 경험에서 얻은 교훈에 집중합니다.
> On the decode side, Wide-EP with EPLB is an established practice, described in [DeepSeek's inference system overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md). vLLM has supported both for a while (see our [large-scale serving post](https://vllm.ai/blog/2025-12-17-large-scale-serving#wide-ep)), and we found that they still work well. This section focuses on our new findings and empirical lessons.

#### Wide-EP의 확장성 / The scaling of Wide-EP

128K 컨텍스트에서 GPU당 요청 64개로 DEP4부터 DEP32까지 디코드 확장성을 측정했습니다(EPLB 활성화, 다음 섹션 참고). 각 랭크가 담당하는 전문가 수가 줄어들면서 타깃 MegaMoE 시간은 13.4 ms에서 6.0 ms로 줄어들지만, 나머지 작업은 약 10 ms 수준을 유지합니다. 그 결과 GPU당 처리량의 향상 폭은 점차 줄어듭니다. DEP32에서는 드래프트의 통신 오버헤드도 더 넓은 병렬화의 이점을 상쇄하기 시작합니다.
> We measured decode scaling from DEP4 to DEP32 with 64 requests per GPU at 128K context (with EPLB enabled; see the next section). Target MegaMoE time drops from 13.4 to 6.0 ms as each rank holds fewer experts, while the remaining work stays around 10 ms. As a result, per-GPU throughput gains gradually diminish. At DEP32, the draft's communication overhead also starts to offset the benefits of wider parallelism.

<iframe class="vllm-embed" src="/assets/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep/dsv41-pd-wide-ep-scaling.html" title="GB200에서 DEP4부터 DEP32까지의 디코드 스텝 시간 분석 / Decode step time breakdown from DEP4 to DEP32 on GB200" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

*그림 5: NVIDIA GB200에서 128K 컨텍스트, GPU당 요청 64개 조건으로 측정한 DEP4부터 DEP32까지의 랭크당 디코드 스텝 시간과 GPU당 출력 처리량. / Figure 5: Decode step time per rank from DEP4 to DEP32 on NVIDIA GB200 with 64 requests per GPU at 128K context, and output throughput per GPU.*

#### EPLB

EPLB가 직전 1,000 스텝의 부하를 바탕으로 3,000 스텝마다 전문가를 재배치하도록 했습니다. EPLB는 균형도를 타깃에서 0.35에서 0.94로, 드래프트에서 0.21에서 0.89로 끌어올려, 균등 라우팅과의 차이를 각각 약 97%와 93% 회복합니다.
> We let EPLB rearrange the experts every 3,000 steps, based on the load of the previous 1,000 steps. EPLB raises balancedness from 0.35 to 0.94 for the target and from 0.21 to 0.89 for the draft, recovering about 97% and 93% of the gap to uniform routing.

<iframe class="vllm-embed" src="/assets/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep/dsv41-pd-eplb.html" title="Wide EP 환경의 EPLB: 균형, 스텝 시간, 처리량 / EPLB at wide EP: balance, step time and throughput" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

*그림 6: GB300 NVL72에서 EPLB를 끈 경우, 타깃 모델에만 EPLB를 적용한 경우, 타깃 모델과 DSpark 드래프트 모두에 적용한 경우의 디코드 결과. 탭에서 스텝별 균형, DEP32의 스텝별 시간 분석, 처리량(3회 실행의 평균, 막대는 범위)을 볼 수 있습니다. / Figure 6: Decode on GB300 NVL72 with EPLB off, EPLB for the target model only, and EPLB for both the target and the DSpark draft. The tabs show per-step balance, the per-step time breakdown at DEP32, and throughput (mean of three runs; bars show the range).*

두 가지 변경 덕분에 이 방식이 동작합니다. [#62009](https://github.com/vllm-project/vllm/pull/62009)는 NVLink 기반 `symm_mem` 백엔드를 도입합니다. 이 백엔드는 디코드를 멈추지 않고 DEP32 전체를 1초 이내에 재조정합니다. 또한 EPLB의 복제본(replica) 매핑과 부하 기록을 MegaMoE 커널에 융합해 스텝당 약 0.2 ms를 절약하고, 디코드 루프에서 호스트 동기화를 없앱니다. [#61395](https://github.com/vllm-project/vllm/pull/61395)는 서로 다른(heterogeneous) 타깃 모델과 드래프트 모델에서 EPLB를 지원합니다. DSpark 드래프트는 라우팅되는 전문가(routed expert)가 128개이고 top-3 라우팅을 쓰는 반면, 타깃은 384개에 top-6 라우팅을 씁니다. 그래서 EPLB는 각 모델의 전문가 배치와 부하를 따로 추적하고 둘 다 재조정합니다.
> Two changes make this work. [#62009](https://github.com/vllm-project/vllm/pull/62009) introduces an NVLink-based `symm_mem` backend that rebalances all of DEP32 in under a second without stopping decode, fuses EPLB's replica mapping and load recording into the MegaMoE kernels (saving about 0.2 ms per step), and removes host synchronization from the decode loop. [#61395](https://github.com/vllm-project/vllm/pull/61395) supports EPLB with heterogeneous target and draft models. The DSpark draft has 128 routed experts with top-3 routing, against the target's 384 with top-6, so EPLB tracks expert placement and load separately for each model and rebalances both.

### 빠른 대규모 디코드에서 CPU 오버헤드 극복하기 / Overcome CPU overhead in fast large-scale decode

Wide-EP 디코드는 레이어마다 모든 랭크를 동기화하므로, 랭크 하나만 느려도 인스턴스 전체가 멈춥니다. 비동기 스케줄링(async scheduling)은 작은 규모에서는 CPU 오버헤드를 잘 숨기지만, 랭크가 많아지면 두 가지 이유로 어려워집니다. GPU 스텝이 짧아져 CPU 작업에 쓸 여유가 줄어들고, 어느 한 랭크가 뒤처질 가능성도 커집니다.
> Wide-EP decode synchronizes all ranks at every layer, so one slow rank stalls the instance. Async scheduling hides CPU overhead well at a small scale, but more ranks make this harder in two ways: GPU steps get shorter, which leaves less slack for CPU work, and it becomes more likely that one rank falls behind.

실제로 최적화하지 않은 DEP32는 GPU 실행이 1.1 ms 더 빨랐는데도 DEP16보다 느렸습니다(스텝당 17.57 ms 대 13.84 ms). 아래 표는 찾아낸 모든 원인을 효과가 큰 순서대로 정리한 것입니다. 효과는 서로 다른 구성에서 측정했으므로, 각 행에 해당 구성을 함께 적었습니다.
> In practice, we observed that an unoptimized DEP32 ran slower than DEP16 (17.57 vs. 13.84 ms per step), although its GPU execution was 1.1 ms faster. The table below lists every cause we found, ordered by effect. The effects come from different setups, so each row gives its own.

<details markdown="1">
<summary>원인, 해결책, 효과 보기 / Show the causes, fixes and their effects</summary>

| 원인 / Cause | 해결책 / Fix | PR / PRs | 효과 / Effect | 구성 / Setup |
|---|---|---|---|---|
| 디코드 워커 스레드에서 KV 읽기를 동기적으로 요청 / KV read posted synchronously on the decode worker thread | UCX scatter-gather를 쓰는 NIXL push, 패킹된 KV 행을 하나의 NIXL 영역으로 전송 / NIXL push with UCX scatter-gather; packed KV rows sent as one NIXL region | [NIXL push connector](https://vllm.ai/blog/2026-10-14-nixl-push-connector), [#61121](https://github.com/vllm-project/vllm/pull/61121) | GPU당 141K → 440K tokens/s, 사용자당 P90 ~40 → 159 tokens/s / 141K → 440K tokens/s per GPU; P90 ~40 → 159 tokens/s per user | GB300, 8P1D, 동시성 3,072 / GB300, 8P1D, concurrency 3,072 |
| Gloo all-reduce 지연 시간이 랭크 수에 비례해 선형으로 증가 / Gloo all-reduce latency grows linearly with the rank count | 재귀적 배가(recursive-doubling) CPU all-reduce / Recursive-doubling CPU all-reduce | [#61767](https://github.com/vllm-project/vllm/pull/61767) ([#60267](https://github.com/vllm-project/vllm/pull/60267) 기반 / building on #60267) | DP 동기화 스텝당 10.8 → 1.5 ms, 스텝 16.10 → 13.15 ms / DP sync 10.8 → 1.5 ms per step; step 16.10 → 13.15 ms | DEP32 디코드 / DEP32 decode |
| 요청 수용(admission) 폭주: 한 스텝에 최대 140개 요청 수용 / Admission bursts: up to 140 requests admitted in one step | 스텝당 새 KV 로드를 4개로 제한하는 옵트인(opt-in) 상한 / Opt-in cap of 4 new KV loads per step | [#61725](https://github.com/vllm-project/vllm/pull/61725) | GPU당 460K → 516K tokens/s, 사용자당 P90 109 → 153 tokens/s / 460K → 516K tokens/s per GPU; P90 109 → 153 tokens/s per user | GB300, 7P1D, 동시성 4,800, 다른 CPU 수정 두 가지와 함께 / GB300, 7P1D, concurrency 4,800, with two other CPU fixes |
| 프롬프트 길이에 따라 커지는 요청 수용 작업 / Admission work that grows with the prompt length | 프롬프트를 하나의 int32 버퍼로 처리, 슬라이딩 윈도우 블록 테이블의 null 블록 건너뛰기, 청크 단위 프롬프트 복사 커널 / Prompt as one int32 buffer; skip the null blocks of sliding-window block tables; chunked prompt copy kernel | [#59908](https://github.com/vllm-project/vllm/pull/59908), [#62702](https://github.com/vllm-project/vllm/pull/62702), [#61724](https://github.com/vllm-project/vllm/pull/61724), [#62698](https://github.com/vllm-project/vllm/pull/62698) | 855K 프롬프트: 요청 수용당 CPU 시간 38.8 → 6.0 ms / 855K prompt: 38.8 → 6.0 ms of CPU per admission | 엔진 마이크로벤치마크 / Engine microbenchmark |

</details>

### 낮은 지연 시간: 작은 배치를 위한 특화된 디코드 경로 / Low latency: a specialized decode path for small batches

또한 DeepSeek-V4.1-Flash의 저지연 커널 지원을 확장했습니다([#61868](https://github.com/vllm-project/vllm/pull/61868), [#61869](https://github.com/vllm-project/vllm/pull/61869)). 이는 [이전 글](https://vllm.ai/blog/2026-10-16-kimi-k3-megakernel)에서 이어지는 작업입니다. 실제 AgentX 세션에서 특화된 디코드 경로는 B300 GPU 4개로 단일 사용자 기준 P90 826 tokens/s에 도달했으며, 이는 표준 디코드 경로(575)보다 44% 높습니다.
> We also extended low-latency kernel support for DeepSeek-V4.1-Flash ([#61868](https://github.com/vllm-project/vllm/pull/61868) and [#61869](https://github.com/vllm-project/vllm/pull/61869)), continuing the efforts from our [previous post](https://vllm.ai/blog/2026-10-16-kimi-k3-megakernel). On real AgentX sessions, the specialized decode path reaches 826 P90 tokens/s for a single user on four B300 GPUs, 44% higher than the standard decode path (575).

<iframe class="vllm-embed" src="/assets/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep/dsv41-pd-low-latency.html" title="B300에서 TP4, 동시성 1, 8, 16일 때 SemiAnalysis AgentX의 P90 상호작용성, InferenceX #4332 적용 전후 / P90 interactivity on SemiAnalysis AgentX with TP4 on B300 at concurrency 1, 8 and 16, before and after InferenceX #4332" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

*그림 7: B300 GPU 4개에서 TP4, 동시성 1, 8, 16일 때의 SemiAnalysis AgentX 결과. [InferenceX #4332](https://github.com/SemiAnalysisAI/InferenceX/pull/4332)가 특화된 디코드 경로로 전환하기 전과 후를 비교했습니다. / Figure 7: SemiAnalysis AgentX with TP4 on four B300 GPUs at concurrency 1, 8 and 16, before and after [InferenceX #4332](https://github.com/SemiAnalysisAI/InferenceX/pull/4332) switched to the specialized decode path.*

## 감사의 글 / Acknowledgements

DeepSeek-V4.1-Flash와 최적화된 커널을 오픈소스로 공개하고 공유해 이 작업의 토대를 마련해 준 **DeepSeek** 에 감사드립니다. 또한 DeepInfer와 그 성능 결과를 공개해, 배우고 비교할 수 있는 든든한 기준을 제공해 준 점에도 감사드립니다. AgentX 벤치마크 플랫폼과 여러 도움을 준 **SemiAnalysis** 에도 감사드립니다.
> We would like to thank **DeepSeek** for open-sourcing and sharing DeepSeek-V4.1-Flash and its optimized kernels, which provided the foundation for this work, and for publishing DeepInfer and its performance results, which gave us a strong reference to learn and compare against. We also thank **SemiAnalysis** for the AgentX benchmark platform and all the help.

성능 최적화에서 긴밀하게 협업하고, vLLM과 Dynamo 전반에서 꾸준히 지원하고, AgentX 제출을 도와준 **NVIDIA** 의 **Pei Li, Hanjie Qiu, Lisa Wang, Yaoyao Ding, Rohit Nagraj, Xin Li** 에게 특히 깊이 감사드립니다. 귀중한 프리필 최적화를 기여한 **vLLM 커뮤니티** 의 **Canlin Guo** 에게도 감사드립니다.
> We are especially grateful to **Pei Li, Hanjie Qiu, Lisa Wang, Yaoyao Ding, Rohit Nagraj, and Xin Li** from **NVIDIA** for their close collaboration on performance optimization, continued support across vLLM and Dynamo, and assistance with the AgentX submissions. We also thank **Canlin Guo** from the **vLLM community** for his valuable prefill optimization.

이 작업은 **Inferact** 가 주도했습니다.
> This work was led by **Inferact**.

<script>
(function () {
  // 임베드된 인터랙티브 그림의 높이를 내용 높이에 맞춥니다(그림이 iframe 안에 있을 때 자신의 높이를 전달합니다).
  window.addEventListener('message', function (event) {
    var d = event.data;
    if (!d || d.type !== 'vllm-embed-resize' || typeof d.height !== 'number') return;
    var frames = document.querySelectorAll('iframe.vllm-embed');
    for (var i = 0; i < frames.length; i++) {
      if (frames[i].contentWindow === event.source) {
        frames[i].style.height = Math.ceil(d.height) + 'px';
      }
    }
  });
})();
</script>
