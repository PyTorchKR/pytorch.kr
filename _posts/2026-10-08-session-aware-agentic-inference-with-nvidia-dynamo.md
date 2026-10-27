---
layout: blog_detail
title: "NVIDIA Dynamo로 구현하는 세션 인식 에이전틱 추론"
author: Ishan Dhanani, Jamie Li, Karen Chung
ext_author: Junghwan Park (박정환)
category: ["pytorch.org", "translation"]
date: 2026-10-08 12:00:00
org_title: "Session-Aware Agentic Inference with NVIDIA Dynamo"
org_link: https://pytorch.org/blog/session-aware-agentic-inference-with-nvidia-dynamo/
---

![NVIDIA Dynamo로 구현하는 세션 인식 에이전틱 추론 대표 이미지 / Session-Aware Agentic Inference with NVIDIA Dynamo](/assets/blog/2026-10-08-session-aware-agentic-inference-with-nvidia-dynamo/hero.png){:style="width:100%"}

### 요약 / TL;DR

에이전틱(agentic) 워크로드는 추론 서버가 받는 트래픽의 모양을 바꿉니다. 한 번 묻고 답하는 채팅과 달리, 에이전트 세션에는 큰 초기 프리필(prefill), 반복되는 모델 호출, 병렬로 실행되는 서브에이전트(subagent)가 포함될 수 있습니다. 그리고 턴과 턴 사이에 도구가 실행되는 동안에도 KV 캐시는 메모리에 그대로 남아 있습니다.
> Agentic workloads change the traffic an inference server sees. Unlike single-turn chat, an agent session can involve a large initial prefill, repeated model calls, and subagents running in parallel, with KV cache remaining resident while tools execute between turns.

이번 글에서는 NVIDIA Dynamo가 세션 단위의 통합 식별자 하나로 요청 단위 서빙 인프라를 프로그램 인식(program-aware) 시스템으로 바꾸는 방법을 자세히 설명합니다. 이 식별자 덕분에 vLLM과 SGLang 전반에서 세션 인식 라우팅(session-aware routing), 공유 KV 캐시 인덱싱, 프로그래밍 방식의 캐시 이동이 가능해집니다.
> This technical blog post details how NVIDIA Dynamo uses a unified, session level identifier to transform request-level serving infrastructure into a program-aware system, unlocking session-aware routing, shared KV cache indexing, and programmatic cache movement across vLLM and SGLang.

## 소개 / Introduction

에이전틱 워크로드는 추론 서버가 받는 트래픽의 모양을 바꿔 놓았습니다. 코딩 세션은 큰 프리필로 시작합니다. 시스템 프롬프트, 도구 정의, 사용자별 지침이 수만 토큰에 이르는 경우가 많고, 그 뒤의 모든 턴은 세션이 매번 통째로 다시 보내는 컨텍스트에 내용을 덧붙입니다. 작업 하나는 이런 호출 수십 개로 퍼져 나가고, 그 옆에서는 수명이 짧거나 긴 서브에이전트가 병렬로 실행됩니다. 세션의 실제 경과 시간(wall-clock time) 대부분은 도구 호출 사이에 흘러가며, 그동안 아무것도 생성하지 않는데도 컨텍스트는 KV 캐시에 그대로 남아 있습니다. 이런 워크로드를 잘 서빙하는 일은 결국 두 가지로 귀결됩니다. 시스템이 그 컨텍스트를 얼마나 많이 메모리에 유지하고 재사용하는지, 그리고 지연 시간 목표를 지키면서 동시에 몇 개의 세션을 감당하는지입니다. 그림 1은 에이전틱 워크로드와 일반 챗봇 워크로드의 이런 차이를 보여줍니다.
> Agentic workloads have changed the shape of the traffic an inference server sees. A coding session opens with a large prefill, often tens of thousands of tokens of system prompt, tool definitions, and user-specific guidance, and every turn after that appends to a context the session resends in full. One task fans out into dozens of such calls, alongside short and long lived subagents running in parallel. Most of a session’s wall-clock time passes between tool calls, with its context sitting resident in the KV cache while nothing generates. Serving these well comes down to how much of that context a system keeps resident and reuses, and how many sessions it holds at once while still meeting latency targets This contrast between agentic workloads and that of a standard chatbot is shown in Figure one.

![일반 챗봇과 에이전틱 워크플로우의 턴 비교 / Standard chatbot turns compared with an agentic workflow](/assets/blog/2026-10-08-session-aware-agentic-inference-with-nvidia-dynamo/Agentic-Model-Chain-GIF-2-1.gif){:style="width:100%"}
*그림 1: 사용자 턴과 모델 턴이 번갈아 오가는 일반 챗봇과, 도구 호출과 도구 응답이 포함된 에이전틱 워크플로우를 비교합니다. 에이전틱 워크플로우에서는 사용자 요청 하나가 여러 번의 추론 호출로 이어질 수 있고, 다음 단계는 중간 결과에 따라 달라집니다. / Figure 1: Compares the alternating user and model turns of a standard chatbot with an agentic workflow that includes tool calls and tool responses. In the agentic workflow, a single user request can trigger multiple inference calls, with subsequent steps depending on intermediate results.*

대부분의 오픈소스 서빙 스택은 여전히 그림 위쪽의 모델에 맞춰 만들어져 있습니다. 요청 하나하나를 따로 라우팅하고, 수락하고(admit), 캐싱하며, 그 요청을 이전 턴과 이어 주는 하네스(harness)나 세션이라는 개념이 없습니다.
> Most open-source serving stacks are still built for the model on top of the figure. They route, admit, and cache each request on its own, with no notion of the harness or the session that ties it to previous turns.

이런 형태의 워크로드를 최적으로 서빙하는 것은 저희가 꾸준히 목표로 삼아 온 일입니다. 이전 글인 [Full-Stack Optimizations for Agentic Inference](https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/)에서는 NVIDIA Dynamo 스택의 세 계층, 즉 라우터, 추론 엔진, KV 캐시 관리자에 걸친 작업을 소개했습니다. 지난 몇 달 동안 이 작업을 하나의 기본 요소(primitive), 즉 공통 세션 단위 식별자를 중심으로 통합했습니다. 세션 단위 ID(논문과 학계에서는 흔히 프로그램(program) 또는 궤적(trajectory)이라고 부릅니다)는 요청을 인식하는 인프라를 프로그램을 인식하는 인프라로 바꾸며, 세션 인식 최적화를 더해 나갈 기반이 됩니다.
> Serving this shape optimally is something we have been building towards. In our previous post, [Full-Stack Optimizations for Agentic Inference](https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/), we shared work across three layers of the NVIDIA Dynamo stack: the router, the inference engine, and the KV cache manager. Over the last few months we’ve unified this work around a single primitive: a common session-level identifier. Session-level ID (often called program or trajectory in papers and academia) turns request-aware infrastructure into program-aware infrastructure, and serves as the foundation for additional session-aware optimizations.

이번 글에서는 그동안의 진척을 네 부분으로 나누어 다룹니다:

1. **Session\_ID:** Dynamo가 기존 하네스 헤더에서 세션과 서브에이전트를 알아보는 방법, 그리고 커스텀 하네스가 같은 정보를 제공하는 방법.
2. **트레이싱과 재생(Tracing and replay):** 세션에 연결된 트레이스가 하네스 활동을 추론 성능과 이어 주고, 오프라인 시뮬레이션과 실시간 벤치마킹을 지원하는 방법.
3. **세션 인식 스케줄링(Session-aware scheduling):** 라우팅과 수락 제어(admission control)가 세션의 워킹 셋(working set)을 고려하고, 도구 경계(tool boundary)에서 배압(backpressure)을 걸어 캐시 스래싱(cache thrashing)을 줄이는 방법.
4. **공유 KV 캐시 인식(Shared KV cache awareness):** 실험적인 공유 풀 인덱서(shared-pool indexer)로 라우터가 요청을 배치할 때 외부 저장소에 있는 재사용 가능한 KV를 고려하는 방법.
5. **프로그래밍 방식 KV 캐시(Programmatic KV cache):** 세션-프리픽스 인덱서(Session-Prefix Indexer)와 제안된 KvHint 인터페이스가 여러 메모리 계층에 걸친 세션 인식 캐시 정책을 지원하는 방법. 이때 Dynamo는 정책의 의도(intent)를 제공하고, 추론 엔진은 실행을 제어합니다.

> This post covers that progress in four parts:
>
> 1. **Session\_ID:** How Dynamo recognizes sessions and subagents from existing harness headers, and how custom harnesses can provide the same information.
> 2. **Tracing and replay:** How session-linked traces connect harness activity to inference performance and support offline simulation and live benchmarking.
> 3. **Session-aware scheduling:** How routing and admission control account for session working sets and apply backpressure at tool boundaries to reduce cache thrashing.
> 4. **Shared KV cache awareness:** How the experimental shared-pool indexer lets the router account for reusable KV in external stores when placing requests.
> 5. **Programmatic KV cache:** How the Session-Prefix Indexer and proposed KvHint interface support session-aware cache policies across memory tiers, with Dynamo supplying policy intent and inference engines controlling execution.

## 세션 ID로 에이전트 세션 식별하기 / Identify Agent Sessions with Session ID

Dynamo 세션 ID는 에이전트의 추론과 도구 호출로 이어지는 하나의 흐름(chain)을 가리키는, 고유하고 변하지 않는 식별자입니다. 하나의 궤적에 속한 모든 LLM 요청은 같은 `session_id` 를 공유합니다. 보통 자식 세션은 `parent_session_id` 를 함께 가지므로, 트레이스와 재생 도구는 에이전트와 그 에이전트가 만든 서브에이전트나 하위 작업(subtask)을 서로 연결할 수 있습니다. 이번 글에서 다루는 모든 최적화는 세션 ID와 부모 세션 ID에 담긴 세션 정보를 사용하며, 각각 선택해서 켜는(opt-in) 방식입니다.
> A Dynamo session ID is a unique stable identifier for a single agentic chain of reasoning and tool calls. Every LLM request in a given trajectory shares the same `session_id`. Usually, child sessions carry a `parent_session_id`, so traces and replay tools can establish a connection between agents and subagents or subtasks they spawn. Every optimization in this post is an opt-in consumer of the session information contained in session ID and parent session ID

Dynamo를 호출하는 쪽이 무엇이냐에 따라, 변하지 않는 세션 식별자를 Dynamo에 전달하는 방법은 몇 가지가 있습니다.
> There are a few different ways to share a stable session identifier with Dynamo, depending on what’s calling it.

### 코딩 에이전트 기본 지원 / Out-of-the-box support for coding agents

Dynamo는 오늘날 널리 쓰이는 코딩 에이전트가 이미 보내고 있는 식별 헤더를 알아보고, 각 헤더를 같은 내부 `agent_context` 에 매핑합니다. 세션 헤더는 `session_id` 가 되고, 부모 헤더가 있으면 `parent_session_id` 가 됩니다. Claude Code, Codex, OpenCode는 별도 설정 없이 동작합니다.
> Dynamo recognizes the identity headers that today’s popular coding agents already emit and maps each onto the same internal `agent_context`: the session header becomes `session_id`, and the parent header, where present, becomes `parent_session_id`. Claude Code, Codex, and OpenCode work without configuration.

| 코딩 에이전트 / Coding agent | 세션 헤더 / Session header | 부모 헤더 / Parent header |
| --- | --- | --- |
| Claude Code | `x-claude-code-session-id` (루트 / root), `x-claude-code-agent-id` (자식 에이전트 / child agents) | `x-claude-code-parent-agent-id` |
| Codex | `thread-id` | `x-codex-parent-thread-id` |
| OpenCode | `x-opencode-session-id` | `x-opencode-parent-session-id` |

생태계 전반의 하네스 제공자와 협력해 이 방식을 어디서나 기본값으로 만들고 있으므로, 사용자가 아무것도 바꾸지 않아도 지원 범위가 넓어질 것입니다. Dynamo의 CI는 정식 지원하는 하네스가 바뀔 때마다 이를 추적합니다. 정식 지원하는 ID의 전체 목록은 [여기](https://docs.nvidia.com/dynamo/dev/agents/session-i-ds)에서 볼 수 있습니다.
> We’re working with harness providers across the ecosystem to make this the default everywhere, so support expands without users changing anything. Dynamo’s CI tracks each first-class harness as it changes. You can find a full list of supported first class IDs [here](https://docs.nvidia.com/dynamo/dev/agents/session-i-ds).

### 세션 식별 정보를 직접 보내지 않는 하네스를 위한 플러그인 / Plugins for harnesses that don’t send session identity natively

호환되는 헤더를 스스로 보내지 않는 하네스를 위해 [agent-plugins 저장소](https://github.com/ai-dynamo/agent-plugins)가 작은 통합 모듈을 제공합니다. 이 모듈은 하네스 고유의 세션 식별 정보를 `x-dynamo-session-id` 로 변환합니다. 현재 세 가지를 쓸 수 있습니다. Dynamo의 OpenAI 호환 엔드포인트를 위한 Pi 프로바이더, Hermes의 `session_id` 를 매핑하는 Hermes 미들웨어, OpenClaw의 `sessionId` 를 매핑하는 OpenClaw 프로바이더입니다. 각 플러그인은 저장소 안의 자체 폴더에 설치, 설정, 검증 절차를 담고 있습니다.
> For harnesses that don’t emit a compatible header on their own, the [agent-plugins repo](https://github.com/ai-dynamo/agent-plugins) provides small integrations that translate a harness’s native session identity into `x-dynamo-session-id`. Three are available today: a Pi provider for Dynamo’s OpenAI-compatible endpoint, a Hermes middleware that maps Hermes’s `session_id`, and an OpenClaw provider that maps OpenClaw’s `sessionId`. Each plugin ships its own install, configuration, and validation steps in its own folder in the repo.

### 커스텀 하네스: 헤더 하나로 선택해서 켜기 / Custom harnesses: opt in with one header

커스텀 하네스는 표준 헤더 하나, `X-Dynamo-Session-ID` 로 이 기능을 켭니다. Dynamo는 이 값을 스택의 나머지 부분이 읽는 내부 `agent_context` 구조체로 정규화하므로, 하위 단계에서는 요청이 어느 하네스에서 왔는지 알 필요가 없습니다. 하네스가 서브에이전트를 지원한다면 `X-Dynamo-Parent-Session-ID` 도 함께 보낼 수 있습니다. 아래 코드 블록은 어떤 커스텀 에이전트든 독자적인(proprietary) API를 도입하지 않고도 Dynamo의 최적화를 활용하는 방법을 보여줍니다.
> A custom harness opts in with one canonical header, `X-Dynamo-Session-ID`. Dynamo normalizes it into an internal `agent_context` struct that the rest of the stack reads, so nothing downstream needs to know which harness the request came from. If the harness supports subagents, you can also provide `X-Dynamo-Parent-Session-ID`. The following code block shows how any custom agent can pick up Dynamo’s optimizations without adopting a proprietary API.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-dummy' \
  -H 'x-dynamo-session-id: research-run-42:researcher' \
  -d '{"model":"my-model","messages":[{"role":"user","content":"..."}]}'
```

## 하네스 ↔ 추론 공동 설계 / Harness ↔ Inference Codesign

에이전트 관측(observability) 도구는 이미 하네스 수준에서 일어나는 일을 기록할 수 있지만, 각 단계 아래에서 추론 스택이 어떻게 동작했는지를 하나로 합친 그림은 아직 보여주지 못합니다. 저희는 이 간극을 하네스 ↔ 추론 공동 설계(codesign)라는 관점에서 봅니다. 같은 요청에 대해 양쪽이 공유하고 측정할 수 있는 트레이스를 주고, 이를 성능 시각화, 재생, 오프라인 시뮬레이션에 쓰며, 이 모든 것을 변하지 않는 세션 ID를 키로 묶는 것입니다. 이 트레이스는 가볍게 설계했습니다. Harbor의 ATIF나 Letta의 새 Trajectory 형식 같은 다른 트레이스 수집 형식과 함께, 선택적인 추가 메타데이터로 쓸 수 있습니다.
> Agent observability tools can already capture what happens at the harness level, but they don’t yet show a unified picture of how the inference stack performed underneath each of those steps. We think about this gap as harness ↔ inference codesign: giving both sides a shared, measurable trace of the same request that can be used for performance visualization, replay, and offline simulation, all keyed by a constant session ID. This trace is meant to be lightweight. It can be used as additional optional metadata alongside other trace collection formats, like Harbor’s ATIF or Letta’s new Trajectory format.

### 요청 트레이스 수집하기 / Capturing request traces

트레이스 수집을 켜려면 `DYN_REQUEST_TRACE=1` 로 배포를 실행합니다. Dynamo는 각 요청의 정보를 자동으로 기록하며, 기본적으로 프롬프트, 응답, 도구 호출 내용은 저장하지 않습니다(이 항목들은 선택해서 켤 수 있습니다). 그 대신 응답 스트림이 끝날 때마다 `request_end` 레코드를 내보냅니다. 이 레코드에는 세션 필드, 출력 토큰 수, 종료 사유(finish reason), KV 캐시 지표, 도구 호출 이름, 그리고 요청의 입력 길이와 블록별 시퀀스 해시를 담은 재생 블록이 들어 있으며, Mooncake 트레이스 형식과 비슷합니다. 시퀀스 해시는 내장 `dynamo.replay` 나 저희의 부하 생성 도구인 AIPerf 같은 도구로 재생하는 데 쓸 수 있습니다.
> To enable trace collection, run your deployment with `DYN_REQUEST_TRACE=1`. Dynamo automatically records information on each request, without storing prompt, response, or tool-call content by default (these can be opt-in enabled). Instead, Dynamo emits a `request_end` record after each response stream finishes, carrying the session fields, the output token count, the finish reason, KV cache metrics, tool-call names, and a replay block with the request’s input length and per-block sequence hashes, similar to the Mooncake trace format. The sequence hashes can be used for replay with tools like the built-in `dynamo.replay` or our load-gen tool, AIPerf.

![에이전트와 도구 호출 트랙이 나뉜 Dynamo 트레이스 화면 / Dynamo trace view with separate tracks for agents and tool calls](/assets/blog/2026-10-08-session-aware-agentic-inference-with-nvidia-dynamo/Dynosym-trace-view.png){:style="width:100%"}
*그림 2: Dynamo 트레이스의 화면 캡처로, 에이전트와 그 도구 호출이 각각 별도 트랙에 표시됩니다. 프리필, 디코드, 도구 실행이 하나의 타임라인에 나타나므로, 세션 전체에서 작업이 어떤 순서로 진행되고 어떻게 겹치는지 볼 수 있습니다. / Figure 2: Shows a screen capture of the dynamo trace, with separate tracks for agents and their tool calls. Prefill, decode, and tool execution appear on the same timeline, showing how work is sequenced and overlaps across the session.*

모든 레코드에 타임스탬프와 종료 사유가 들어 있으므로, 도구 호출의 분포를 보고 이를 KV 캐시 축출(eviction) 동작과 연결해 볼 수 있습니다. 내부적으로는 이 기법을 써서, 수집한 트레이스를 백테스트(backtesting) 데이터로 삼아 모델 배포를 엔드투엔드(end-to-end)로 최적화해 왔습니다. 이 데이터와 재생 도구의 상당 부분은 내부 자동 연구(autoresearch) 하네스에도 쓰입니다. 이 하네스로 AI가 새로운 라우팅 전략을 만들어 내고, 아래에서 다룰 선제적 KV 캐시 오프로딩(offloading) 전략을 연구합니다.
> Because every record carries timestamps and finish reasons, we can see tool-call distributions and connect them to KV cache eviction behavior. Internally, we’ve used these techniques to optimize our model deployments end to end, using captured traces as backtesting data. Much of this data and these replay tools also power our internal autoresearch harnesses, which let AI generate novel routing strategies and research proactive KV cache offloading strategies, which we cover below.

### 에이전틱 트레이스 재생하기 / Replaying agentic traces

수집한 트레이스는 다시 쓸 수 있는 벤치마크입니다. 트레이스는 에이전트의 결정이 아니라 에이전트가 보낸 워크로드를 기록하므로, 한 번 수집한 트레이스로 같은 요청 일정을 원하는 만큼 재생할 수 있습니다. 이때 모델, 도구, 외부 API를 다시 실행할 필요가 없습니다. 재생도 트레이스와 마찬가지로 내용을 담지 않습니다. 시퀀스 해시는 합성(synthetic) 블록 ID가 되며, 이 ID는 원래 토큰을 드러내지 않으면서도 턴과 세션 사이의 프롬프트 프리픽스 공유 관계를 그대로 보존합니다.
> A captured trace is a reusable benchmark. Because it records the workload the agent sent rather than the agent’s decisions, one capture replays the same request schedule as many times as you want without rerunning the model, the tools, or any external API. Replay stays content-free the same way the trace does: sequence hashes become synthetic block IDs that preserve prompt-prefix sharing across turns and sessions without exposing the original tokens.

수집한 트레이스 하나로 두 가지 경로를 돌릴 수 있습니다. 오프라인에서는 [AISimulate](https://github.com/ai-dynamo/aisimulate)가 Dynamo의 시뮬레이션된 스케줄러, 라우터, KV 캐시를 상대로 요청 그래프를 실행합니다. 덕분에 GPU 시간을 쓰지 않고도 워커 수, 서빙 토폴로지, 라우팅 정책, 캐시 용량을 비교할 수 있습니다. 실시간으로는 AIPerf가 같은 그래프를 실제 GPU 위의 실제 Dynamo 엔드포인트를 상대로 재생해, 최종 지연 시간, 처리량, 캐시 수치를 얻습니다.
> One capture drives two paths. Offline, [AISimulate](https://github.com/ai-dynamo/aisimulate) runs the request graph against Dynamo’s simulated scheduler, router, and KV cache, letting you compare worker counts, serving topologies, routing policies, and cache capacity without spending GPU time. Live, AIPerf replays the same graph against a real Dynamo endpoint on real GPUs for final latency, throughput, and cache numbers.

## 세션 계층의 라우팅과 수락 제어 / Routing and admission control at the session layer

KV 캐시는 효율적인 LLM 추론에서 언제나 중요했지만, 에이전틱 추론은 실질적인 복잡성을 더합니다. 캐시가 어디에 있는지, 메모리 계층마다 얼마나 재사용할 수 있는지, 그리고 시스템이 우선순위 지정, 축출, 공유를 선제적으로 처리하는지가 문제가 됩니다. 결국 처리량은 순수한 연산 능력보다 스택이 이 질문들에 얼마나 잘 답하느냐에 더 크게 좌우됩니다.
> KV cache has always mattered for efficient LLM inference, but agentic inference adds real complexity: where the cache lives, how much of it can be reused at each memory tier, and whether the system is proactive about prioritization, eviction, and sharing. Throughput ends up being determined less by raw compute and more by how well the stack answers those questions.

Dynamo 라우터는 바로 다음 요청 이상의 정보를 바탕으로 동작할 만큼 유연하기 때문에, 이런 특성을 직접 겨냥한 세션 인식 라우팅과 수락 제어 전략을 반복해서 개선할 수 있었습니다. 그 결과 에이전틱 형태의 워크로드(RL 사후 학습(post-training) 포함)의 서빙 성능이 좋아졌습니다. 목표는 KV 캐시를 최대한 재사용하고 스래싱을 피하는 것입니다. 스래싱이란 HBM, 나아가 DRAM까지도 모든 에이전트의 워킹 셋을 한꺼번에 담을 수 없어서 요청마다 축출과 프리필을 반복하는 상황을 말합니다.
> Because the Dynamo router is flexible enough to act on more than just the next request, we’ve been able to iterate on session-aware routing and admission control strategies that target these characteristics directly, improving performance for serving agentic-shaped workloads (including RL post-training). The goal is to reuse as much KV cache as possible and avoid thrashing: evicting and re-prefilling for every request because HBM, and eventually DRAM, can’t hold every agent’s working set at once.

### 요청 단위 라우팅의 문제 / The problem with request-level routing

Dynamo의 기본 `KvRouter` 를 포함해, 요청 단위 라우터는 배치 문제를 한 번에 요청 하나씩 풉니다. 프롬프트가 주어지면 어느 워커가 캐시를 가장 많이 겹쳐 갖고 있는지를 찾는 식입니다. 한 턴에는 맞는 단위지만 에이전트에는 맞지 않는 단위이며, 동시성이 높으면 요청 뒤에 있는 에이전트의 두 가지 특성을 놓칩니다:

- **캐시 점유량 폭증(Cache-occupancy blowup).** 턴과 턴 사이에도 에이전트의 캐시는 해제되지 않습니다. 에이전트가 GPU 밖에서 도구 호출을 실행하는 동안에도 캐시는 HBM에 남아 블록을 차지합니다. 에이전트 N개가 동시에 k번째 단계에 있으면 전체 워킹 셋은 `N × context_k` 인데, 요청 단위 라우터에는 이 총합이라는 개념이 없습니다. 라우터는 언제나 한 턴의 컨텍스트만 봅니다. 워킹 셋이 HBM을, 이어서 DRAM을 넘어서면 엔진은 살아 있는 세션의 캐시를 축출하기 시작하고, 그 뒤의 모든 턴은 캐시 적중(cache hit) 대신 전체 프리필을 다시 치러야 합니다.
- **도구 경계에서의 배압 부재(No tool-boundary backpressure).** 워커가 용량을 넘어섰을 때 요청 단위 라우터가 쓸 수 있는 수단은 진행 중인 요청을 취소하거나 대기열에 넣는 것뿐입니다. 둘 다 다른 방법보다 나쁩니다. 그 방법이란 세션이 다음 턴을 보내기 직전의 멈춤 지점에서 세션을 미뤄 두는 것입니다. 그 시점에는 세션이 어차피 곧 쉬게 됩니다.

> A request-level router, Dynamo’s stock `KvRouter` included, solves placement one request at a time: given a prompt, which worker has the most cache overlap. That’s the right unit for a single turn, but it’s the wrong unit for an agent, and at high concurrency it misses two things about the agent behind the request:
>
> - **Cache-occupancy blowup.** Between turns, an agent’s cache doesn’t get freed. It sits in HBM holding blocks while the agent runs a tool call outside the GPU. With N agents concurrently at step k, the aggregate working set is `N × context_k`, and a request-level router has no notion of that aggregate. It only ever sees one turn’s context at a time. Once the working set exceeds HBM, and then DRAM, the engine starts evicting live sessions’ cache, and every next turn pays a full re-prefill instead of a cache hit.
> - **No tool-boundary backpressure.** When a worker is over capacity, a request-level router’s only levers are to cancel or queue in-flight requests. Both are worse than the alternative: deferring the session at the pause point right before its next turn arrives, when it’s about to be idle anyway.

### 세션 인식 수락 제어 / Session Aware admission control

첫 번째 에이전트 인식 라우팅 전략은 ThunderAgent 논문(Kang et al., 2026)의 스케줄러를 Dynamo의 KV 인식 라우터 위로 옮겨 온 것입니다. Rust로 작성했으며, 추가 프록시 홉(hop)으로 `KvRouter` 앞에 놓이는 대신 `KvRouter` 를 직접 소유합니다. 또한 원시 바이트로 토큰 수를 추정하지 않고, 각 응답에서 실제 `prompt_tokens + completion_tokens` 를 추적합니다. 네이티브 Dynamo 플러그인으로 작성했기 때문에, [소스 코드](https://github.com/ai-dynamo/dynamo/tree/main/lib/router-plugins/builtin/src/thunderagent)를 가져가 Dynamo의 기반 라우터 코드를 바꾸지 않고도 자신의 워크로드 형태에 맞게 고칠 수 있습니다.
> Our first agent-aware routing strategy ports the scheduler from the ThunderAgent paper (Kang et al., 2026) on top of Dynamo’s KV-aware router. Written in Rust, it owns a `KvRouter` directly rather than sitting in front of one as an extra proxy hop, and it tracks real `prompt_tokens + completion_tokens` off each response instead of estimating token counts from raw bytes. Because this is written as a native Dynamo plugin, you can take its [source code](https://github.com/ai-dynamo/dynamo/tree/main/lib/router-plugins/builtin/src/thunderagent) and adapt it to your workload shape without changing Dynamo’s underlying router code.

스케줄러는 요청을 `program_id`(세션 ID)로 묶고, 각 프로그램을 서로 독립적인 두 상태 축 `(REASONING | ACTING) × (ACTIVE | PAUSED)` 사이에서 옮깁니다. 프로그램은 도구 경계에서 `ACTING` 상태가 됩니다. 메모리 압박이 생기면 스케줄러는 `ACTING` 프로그램을 논리적으로 일시 정지합니다. 디코드 선점(preemption)은 없습니다. 진행 중인 턴은 끝까지 실행되며, 그러면 엔진이 해당 KV 블록을 축출할 수 있게 됩니다. 워커에 일시 정지할 ACTING 프로그램이 더 남아 있지 않으면, 대신 가장 작은 REASONING 프로그램을 골라 다음 도구 경계에서 일시 정지하도록 표시합니다. 워커에 다시 여유가 생기면 스케줄러는 토큰이 가장 적은 일시 정지 프로그램부터 재개하며, 임계값 아래에 들어오도록 빈 패킹(bin-packing) 방식으로 다시 채워 넣습니다.
> The scheduler groups requests by `program_id` (the session ID) and moves each program through two independent states: `(REASONING | ACTING) × (ACTIVE | PAUSED)`. A program enters `ACTING` at a tool boundary. Under memory pressure, the scheduler pauses `ACTING` programs logically. There’s no decode preemption; the in-flight turn is allowed to finish, which frees the engine to evict their KV blocks. If a worker has no ACTING programs left to pause, it instead marks the smallest REASONING program for pause at its next tool boundary. When a worker has headroom again, the scheduler resumes the smallest-token paused programs first, bin-packing them back under threshold.

### 제어 루프 / The control loop

일시 정지와 재개는 모두 워커마다 하나의 값, 즉 사용률(utilization)로 결정됩니다. 사용률은 워커의 KV 풀 가운데 그 워커에 할당된 프로그램들의 워킹 셋이 차지하는 비율입니다:
> Pause and resume are both driven off one quantity per worker: utilization, the fraction of a worker’s KV pool occupied by its assigned programs’ working sets:

```text
U_worker = (sum of token weights for ACTING programs on worker) / (worker's KV pool capacity)
```

임계값 세 개(괄호 안은 기본값)가 이 값을 제어 루프로 바꿉니다:

- **U ≥ 0.95 (pause-threshold):** 워커가 초과 할당된 상태입니다. 매 틱(tick)마다 가장 작은 ACTING 프로그램부터 일시 정지해, U가 0.80(pause-target)으로 내려올 때까지 계속합니다.
- **0.80 ≤ U < 0.95 (소프트 강등 구간 / soft-demote band):** 프로그램을 아직 일시 정지하지는 않지만, 강제 일시 정지가 필요해지기 전에 미리 거는 배압으로 -2.0의 우선순위 페널티를 줍니다.
- **재개는 U ≤ 0.85가 되어야 일어납니다:** 이 값은 pause-threshold에서 resume-hysteresis 0.10을 뺀 것입니다. 일시 정지 기준선 아래에 둔 이 여유 폭 덕분에, 루프가 임계값 바로 근처에서 일시 정지와 재개를 오가며 흔들리지(flapping) 않습니다.

> Three thresholds (defaults shown) turn that number into a control loop:
>
> - **U ≥ 0.95 (pause-threshold):** the worker is over-subscribed. The tick pauses the smallest ACTING programs first until U falls back to 0.80 (pause-target).
> - **0.80 ≤ U < 0.95 (soft-demote band):** programs aren’t paused yet, but take a -2.0 priority penalty as early backpressure before a hard pause is needed.
> - **Resume only fires once U ≤ 0.85:** that’s pause-threshold minus a 0.10 resume-hysteresis, a buffer below the pause line that keeps the loop from flapping between pausing and resuming right at the threshold.

재개 자체는 빈 패킹 단계로 진행됩니다. 일시 정지된 프로그램을 토큰 수 오름차순으로 정렬한 뒤, 스케줄러가 가장 작은 것부터 수락하며, 다음 프로그램을 넣으면 사용률이 임계값을 넘게 되는 지점에서 멈춥니다. 이때 프로그램마다 고정된 버퍼를 함께 계산합니다. 재개된 요청은 1초 동안 우선순위가 올라가므로 새로 들어온 요청 뒤에 묶이지 않으며, 30분 강제 재개 상한이 있어 어떤 프로그램도 끝없이 기아 상태(starvation)에 빠지지 않습니다.
> Resume itself is a bin-packing pass: paused programs are sorted by token count ascending, and the scheduler admits them smallest-first until the next one would push utilization back over threshold, accounting for a fixed per-program buffer. Resumed requests get a one-second priority boost so they don’t get stuck behind fresh admissions, and a 30-minute forced-resume cap guarantees no program starves indefinitely.

### 결과 / Results

중간에서 높은 수준의 동시성에서는 처리량이 크게 늘어납니다. 그 차이는 처리량에 그대로 드러납니다. SWE-bench에서 8xH100 노드 하나에 TP4 MiniMax-M2 복제본 두 개를 실행했을 때, 프로그램 인식 스케줄링은 KV 라우팅만 쓸 때보다 처리량을 약 12~16% 높였습니다. 이 격차는 요청 단위 라우팅으로는 피할 수 없는 재프리필(re-prefill) 작업에서 나옵니다.
> At moderate to high concurrency, we see a large increase in throughput. the difference shows up directly in throughput. On SWE-bench, running two TP4 MiniMax-M2 replicas on a single 8xH100 node, program-aware scheduling improved throughput roughly 12-16% over KV-routing alone, with the gap coming from the re-prefill work that request-level routing can’t avoid.

같은 전략은 에이전틱 RL 롤아웃(rollout)에서도 비슷하게 큰 향상을 보입니다. 8xH20-3e에서 Qwen3-Coder-30B-A3B-Instruct로 Uni-Agent SWE-Bench를 실행하고 VERL의 기본 Global LB와 비교했습니다. 일시 정지가 필요 없는 낮은 동시성(CC32-128)에서는 저희 전략이 거의 비슷한 수준을 유지했습니다. 중간 동시성(CC192-256)에서는 앞서기 시작해 Global LB보다 모델 토큰 처리량이 11.0~14.6% 높았고, 높은 동시성(CC384-512)에서는 Global LB의 처리량이 급격히 떨어진 반면 저희 전략은 계속 확장되었습니다. 전체 측정 범위에서 프리픽스 캐시 적중률은 94.5% 이상을 유지했습니다.
> The same strategy shows a similarly large boost on agentic RL rollouts. Running Uni-Agent SWE-Bench on 8xH20-3e with Qwen3-Coder-30B-A3B-Instruct against VERL’s default Global LB, our strategy stayed roughly even at low concurrency (CC32-128), where nothing needed to pause. At medium concurrency (CC192-256) it pulled ahead, with 11.0-14.6% higher model-token throughput than Global LB, and at high concurrency (CC384-512) Global LB throughput dropped sharply while our strategy kept scaling. Across the full sweep it held prefix-cache hit rate above 94.5%.

![동시성 구간별 모델 토큰 처리량 비교 차트 / Model-token throughput across concurrency regimes](/assets/blog/2026-10-08-session-aware-agentic-inference-with-nvidia-dynamo/Dynamo-Concurrency-Regime-Chart.png){:style="width:100%"}
*그림 3: 에이전트 수락 동시성 수준에 따른 모델 토큰 처리량을 VERL Global LB, Dynamo Python TA, Dynamo의 네이티브 세션 인식 스케줄러 사이에서 비교합니다. 높은 동시성에서는 네이티브 스케줄러가 셋 가운데 가장 높은 처리량을 유지하는 반면, Global LB의 처리량은 급격히 떨어집니다. / Figure 3: Compares model-token throughput across agent admission concurrency levels for VERL Global LB, Dynamo Python TA, and Dynamo’s native session-aware scheduler. At high concurrency, the native scheduler maintains the highest throughput of the three, while Global LB throughput drops sharply.*

### 라우터의 공유 KV 캐시 인식 / Shared KV cache awareness in the router

워킹 셋이 HBM보다 커지면 KV 캐시 배치는 더 어려워집니다. 캐시가 사라지는 것이 아니라 CPU 메모리로, 이어서 Mooncake나 NIXL 기반 스토리지 같은 공유 저장소로 옮겨 가기 때문입니다. 지금까지 Dynamo 라우터는 GPU 메모리와 프레임워크 자체의 CPU 오프로드 메모리만 알고 있었습니다. KV가 외부 저장소로 옮겨 가면 Dynamo는 이를 비용 함수에 반영하지 못했습니다. 이 문제를 풀기 위해 Mooncake 팀과 협력해 Mooncake Store가 KV 이벤트를 직접 내보내도록 했고, Dynamo의 매우 빠른 FlashIndexer와 함께 동작하는 실험적인 공유 풀 인덱서를 새로 구현했습니다. KV가 HiCache(SGLang 고유의 CPU 오프로딩 메커니즘)에서 Mooncake로 옮겨 가면, SGLang은 제거(removal) 이벤트를, Mooncake는 저장(store) 이벤트를 내보냅니다. 페이지는 워커의 병렬화 구성(parallelism layout)에 필요한 물리 객체가 모두 있을 때만 적중으로 인정되며, 이 점수는 디바이스 로컬 프리픽스를 넘어선 부분에 적용됩니다.
> KV cache placement gets harder once our working set outgrows HBM. The cache doesn’t disappear, it moves into CPU memory and then into shared stores like Mooncake or NIXL backed storage. So far the Dynamo router was only aware of GPU and native framework CPU offloaded memory. When KV moved into an external store, Dynamo was unable to factor it into the cost function itself. To solve this, we worked with the Mooncake team to enable KV event emission natively from the Mooncake Store and implemented a new experimental shared-pool indexer that works in conjunction with Dynamo’s blazing fast FlashIndexer. When KV is moved from HiCache (SGLang’s native CPU offloading mechanism) into Mooncake, a removal event is emitted from SGLang, and a store event is emitted from Mooncake. A page only counts as a hit when every physical object for the worker’s parallelism layout is present, and credit applies beyond its device-local prefix.

요청이 들어오면 Dynamo는 프롬프트를 SGLang KV 페이지 단위로 해싱하고, 각 논리 페이지를 워커의 TP/PP 구성, K/V 텐서, 선택적 백엔드 태그가 정하는 물리 Mooncake 객체들로 펼칩니다. Mooncake는 순서가 보장된 `Store` 와 `Remove` 이벤트를 발행하고, Dynamo는 이를 받아 전역 공유 풀 인덱스에 반영합니다. 이때 필요한 물리 객체가 모두 있을 때만 페이지를 재사용 가능하다고 보수적으로 판단합니다. 인덱스는 공유 적중을 연속된 페이지 범위로 돌려주므로, 라우팅의 핫 패스(hot path)에서 Mooncake를 조회할 필요가 없습니다. FlashIndexer는 이와 별개로 후보 워커마다 자체 디바이스, CPU, 디스크의 프리픽스 겹침을 계산합니다. 디바이스에 상주하는 프리픽스 깊이가 `D` 인 워커에 대해, 라우터는 위치가 `>= D` 인 공유 적중에만 점수를 줍니다: `shared_beyond_device = |shared_hit_ranges ∩ [D, request_pages)|`. 이렇게 하면 GPU 로컬 적중을 두 번 세는 일을 막으면서도, 로컬 프리픽스가 더 짧은 워커가 Mooncake에 이미 있는 뒷부분(suffix)으로 점수를 받을 수 있습니다. 최종 배치 점수는 가중치를 준 네이티브 계층 겹침, 설정할 수 있는 공유 풀 점수, 실시간 워커 부하를 결합해 계산합니다.
> At request time, Dynamo hashes the prompt into SGLang KV pages and expands each logical page into the physical Mooncake objects implied by the worker’s TP/PP layout, K/V tensors, and optional backend tag. Mooncake publishes ordered `Store` and `Remove` events and Dynamo consumes them into a global shared-pool index, conservatively treating a page as reusable only when every required physical object is present. The index returns shared hits as contiguous page ranges rather than probing Mooncake on the routing hot path. FlashIndexer independently computes each candidate worker’s native device, CPU, and disk prefix overlap. For a worker with device-resident prefix depth `D`, the router credits only shared hits at positions `>= D`: `shared_beyond_device = |shared_hit_ranges ∩ [D, request_pages)|`. That prevents a GPU-local hit from being counted twice, while allowing a worker with a shorter local prefix to receive credit for the suffix already available in Mooncake. The final placement score combines weighted native-tier overlap, configurable shared-pool credit, and live worker load.

## 프로그래밍 방식 KV 캐시를 향해 / Towards Programmatic KV Cache

결국 에이전틱 워크로드에서 추론 성능을 내려면 세션을 인식하는 KV 블록 관리가 필요합니다. 이런 워크로드에서 특정 KV 캐시 블록의 우선순위 가치를 예측하려면 라우터와 엔진의 정보가 서로 보완되어야 합니다. 라우터는 워커 전반에 걸친 세션 수명 주기와 재사용 패턴을 알고, 엔진은 각자 보유한 KV 캐시의 상태를 압니다. 따라서 엔진(vLLM, SGLang)이 최대한 많은 정보를 바탕으로 캐시 정책을 실행하려면, (1) Dynamo가 정책의 의도를 표현하고, (2) 엔진이 그 의도를 정확한 캐시 상태에 맞춰 해석하고 실행해야 합니다.
> Ultimately, performant inference on agentic workloads requires session-aware KV block management. To predict the priority value of a given KV cache block in such workloads, we need complementary information from the router and the engine. The router is aware of session lifecycle and reuse patterns across workers; the engine is individually aware of the KV cache status it holds. Hence, for engines (vLLM, SGLang) to execute a maximally informed cache policy, (1) Dynamo should express policy intent, and (2) the engine should resolve that intent against exact cache state and execute it.

이것이 **프로그래밍 방식 KV 캐시(programmatic KV cache)** 의 핵심 아이디어입니다. 간단히 말해 Dynamo 라우터는 세션, 부하, 캐시를 고려한 똑똑한 KV 이동 정책을 계산하는 "두뇌" 역할을 하고, 엔진은 Dynamo가 지시한 이동을 "실행"합니다. 이를 위해 Dynamo와 vLLM/SGLang 양쪽에 기능을 개발하고 있습니다. Dynamo에는 KV 정책을 계산하고 내보내는 장치를, 엔진에는 Dynamo의 세션 기반 정책을 엔진 내부의 구체적인 KV 이동으로 저수준화(lowering)하는 데 필요한 일급(first-class) 추상화를 만들고 있습니다.
> This is the central idea of **programmatic KV cache**. Essentially, the Dynamo router provides the “brain” that calculates smart, session/load/cache-aware KV movement policies, while the engine “executes” the movements that Dynamo mandates. As such, we have been developing features for both Dynamo and vLLM/SGLang: the machinery for Dynamo to calculate+emit KV policies, and the first-class abstractions in the engines required to lower Dynamo’s session-based policies into concrete, in-engine KV movements.

## 라우터 주도 KV 힌트 정책 / Router-Driven KV Hints Policy

Dynamo의 프로그래밍 방식 KV 캐시 이동은 **라우터가 시작하는 좁은 힌트 인터페이스("KvHint")** 로 이루어집니다. 이 인터페이스로 Dynamo는 KV 캐시에 대한 의도를 vLLM/SGLang에 전달합니다. 이렇게 하면 엔진 고유의 스케줄러와 캐시 관리자를 크게 뜯어고치지 않아도 됩니다. ([SGLang 설계](https://github.com/sgl-project/sglang/issues/27574), [vLLM 설계](https://github.com/vllm-project/vllm/issues/51428)).
> Dynamo’s programmatic KV cache movements are driven by a **narrow, router-initiated hint surface (”KvHint”)** which enables Dynamo to pass KV cache intent to vLLM/SGLang. This avoids invasive changes to the engine’s native scheduler and cache manager. ([SGLang design](https://github.com/sgl-project/sglang/issues/27574), [vLLM design](https://github.com/vllm-project/vllm/issues/51428)).

오케스트레이터(orchestrator)는 요청 단위 정책이 볼 수 없는 정보를 갖고 있습니다. 어떤 세션이 아직 살아 있는지, 어떤 토큰 범위가 공유 프리픽스이고 어떤 범위가 고유한 뒷부분인지, 도구 호출 공백이 몇 밀리초 만에 끝날지 몇 분이나 이어질지, 서브에이전트가 언제 열리고 닫히는지 같은 정보입니다. 이런 지식은 하네스와의 약속(contract)과 세션 수명 주기가 이미 자리 잡은 라우터 계층에 있어야 하며, 계속 그곳에 머물러야 합니다. 엔진의 역할은 메커니즘을 노출하고, 의도는 라우터가 제공하게 두는 것입니다.
> The orchestrator holds what a request-local policy cannot see. Which sessions are still live, which token ranges are a shared prefix versus a unique tail, whether a tool-call gap will last milliseconds or minutes, when a subagent opens and closes. That knowledge belongs in the router layer, where the harness contract and the session lifecycle already live, and it should stay there. The engine’s job is to expose the mechanism and let the router supply the intent.

제안하는 것은 라우터가 엔진에 보낼 수 있는, 강제력이 없는 소프트 힌트(soft hint)의 좁은 집합입니다. 다음은 가능한 KvHint와 그 기능을 일부 분류한 것입니다:
> The proposal is a narrow set of soft hints the router can emit to the engine. Here is a partial taxonomy of potential KvHints and their functions:

| KvHint 이름 / KvHint name | 기능 / Functionality | 사용 예 / Example use |
| --- | --- | --- |
| Share | 캐시된 프리픽스를 p2p 전송으로 한 워커에서 다른 워커로 옮깁니다. / Moves a cached prefix from one worker to another via p2p transfer | 새 엔진 워커가 기존 워커의 캐시로 자신의 캐시를 미리 채웁니다. / A new engine worker warms its cache from the existing workers’ cache. |
| Prefetch | 필요해지기 전에 더 높은 계층(예: DRAM)에서 더 낮은 계층으로 KV를 가져옵니다. / Fetches KV from a higher tier (e.g. DRAM) to lower tier, ahead of need | 서브에이전트가 끝나 갈 때 메인 에이전트의 컨텍스트를 HBM으로 다시 불러옵니다. / Reloading a main agent’s context into HBM as a subagent approaches completion. |
| Demote | KV를 더 낮은 계층에서 더 높은 계층으로 밀어냅니다. / Pushes KV from a lower tier to a higher tier. | 긴 도구 호출, 또는 일시 정지된 궤적. / Long tool call, or paused trajectory. |
| Pin | 가치가 높은 KV 캐시를 정해진 TTL 동안 현재 계층에 고정합니다. / Pins a high-value KV cache onto its current tier for a bounded TTL | 모든 요청이 공유하는 시스템 캐시를 고정합니다. / Pinning a system cache shared by all requests. |
| Retain | 캐시 압박이 있을 때 KV 블록이 남겨질 가능성을 높이거나(또는 낮추도록) 편향을 줍니다. / Biases KV blocks to be more (or less) likely to be retained under cache pressure. | 도구 호출 중에 서브에이전트의 KV 블록을 남겨 둡니다. / Retaining a subagent’s KV blocks during a tool call. |

각 힌트에는 세션 ID가 함께 실려 가므로, KV를 받는 계층은 그 KV가 어느 세션에 속하는지 알 수 있습니다. 그래서 KV를 이름 없는 블록이 아니라 하나의 단위로 묶고, 배치하고, 회수할 수 있습니다. 이 점은 더 차가운(colder) 계층에서 가장 중요합니다. 호스트 메모리나 디스크를 쓰는 스토리지 백엔드는 그렇지 않으면 불투명한 해시만 보게 되고, 살아 있는 궤적의 프리픽스와 끝난 궤적의 프리픽스를 구분할 방법이 없기 때문입니다. 엔진은 어떤 힌트든 잘라내거나, 미루거나, 무시할 수 있으며, 힌트를 보내지 않는 워크로드는 지금과 똑같이 동작합니다.
> Each hint carries the session ID with it, so the tier receiving the KV knows which session it belongs to and can group, place, and reclaim it as a unit rather than as anonymous blocks. That matters most in the colder tiers, where a storage backend holding host memory or disk otherwise sees only opaque hashes and has no way to tell a live trajectory’s prefix from a finished one. The engine can clip, defer, or ignore any hint, and workloads that send no hints behave exactly as they do today.

Share는 최소한의 스케줄러 훅(hook)만으로 HiCache 위에 얹은 구현이 이미 동작하며, Prefetch와 Demote는 Share가 반영되고 나면 그 장치 위에 구축하도록 설계되어 있습니다.
> Share already has a working implementation layered into HiCache with minimal scheduler hooks, and Prefetch and Demote are designed to build on that machinery once Share lands.

## 세션-프리픽스 인덱서로 KV 힌트 정책 계산하기 / Enabling KV Hint Policy Computations with Session-Prefix Indexer

Dynamo가 세션 단위로 KvHint 정책을 구성하고 실행하려면, KV 블록과 그 블록의 세션 정체성, 세션 안에서의 위치상 계보(lineage)를 양방향으로 매핑하는 구조를 직접 갖고 계속 갱신해야 합니다. 이를 위해 세션-프리픽스 인덱서(Session-Prefix Indexer)를 구현했습니다. 세션-프리픽스 인덱서는 기수 트리(radix tree) 구조이며, 각 가지는 고유한 세션 하나(또는 다른 세션에서 갈라져 나온 서브에이전트 세션)를 나타냅니다.
> For Dynamo to compose and execute these KvHint policies on the session level, it must own a continuously updated structure which bidirectionally maps between KV blocks, their session identities, and their in-session positional lineage. For this, we implemented the Session-Prefix Indexer, a radix tree structure for which each branch represents a unique session (or subagent session which branched from another).

세션-프리픽스 인덱서는 추론 엔진의 KV 이벤트(Stored / Removed)와 Dynamo 라우터의 메인 인덱서("FlashIndexer")가 찾은 프리픽스 캐시 적중, 이 두 가지로 계속 채워집니다. 그 결과 Dynamo 라우터는 FlashIndexer와 세션-프리픽스 인덱서로 (1) 추론 엔진 워커가 어떤 블록을 갖고 있는지와 (2) 그 블록들이 어떤 세션에 속하는지를 함께 누적해서 볼 수 있습니다. 이렇게 정보가 풍부한 시야는 KV 블록이 어느 워커에 있는지, 활성 세션의 수명 주기, 엔진 워크로드에 대한 신호가 필요한 다양한 KV 힌트 정책이 활용할 수 있습니다.
> The Session-Prefix Indexer is continuously hydrated from both inference engine KV events (Stored / Removed) and the prefix cache hits from Dynamo router’s main indexer (“FlashIndexer”). Consequently, through the FlashIndexer and Session-Prefix Indexer, the Dynamo router has a cumulative view of (1) inference engine workers’ block ownership as well as (2) those same blocks’ session membership. This information-rich view can be consumed by various KV hint policies that require signals on KV blocks’ worker location, active session lifecycles, and/or engine workloads.

이 인덱서들을 바탕으로, 에이전틱 워크로드 데이터셋에서 Dynamo vLLM/SGLang과 함께 KvHint 정책을 활발히 실험하고 있습니다.
> With these indexers, we are actively experimenting with KvHint policies with Dynamo vLLM/SGLang on agentic workload datasets.

![Dynamo 라우터 안의 세션-프리픽스 인덱서 / Dynamo router with the Session-Prefix Indexer](/assets/blog/2026-10-08-session-aware-agentic-inference-with-nvidia-dynamo/Dynamo-Router-with-Session-Prefix-.png){:style="width:100%"}
*그림 4: 세션-프리픽스 인덱서는 Dynamo 라우터 안에서 공유 프리픽스, 세션 계보, KV 블록 위치를 추적합니다. 추론 워커가 보내는 KV 이벤트가 이 정보를 갱신하며, 이를 바탕으로 라우팅 결정과, vLLM이나 SGLang에 선택적인 KV 힌트를 보내는 정책을 지원합니다. / Figure 4: The Session-Prefix Indexer tracks shared prefixes, session lineage, and KV block locations within the Dynamo router. KV events from inference workers update this view, supporting routing decisions and policies that send optional KV hints to vLLM or SGLang.*

## 앞으로의 방향 / Looking forward

효율적인 에이전틱 추론은 세션의 전체 수명 주기에 걸쳐 컨텍스트를 관리하는 데 달려 있습니다. Dynamo의 세션 ID는 하네스의 동작을 라우팅, 수락 제어, 캐시 관리와 이어 주며, 서빙 스택이 시시각각 변하는 에이전트의 워킹 셋을 함께 바라볼 수 있게 합니다.
> Efficient agentic inference depends on managing context across the full session lifecycle. Dynamo’s session ID connects harness behavior to routing, admission control, and cache management, giving the serving stack a shared view of an agent’s evolving working set.

세션에 연결된 트레이스와 재생은 정책을 측정할 수 있게 해 주고, 세션 인식 수락 제어는 도구 경계에서 배압을 걸어 캐시 스래싱을 줄입니다. 이번 글에서 시험한 SWE-bench 구성에서 프로그램 인식 스케줄링은 KV 인식 라우팅만 쓸 때보다 처리량을 약 12~16% 높였습니다. 실험적인 공유 풀 인덱서는 라우팅이 볼 수 있는 범위를 외부 KV 저장소까지 넓힙니다.
> Session-linked traces and replay make policies measurable, while session-aware admission control reduces cache thrashing by applying backpressure at tool boundaries. In the SWE-bench configuration tested here, program-aware scheduling improved throughput by roughly 12–16% over KV-aware routing alone. The experimental shared-pool indexer extends routing visibility to external KV stores.

세션-프리픽스 인덱서와 제안된 KvHint 인터페이스는 선제적인 캐시 이동을 향한 기반을 쌓습니다. Dynamo는 정책의 의도를 제공하고, 엔진은 실행에 대한 제어권을 유지합니다. 핵심은 캐시 재사용과 세션 동시성을 함께 최적화해야 한다는 점입니다. 턴이 이어지는 동안 쓸모 있는 컨텍스트를 계속 쓸 수 있게 유지하고, 동시성이 커져도 프리필을 반복하지 않는 것이 더 많은 에이전트 세션을 효율적으로 서빙하는 데 핵심입니다.
> The Session-Prefix Indexer and proposed KvHint interface build toward proactive cache movement: Dynamo supplies policy intent, and engines retain control over execution. The key takeaway is that cache reuse and session concurrency must be optimized together. Keeping useful context available across turns and avoiding repeated prefill as concurrency grows are central to serving more agent sessions efficiently.
