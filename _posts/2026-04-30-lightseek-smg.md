---
layout: blog_detail
title: "LLM 서빙에서 CPU를 GPU로부터 분리해야 하는 이유"
author: Simo Lin, Chang Su, and Keyang Ru, members of LightSeek Foundation
category: ["pytorch.org", "translation"]
org_title: "The Case for Disaggregating CPU from GPU in LLM Serving"
org_link: https://pytorch.org/blog/lightseek-smg/
---

## 시작: 대규모에서 GIL 벽에 부딪히다 / How It Started: Hitting the GIL Wall at Scale

프로덕션 모델 서빙을 수년간 운영해왔습니다. Shepherd Model Gateway를 처음 만들기 시작했을 때 목표는 소박했습니다. 캐시 인식(cache-aware) 부하 분산이 추론 복제본(replica) 간 라우팅을 개선할 수 있을지 알아보는 것이었습니다.
> We've been running production model serving for many years. When we first started building Shepherd Model Gateway, the goal was modest: figure out if cache-aware load balancing could improve routing across inference replicas.

가능했습니다. 그리고 더 깊이 들어갈수록 훨씬 더 큰 문제를 발견했습니다.
> It could. And as we went deeper, we found a much bigger problem.

SGLang과 vLLM 모두에서 토큰화(tokenization)와 역토큰화(detokenization)가 병목이 되어 있었습니다. 이론상이 아니라, 실제 트래픽이 흐르는 프로덕션 환경에서였습니다. 근본 원인은 아키텍처에 있었습니다. 두 엔진 모두 내부적으로 Rust 또는 C++ 토크나이저 라이브러리를 사용하지만, 호출은 Python을 거칩니다. 즉 GIL(Global Interpreter Lock)을 거치게 됩니다. 이는 서빙 경로 한가운데에 위치한 CPU 바운드 작업에 단일 스레드 천장이 생긴다는 뜻입니다.
> In both SGLang and vLLM, tokenization and detokenization had become bottlenecks. Not in theory — in production, under real traffic. The root cause was architectural: although both engines use Rust or C++ tokenizer libraries underneath, the calls go through Python. That means the GIL. That means a single-threaded ceiling on CPU-bound work that sits directly in the serving path.

작은 규모에서는 문제가 되지 않습니다. 하지만 대규모 prefill-decode 분리 서빙이나, GPU 클러스터 전반에 걸친 대규모 전문가 병렬화(expert parallelism)에서는 엄청난 차이를 만듭니다. 이러한 구성은 GPU를 극도로 빠르게 만들기 때문에 파이프라인의 CPU 측이 제약 요소가 되어버립니다. GIL에 묶인 토큰화의 1마이크로초는 수십만 달러짜리 GPU가 입력을 기다리며 놀고 있는 1마이크로초입니다.
> At a small scale, this doesn't matter. At large-scale prefill-decode disaggregated serving, and at large-scale expert parallelism across GPU clusters, it matters enormously. These configurations make GPUs extremely fast — fast enough that the CPU side of the pipeline becomes the constraint. Every microsecond of GIL-bound tokenization is a microsecond where GPUs worth hundreds of thousands of dollars sit idle, waiting for input.

여기서 여정이 본격적으로 시작되었습니다. 게이트웨이에 대한 거창한 비전이 아니라, 프로덕션 문제에서 출발했습니다. CPU 워크로드 전체를 GPU 경로에서 분리하여 Rust로 실행할 수 있을까? Python이 Rust를 호출하는 방식이 아니라, 순수 Rust로. GIL 없이. 단일 스레드 천장 없이. Python 프로세스 경계 없이.
> That's where the journey really started. Not with a gateway vision — with a production problem. Could we disaggregate the entire CPU workload from the GPU path and run it in Rust? Not Python-calling-Rust. Pure Rust. No GIL. No single-threaded ceiling. No Python process boundaries.

답은 "가능하다" 였고, 이를 증명한 프로젝트가 Shepherd Model Gateway가 되었습니다.
> The answer was yes, and the project that proved it became Shepherd Model Gateway.

![SMG 아키텍처: 클라이언트 → 게이트웨이 → 라우터 → 워커 / SMG Architecture: Clients → Gateway → Router → Workers](/assets/blog/2026-04-30-lightseek-smg/architecture-animated.svg){:style="width:100%"}
*SMG 아키텍처: 클라이언트 → 게이트웨이 → 라우터 → 워커 / SMG Architecture: Clients → Gateway → Router → Workers*

SMG의 아키텍처는 하나의 원칙 위에 세워졌습니다. **GPU는 텐서 수학을 해야 하고, 그 외 모든 것은 전용 서빙 계층에 속해야 한다.**
> SMG's architecture is built on one principle: **GPUs should do tensor math. Everything else belongs in a dedicated serving layer.**

모델 서빙 스택을 살펴보며 GPU 추론과 얽혀 있는 모든 CPU 바운드 워크로드를 식별했습니다. 토큰화, 역토큰화, 추론(reasoning) 출력 파싱, 함수 호출 추출, MCP 도구 오케스트레이션, 멀티모달 전처리, 채팅 이력 관리, 구조화된 출력 검증, 정지 시퀀스 감지 등이 그것입니다. 이러한 작업이 Python GIL 뒤에서 GPU 프로세스와 함께 배치되면, 랙(rack)에서 가장 비싼 하드웨어에 백프레셔(back-pressure)를 발생시킵니다.
> We looked at the model-serving stack and identified every CPU-bound workload entangled with GPU inference: tokenization, detokenization, reasoning output parsing, function call extraction, MCP tool orchestration, multimodal preprocessing, chat history management, structured output validation, stop sequence detection. Each one is a CPU task that, when co-located with the GPU process behind the Python GIL, creates back-pressure on the most expensive hardware in the rack.

SMG는 이 모든 작업을 Rust 게이트웨이 계층으로 옮기고, 추론 엔진과는 gRPC로 통신합니다. 프로토콜은 최소화되어 있으며 GPU에 집중되어 있습니다. 즉, 전처리된 토큰을 입력으로 보내고 생성된 토큰을 스트리밍으로 받아옵니다. 그 외 모든 것은 게이트웨이의 책임입니다.
> SMG moves all of these into a Rust gateway layer that communicates with inference engines over gRPC. The protocol is minimal and GPU-focused: send preprocessed tokens in, stream generated tokens out. Everything else is the gateway's responsibility.

이는 분산 추론 분야의 대부분 프로젝트가 택한 접근과는 다릅니다. NVIDIA Dynamo, llm-d 같은 훌륭한 프로젝트들이 추론 엔진 계층과 그 주변의 오케스트레이션을 최적화하는 데 집중하고 있으며, 그 작업을 상호 보완적인 것으로 봅니다. 다만 SMG의 베팅은 다릅니다. 엔진을 더 똑똑하게 만드는 대신 게이트웨이를 더 똑똑하게 만드는 것입니다. GPU가 필요 없는 모든 것을 독립적으로 확장·진화하면서 GIL 경합 없이 동작하는 전용 Rust 계층으로 떠넘기는 방식입니다.
> This isn't the approach most projects in the distributed inference space have taken. Excellent work is happening with projects like NVIDIA Dynamo and llm-d, which focus on optimizing the inference engine layer and the orchestration around it. We see that work as complementary. But SMG's bet is different: rather than making the engine smarter, make the gateway smarter. Offload everything that doesn't require a GPU onto a purpose-built Rust layer that scales independently, evolves independently, and runs with zero GIL contention.

## gRPC 재설계: 실현하기 / The gRPC Re-Architecture: Making It Real

![gRPC 파이프라인: 엔진 인계 이전 게이트웨이 측 처리 / The gRPC Pipeline: Gateway-side processing before engine handoff](/assets/blog/2026-04-30-lightseek-smg/2.svg){:style="width:100%"}
*gRPC 파이프라인: 엔진 인계 이전 게이트웨이 측 처리 / The gRPC Pipeline: Gateway-side processing before engine handoff*

SMG 역사상 가장 큰 단일 기술 투자는 네이티브 Rust gRPC 데이터 플레인(data plane)을 중심으로 전체 서빙 파이프라인을 재구축한 것이었습니다. 이는 분리(disaggregation) 가설에 대한 아키텍처적 증명이었습니다.
> The single largest technical investment in SMG's history was rebuilding the entire serving pipeline around a native Rust gRPC data plane. This was the architectural proof of the disaggregation thesis.

**토큰화 및 역토큰화(Tokenization and detokenization)** 가 게이트웨이로 이동합니다. SMG는 토크나이저를 Rust에서 네이티브로 실행하며, 2단계 캐시를 사용합니다. 반복 프롬프트에 대한 L0 정확 일치(exact-match) 캐시와, 특수 토큰 경계에서 동작하는 L1 접두사 인식(prefix-aware) 캐시입니다. 추론 엔진은 사전 토큰화된 입력을 받으며 토크나이저를 절대 만지지 않습니다. Python 없음. GIL 없음.
> **Tokenization and detokenization** move into the gateway. SMG runs tokenizers natively in Rust with a two-level cache — L0 exact-match for repeated prompts, L1 prefix-aware at special-token boundaries. The inference engine receives pre-tokenized input and never touches a tokenizer. No Python. No GIL.

**추론 및 도구 호출 파싱(Reasoning and tool call parsing)** 은 게이트웨이의 스트리밍 파이프라인에서 실행됩니다. 토큰이 gRPC로 도착하면 SMG의 파서들 — Cohere Command, DeepSeek, Llama, Nemotron, Kimi-K2, GLM-4, Qwen Coder 등 — 이 추론 블록, 함수 호출, 구조화된 출력을 실시간으로 추출합니다. 엔진 측에 별도의 후처리 단계는 없습니다.
> **Reasoning and tool call parsing** runs in the gateway's streaming pipeline. As tokens arrive over gRPC, SMG's parsers — including Cohere Command, DeepSeek, Llama, Nemotron, Kimi-K2, GLM-4, and Qwen Coder — extract reasoning blocks, function calls, and structured output in real-time. No post-processing step on the engine side.

**멀티모달 처리(Multimodal processing)** 는 가장 야심찬 작업이었습니다. Hugging Face transformers의 이미지 프로세서 주요 구성 요소를 Python에서 Rust로 다시 작성했습니다. 비전 전처리 파이프라인, 텐서 연산, 모델별 변환을 완전히 다른 언어와 런타임에서 재구현한 것입니다. 그 결과, SMG는 전처리된 텐서를 Python 오버헤드 없이 gRPC로 곧장 엔진에 전달합니다. Llama 4 Vision, Qwen VL, 그리고 주요 비전-언어 모델 전반을 지원하며, SGLang, vLLM, TensorRT-LLM에 대한 백엔드별 최적화도 포함합니다. 저희가 아는 한, 이는 업계 최초입니다.
> **Multimodal processing** was the most ambitious piece. We rewrote major components of Hugging Face's transformers image processor from Python to Rust — reimplementing vision preprocessing pipelines, tensor operations, and model-specific transformations in a completely different language and runtime. The result: SMG communicates preprocessed tensors directly to engines via gRPC with zero Python overhead. Support for Llama 4 Vision, Qwen VL, and all major vision-language models, with backend-specific optimizations for SGLang, vLLM, and TensorRT-LLM. This is, to our knowledge, an industry first.

**MCP 도구 오케스트레이션(MCP tool orchestration)** 은 인증 인식(auth-aware) 커넥션 풀링, 동시 배치 실행, 승인 워크플로우, 자동 재연결, HTTP 헤더 전달과 함께 전적으로 게이트웨이에서 실행됩니다. 추론 엔진은 MCP를 전혀 알지 못합니다. 또한 어떤 모델에든 어떤 MCP 서버든 네이티브 기능(FileSearch, WebSearch, CodeInterpreter)으로 만들어주는 완전한 내장 도구 라우팅 인프라도 구축했습니다. Llama나 Qwen을 GPT-4와 동일한 내장 도구와 함께 배포할 수 있습니다.
> **MCP tool orchestration** runs entirely in the gateway with auth-aware connection pooling, concurrent batch execution, approval workflows, automatic reconnection, and HTTP header forwarding. The inference engine has no knowledge of MCP. We also built a complete built-in tool routing infrastructure — turning any MCP server into native capabilities (FileSearch, WebSearch, CodeInterpreter) for any model. Deploy Llama or Qwen with the same built-in tools as GPT-4.

**채팅 이력 관리(Chat history management)** 는 플러그형 저장소(PostgreSQL, OracleDB, Redis, 인메모리), Flyway를 통한 스키마 버전 관리, 테이블/컬럼 이름 커스터마이즈, 영속화 전후 콜백을 위한 저장소 훅을 제공합니다. 모두 게이트웨이에 있으며, 엔진은 상태가 없도록(stateless) 유지됩니다.
> **Chat history management** with pluggable storage (PostgreSQL, OracleDB, Redis and in-memory), schema versioning via Flyway, customizable table/column names, and storage hooks for pre/post persistence callbacks. All in the gateway, keeping the engine stateless.

**WASM 미들웨어(WASM middleware)** 는 코드베이스를 포크(fork)하지 않고도 프로그래머블한 확장성을 제공합니다. 커스텀 인증, 컴플라이언스 로깅, PII 마스킹, 비용 추적, 압축 — 모두 샌드박스 격리가 적용된 WebAssembly 플러그인으로 처리됩니다. 또 다른 업계 최초입니다.
> **WASM middleware** provides programmable extensibility without forking the codebase. Custom authentication, compliance logging, PII redaction, cost tracking, compression — all via WebAssembly plugins with sandboxed isolation. Another industry first.

gRPC 프로토콜 자체는 PyPI에 `smg-grpc-proto`로 게시되어 있으며, 게이트웨이와 엔진 사이의 좁은 계약(contract)을 정의합니다. 이 설계 덕분에 추론 엔진을 건드리지 않고도 게이트웨이를 업그레이드(새로운 파서, 새로운 프로토콜, 새로운 도구)할 수 있고, 게이트웨이를 건드리지 않고도 엔진을 업그레이드(새로운 GPU 커널, 새로운 양자화(quantization))할 수 있습니다. 인터페이스가 깨끗하기 때문에 둘이 독립적으로 진화합니다.
> The gRPC protocol itself — published as smg-grpc-proto on PyPI — defines the narrow contract between gateway and engine. This design means you can upgrade your gateway (new parsers, new protocols, new tools) without touching your inference engine, and upgrade your engine (new GPU kernels, new quantization) without touching your gateway. They evolve independently because the interface is clean.

## SMG가 오늘 제공하는 것 / What SMG Delivers Today

[SMG](https://github.com/lightseekorg/smg)는 LightSeek Foundation의 일원인 Simo Lin과 Chang Su가 만들었습니다. 약 6개월 동안 열세 번의 릴리스를 출시했습니다. 각 릴리스를 일일이 살펴보는 대신, 현재 프로젝트가 무엇을 제공하는지와 각 기능의 근거를 정리합니다.
> [SMG](https://github.com/lightseekorg/smg) was created by Simo Lin and Chang Su, members of the LightSeek Foundation. In roughly six months, we shipped thirteen releases. Rather than walk through each one, here is what the project delivers today — and the evidence behind each capability.

### 멀티 모델 추론 게이트웨이 / Multi-Model Inference Gateway

단일 SMG 프로세스가 전체 플릿(fleet)의 정면에 위치합니다. 여러 모델, 여러 엔진, 단일 진입점. SGLang, vLLM, TensorRT-LLM, MLX 백엔드에 동시에 요청을 라우팅합니다. OpenAI, Anthropic, Google Gemini, AWS Bedrock, Azure OpenAI를 외부 제공자(provider)로 추가할 수 있습니다. 하나의 게이트웨이, 모든 엔진, 모든 벤더.
> A single SMG process fronts your entire fleet — multiple models, multiple engines, one entry point. Route requests across SGLang, vLLM, TensorRT-LLM, and MLX backends simultaneously. Add OpenAI, Anthropic, Google Gemini, AWS Bedrock, and Azure OpenAI as external providers. One gateway, every engine, every vendor.

### 5개의 네이티브 에이전트 API / Five Native Agentic APIs

SMG는 Chat Completions(OpenAI), Responses API(OpenAI), Messages API(Anthropic), Interactions API(Gemini), Realtime API(WebSocket/WebRTC)를 네이티브로 지원합니다. 변환 계층(translation layer)이 아니라 각각이 그 자체로 완전한 일급(first-class) 구현입니다. Messages API는 `ThinkingConfig`, `thinking_delta` 스트리밍 이벤트, 그리고 추론(reasoning)·텍스트·도구 사용 콘텐츠 블록의 인터리브(interleaved)를 통해 사고(thinking) 블록을 종단 간 보존합니다. Responses API는 OpenAI의 대화 관리 기능을 Llama, DeepSeek, Qwen 등 모든 오픈소스 모델에 적용해 줍니다. SMG는 이를 지원하는 유일한 오픈소스 게이트웨이입니다. Claude를 위해 설계된 에이전트 워크플로우를 Llama 4, Qwen 3, DeepSeek, Kimi에서 완전한 프로토콜 충실도(fidelity)로 실행할 수 있습니다.
> SMG natively supports Chat Completions (OpenAI), Responses API (OpenAI), Messages API (Anthropic), Interactions API (Gemini), and Realtime API (WebSocket/WebRTC). These are not translation layers — each is a first-class implementation. The Messages API preserves thinking blocks end-to-end with ThinkingConfig, thinking_delta streaming events, and interleaved reasoning + text + tool use content blocks. The Responses API brings OpenAI's conversation management to Llama, DeepSeek, Qwen, and every open-source model — SMG remains the only open-source gateway supporting it. Run agentic workflows designed for Claude on Llama 4, Qwen 3, DeepSeek, or Kimi with full protocol fidelity.

### 네이티브 Rust gRPC 데이터 플레인 / Native Rust gRPC Data Plane

![2단계 토크나이저 캐시: L0 정확 일치, L1 접두사 인식 / Two-Level Tokenizer Cache: L0 exact-match, L1 prefix-aware](/assets/blog/2026-04-30-lightseek-smg/tokenization-cache.svg){:style="width:100%"}
*2단계 토크나이저 캐시: L0 정확 일치, L1 접두사 인식 / Two-Level Tokenizer Cache: L0 exact-match, L1 prefix-aware*

아키텍처의 핵심은 게이트웨이와 엔진 사이의 네이티브 Rust gRPC 파이프라인입니다. 계약은 최소화되어 있습니다. 전처리된 토큰을 입력으로, 생성된 토큰을 출력으로. 그 외 모든 것은 게이트웨이의 책임입니다. 토큰화는 2단계 캐시(L0 정확 일치, L1 접두사 인식)와 함께 Rust에서 실행됩니다. 추론(reasoning) 및 도구 호출 파싱은 토큰이 도착하는 대로 스트리밍 파이프라인에서 실행되며, DeepSeek-R1, Qwen3, GLM-4, Kimi, Llama-4, Cohere Command 등 15개 모델 패밀리를 지원합니다. Python 없음. GIL 없음. gRPC 프로토콜은 PyPI에 `smg-grpc-proto`로 게시되어 있으며, vLLM([PR #36169](https://github.com/vllm-project/vllm/pull/36169))과 NVIDIA TensorRT-LLM(병합된 5개의 PR) 양쪽 모두에 업스트림으로 채택되었습니다.
> The architectural core: a native Rust gRPC pipeline between gateway and engine. The contract is minimal — preprocessed tokens in, generated tokens out. Everything else is the gateway's responsibility. Tokenization runs in Rust with a two-level cache (L0 exact-match, L1 prefix-aware). Reasoning and tool call parsing runs in the streaming pipeline as tokens arrive — supporting fifteen model families including DeepSeek-R1, Qwen3, GLM-4, Kimi, Llama-4, Cohere Command, and more. No Python. No GIL. The gRPC protocol is published as smg-grpc-proto on PyPI, and both vLLM ([PR #36169](https://github.com/vllm-project/vllm/pull/36169)) and NVIDIA TensorRT-LLM (five merged PRs) have adopted it upstream.

### 지능형 라우팅 / Intelligent Routing

![캐시 인식 라우팅 흐름 / Cache-Aware Routing Flow](/assets/blog/2026-04-30-lightseek-smg/4.svg){:style="width:100%"}
*캐시 인식 라우팅 흐름 / Cache-Aware Routing Flow*

8가지 부하 분산 정책을 제공합니다. 캐시 인식(cache-aware), 라운드 로빈, 랜덤, power-of-two, 일관성 해싱(consistent hashing), 접두사 해시(prefix hash), 수동(스티키 세션, sticky sessions), 버킷 기반(bucket-based). 캐시 인식 라우팅은 처음부터 다시 작성되었습니다. 10~12배 빠르고(초당 216,000회 삽입), 메모리 사용량이 99% 줄었습니다(노드당 180 KB → 1.4 KB, 캐시된 접두사 10,000개 기준 1.8 GB → 14 MB). 이벤트 기반 KV 캐시 라우팅은 `SubscribeKvEvents` RPC를 통해 모든 백엔드로부터 실시간 캐시 상태를 스트리밍하며, 블록 크기는 자동 학습됩니다. Llama 복제본 8개에서의 프로덕션 결과: TTFT 평균 23% 감소, TTFT p99 28% 감소. Prefill-decode 분리는 prefill 단계와 decode 단계를 각각 독립된 정책의 워커 풀로 라우팅합니다. PD 구성에서 TTFT가 20~30% 개선됩니다.
> Eight load-balancing policies: cache-aware, round robin, random, power-of-two, consistent hashing, prefix hash, manual (sticky sessions), and bucket-based. Cache-aware routing was rewritten from the ground up — 10–12x faster (216,000 insertions/sec), 99% memory reduction (180 KB → 1.4 KB per node, 10,000 cached prefixes: 1.8 GB → 14 MB). Event-driven KV cache routing streams real-time cache state from all backends via SubscribeKvEvents RPC, with auto-learned block sizes. Production results on 8 Llama replicas: TTFT average down 23%, TTFT p99 down 28%. Prefill-decode disaggregation routes prefill and decode phases to separate worker pools with independent policies — 20–30% TTFT improvement in PD setups.

### Rust 멀티모달 처리 / Multimodal Processing in Rust

Hugging Face의 이미지 프로세서 주요 구성 요소를 Python에서 Rust로 다시 작성했습니다. 비전 전처리 파이프라인, 텐서 연산, 모델별 변환을 완전히 다른 언어와 런타임으로 옮긴 것입니다. 8개의 비전 모델 패밀리를 지원합니다. Kimi K2.5, Llama-4 Vision, LLaVA, Phi-3/Phi-4 Vision, Pixtral, Qwen-VL, Qwen2-VL, Qwen3-VL. 전처리된 텐서는 Python 오버헤드 없이 gRPC로 곧장 엔진에 전달됩니다. 저희가 아는 한, 업계 최초입니다.
> We rewrote major components of Hugging Face's image processors from Python to Rust — vision preprocessing pipelines, tensor operations, and model-specific transformations in a completely different language and runtime. Eight vision model families supported: Kimi K2.5, Llama-4 Vision, LLaVA, Phi-3/Phi-4 Vision, Pixtral, Qwen-VL, Qwen2-VL, and Qwen3-VL. Preprocessed tensors flow directly to engines via gRPC with zero Python overhead. To our knowledge, an industry first.

### MCP 도구 오케스트레이션 & 내장 도구 / MCP Tool Orchestration & Built-in Tools

![MCP 아키텍처: 게이트웨이에서의 도구 오케스트레이션 / MCP Architecture: Tool orchestration in the gateway](/assets/blog/2026-04-30-lightseek-smg/mcp-architecture.svg){:style="width:100%"}
*MCP 아키텍처: 게이트웨이에서의 도구 오케스트레이션 / MCP Architecture: Tool orchestration in the gateway*

MCP는 인증 인식 커넥션 풀링, 동시 배치 실행, 승인 워크플로우, 자동 재연결, 그리고 4개의 전송(transport)인 STDIO, HTTP, SSE, Streamable과 함께 전적으로 게이트웨이에서 실행됩니다. Universal MCP Built-in Tools는 어떤 MCP 서버든 모델에게 네이티브 기능 — FileSearch, WebSearch, CodeInterpreter — 으로 노출시킵니다. Llama나 Qwen을 GPT-4와 동일한 내장 도구와 함께 배포할 수 있습니다. 테넌트별 격리, 정책 기반 신뢰 레벨, 실행 메트릭이 기본으로 제공됩니다.
> MCP runs entirely in the gateway with auth-aware connection pooling, concurrent batch execution, approval workflows, automatic reconnection, and four transports (STDIO, HTTP, SSE, Streamable). Universal MCP Built-in Tools turn any MCP server into native capabilities — FileSearch, WebSearch, CodeInterpreter — for any model. Deploy Llama or Qwen with the same built-in tools as GPT-4. Per-tenant isolation, policy-based trust levels, and execution metrics come standard.

### WASM 미들웨어 / WASM Middleware

![WASM 플러그인 파이프라인 / WASM Plugin Pipeline](/assets/blog/2026-04-30-lightseek-smg/wasm-plugins.svg){:style="width:100%"}
*WASM 플러그인 파이프라인 / WASM Plugin Pipeline*

샌드박스 격리가 적용된 WebAssembly 플러그인을 통한 프로그래머블 확장성 — 또 하나의 업계 최초입니다. 커스텀 인증, 컴플라이언스 로깅, PII 마스킹, 비용 추적, 압축 — 코드베이스를 포크하지 않고 모두 가능합니다. Wasmtime 위에 Component Model과 비동기 지원을 더해 구축했습니다. 저장소 훅은 채팅 이력 연산을 가로채 커스텀 전후 처리(pre/post processing)를 가능하게 합니다.
> Programmable extensibility via WebAssembly plugins with sandboxed isolation — another industry first. Custom authentication, compliance logging, PII redaction, cost tracking, compression — all without forking the codebase. Built on Wasmtime with Component Model and async support. Storage hooks intercept chat history operations for custom pre/post processing.

### 엔터프라이즈 보안 & 관측성 / Enterprise Security & Observability

![TLS/mTLS 아키텍처 / TLS/mTLS Architecture](/assets/blog/2026-04-30-lightseek-smg/tls-architecture.svg){:style="width:100%"}
*TLS/mTLS 아키텍처 / TLS/mTLS Architecture*

JWKS 디스커버리를 갖춘 JWT/OIDC 인증, 역할 기반 접근 제어(RBAC), API 키 인증, 멀티 테넌트 속도 제한(rate limiting). 클라이언트 대상과 노드 간 통신 모두에 대한 TLS 및 mTLS. 6계층 메트릭 시스템에 HTTP, 라우터, 워커, 추론, 디스커버리, MCP, 데이터베이스, 메시(mesh) 계층을 아우르는 40개 이상의 Prometheus 메트릭. 완전한 OpenTelemetry 분산 추적. 요청 상관관계가 포함된 구조화된 JSON 로깅.
> JWT/OIDC authentication with JWKS discovery, role-based access control, API key auth, and multi-tenant rate limiting. TLS and mTLS for both client-facing and inter-node communication. A six-layer metrics system with 40+ Prometheus metrics covering HTTP, router, worker, inference, discovery, MCP, database, and mesh layers. Full OpenTelemetry distributed tracing. Structured JSON logging with request correlation.

### 신뢰성 & 고가용성 / Reliability & High Availability

![서킷 브레이커 상태 머신 / Circuit Breaker State Machine](/assets/blog/2026-04-30-lightseek-smg/closed.png){:style="width:100%"}
*서킷 브레이커 상태 머신 / Circuit Breaker State Machine*

워커별 서킷 브레이커(closed/open/half-open), 지수 백오프와 지터(jitter)가 포함된 자동 재시도, 주기적 헬스 체크, 동시 요청 속도 제한, 요청 타임아웃, 설정 가능한 우아한 종료(graceful shutdown). 멀티 노드 배포를 위한 CRDT 기반 상태 동기화를 갖춘 SWIM 프로토콜 가십(gossip) 메시. 클러스터 노드 간 일관성 해싱을 통한 분산 속도 제한. 설계상 분할 내성(partition-tolerant)이 보장됩니다.
> Per-worker circuit breakers (closed/open/half-open), automatic retries with exponential backoff and jitter, periodic health checks, concurrent request rate limiting, request timeouts, and configurable graceful shutdown. SWIM-protocol gossip mesh with CRDT-based state sync for multi-node deployments. Distributed rate limiting via consistent hashing across cluster nodes. Partition-tolerant by design.

### 데이터 영속성 & 서비스 디스커버리 / Data Persistence & Service Discovery

![서비스 디스커버리: Kubernetes, DNS, 수동 / Service Discovery: Kubernetes, DNS, Manual](/assets/blog/2026-04-30-lightseek-smg/ksd.svg){:style="width:100%"}
*서비스 디스커버리: Kubernetes, DNS, 수동 / Service Discovery: Kubernetes, DNS, Manual*

플러그형 저장소(PostgreSQL, OracleDB, Redis, 인메모리)와 함께 채팅 이력을 관리하며, 스키마 버전 관리와 테이블/컬럼 이름 커스터마이즈를 지원합니다. Kubernetes 레이블 기반 파드 디스커버리, DNS 디스커버리, 수동 워커 URL 중에서 선택할 수 있습니다. 모델 ID는 파드 네임스페이스, 레이블, 또는 어노테이션에서 가져옵니다. PD 구성에서는 부트스트랩 포트 어노테이션을 통해 prefill 포트를 자동 디스커버리합니다.
> Chat history management with pluggable storage — PostgreSQL, OracleDB, Redis, or in-memory — with schema versioning and customizable table/column names. Kubernetes label-based pod discovery, DNS discovery, or manual worker URLs. Model ID sourced from pod namespace, labels, or annotations. Bootstrap port annotation for automatic prefill port discovery in PD setups.

### 범용 플랫폼 지원 / Universal Platform Support

Linux, Windows, macOS, x86, ARM — 단일 Python wheel(`pip install smg`)로 지원합니다. Python 3.8–3.14. Python, Rust, Java, Go로 작성된 프로덕션 준비 완료 클라이언트 SDK. 엔진별 Docker 이미지. 독립 크레이트로 완전히 모듈화: `smg-auth`, `smg-mesh`, `smg-mcp`, `smg-wasm`, `smg-grpc-client`, `smg-kv-index`, `llm-tokenizer`, `llm-multimodal`, `openai-protocol` 등.
> Linux, Windows, macOS, x86, ARM — from a single Python wheel (pip install smg). Python 3.8–3.14. Production-ready client SDKs in Python, Rust, Java, and Go. Engine-specific Docker images. Full modularization into standalone crates: smg-auth, smg-mesh, smg-mcp, smg-wasm, smg-grpc-client, smg-kv-index, llm-tokenizer, llm-multimodal, openai-protocol, and more.

## 가설 입증: gRPC 게이트웨이 벤치마크 / Proving the Thesis: gRPC Gateway Benchmarks

분리 가설은 CPU 워크로드를 GPU 경로 밖으로 옮기면 측정 가능한 이득이 있어야 한다는 것 — 특히 프로덕션 조건에서 — 을 예측합니다. 이를 체계적으로 검증했습니다.
> The disaggregation thesis predicts that moving CPU workloads off the GPU path should show measurable benefits — especially under production conditions. We tested this systematically.

### 방법론 / Methodology

모든 벤치마크는 NVIDIA H100 GPU에서 GitHub Actions의 SMG nightly 벤치마크 스위트를 통해 NVIDIA GenAI-Perf(`genai-perf`)로 실행되었습니다. 8개 모델(GPT-OSS-20B, Llama-3.1-8B, Llama-3.3-70B, Llama-3.3-70B-FP8, Llama-4-Maverick, Llama-4-Scout, Qwen2.5-7B, Qwen3-30B-MoE), 2개 런타임(SGLang, vLLM), 5개 트래픽 시나리오, 9개 동시성(concurrency) 레벨(1–256). 총 1,082개의 매칭된 gRPC vs HTTP 비교 지점.
> All benchmarks run on NVIDIA H100 GPUs using NVIDIA GenAI-Perf (genai-perf) via the SMG nightly benchmark suite on GitHub Actions. 8 models (GPT-OSS-20B, Llama-3.1-8B, Llama-3.3-70B, Llama-3.3-70B-FP8, Llama-4-Maverick, Llama-4-Scout, Qwen2.5-7B, Qwen3-30B-MoE), 2 runtimes (SGLang, vLLM), 5 traffic scenarios, 9 concurrency levels (1–256). Total: 1,082 matched gRPC vs HTTP comparison points.

### 스케일링 이야기: 동시성이 커질수록 이점도 커진다 / The Scaling Story: Advantage Grows with Concurrency

동시성 1에서는 gRPC와 HTTP가 측정 노이즈 범위 내에서 비슷한 성능을 보입니다. 동시성 256에서는 gRPC가 약 8% 더 높은 처리량(throughput)을 제공합니다. 게이트웨이의 바이너리 직렬화(serialization)와 HTTP/2 멀티플렉싱(multiplexing)은 부하가 커질수록 — 즉 가장 중요한 순간에 — 누적되어 효과를 발휘합니다.
> At concurrency 1, gRPC and HTTP perform within noise. At concurrency 256, gRPC delivers ~8% more throughput. The gateway's binary serialization and HTTP/2 multiplexing compound under load — exactly when it matters.

![동시성에 따른 gRPC vs HTTP 처리량 / gRPC vs HTTP throughput across concurrency](/assets/blog/2026-04-30-lightseek-smg/smg-blog.png){:style="width:100%"}

### 긴 컨텍스트: gRPC가 성능을 바꾸는 지점 / Long Contexts: Where gRPC Transforms Performance

HTTP/JSON 직렬화 비용은 프롬프트 길이에 따라 선형으로 증가합니다. gRPC/protobuf는 컴팩트한 바이너리 인코딩을 사용해 이 비용을 지불하지 않습니다. 7,800 입력 토큰에서 직렬화 비용은 상당해집니다. D(7800,200) 시나리오에서는 모든 모델 평균 +12.2%의 처리량 우위를 보입니다.
> HTTP/JSON serialization cost grows linearly with prompt length. gRPC/protobuf uses compact binary encoding that doesn't pay this tax. At 7800 input tokens, the serialization cost is substantial. The D(7800,200) scenario shows +12.2% throughput advantage across all models.

가장 극적인 결과는 7,800 토큰 입력의 Llama-3.3-70B-FP8입니다. 이 모델은 H100에서 FP8 양자화로 실행되며, HTTP 직렬화가 지배적 병목이 될 만큼 충분히 빠릅니다. gRPC는 최대 3.5배 더 높은 출력 처리량을 제공합니다. 초당 1,150 토큰 vs 327 토큰.
> The most dramatic result: Llama-3.3-70B-FP8 with 7800-token inputs. This model, running FP8 quantization on H100, is fast enough that HTTP serialization becomes the dominant bottleneck. gRPC delivers up to 3.5x higher output throughput: 1,150 tok/s vs 327 tok/s.

![긴 컨텍스트 처리량 비교 / Long-context throughput comparison](/assets/blog/2026-04-30-lightseek-smg/long-context.png){:style="width:100%"}

![긴 컨텍스트 결과 추가 차트 / Additional long-context chart](/assets/blog/2026-04-30-lightseek-smg/blog.png){:style="width:100%"}

### 고동시성에서의 모델별 세부 분석 / Per-Model Breakdown at High Concurrency

프로덕션 동시성 수준(32–256)에서 gRPC의 이점은 모델 아키텍처에 따라 다릅니다. Llama-3.3-70B-FP8이 가장 큰 이득을 봅니다(E2E p99 +15.8%, 출력 처리량 +44.6%). 더 작은 dense 모델(Llama-3.1-8B, Qwen2.5-7B)은 완만한 개선을 보입니다. 패턴은 분명합니다: GPU가 빠를수록 → gRPC의 이점이 커집니다. CPU 오버헤드가 전체 지연 시간에서 차지하는 비중이 더 커지기 때문입니다.
> At production concurrency levels (32–256), the gRPC advantage varies by model architecture. Llama-3.3-70B-FP8 sees the largest gains (+15.8% E2E p99, +44.6% output throughput). Smaller dense models (Llama-3.1-8B, Qwen2.5-7B) show modest improvements. The pattern is clear: faster GPUs → larger gRPC advantage, because CPU overhead becomes a bigger fraction of total latency.

![모델별 고동시성 비교 / Per-model breakdown at high concurrency](/assets/blog/2026-04-30-lightseek-smg/blog-formal.png){:style="width:100%"}

## 생태계 / The Landscape

LLM 인프라를 다루는 팀이 저희만 있는 것은 아닙니다.
> We're not the only team working on LLM infrastructure.

NVIDIA Dynamo는 깊은 하드웨어 통합과 최적화된 추론 오케스트레이션을 제공합니다. llm-d는 Kubernetes 네이티브 접근으로 분산 추론 스케줄링을 다룹니다. 두 프로젝트 모두 엔진과 클러스터 계층에서 중요한 일을 하고 있습니다.
> NVIDIA Dynamo brings deep hardware integration and optimized inference orchestration. llm-d tackles distributed inference scheduling with a Kubernetes-native approach. Both are doing important work at the engine and cluster layer.

SMG는 다른 경계에서 동작합니다. 서빙과 프로토콜 계층입니다. 클라이언트와 GPU 사이의 모든 것 — 토큰화, 에이전트 프로토콜 변환, 도구 오케스트레이션, 캐시 인식 라우팅, 멀티모달 전처리, 신뢰성 — 을 책임집니다. 하나의 계층, 외부 의존성 0, 순수 Rust.
> SMG operates at a different boundary: the serving and protocol layer. We own everything between the client and the GPU — tokenization, agentic protocol translation, tool orchestration, cache-aware routing, multimodal preprocessing, reliability. One layer, zero external dependencies, pure Rust.

핵심 통찰은 이 접근들이 서로 조합된다는 점입니다. SMG를 llm-d가 관리하는 vLLM 앞에 둘 수도 있고, Dynamo가 GPU 오케스트레이션을 담당하는 TensorRT-LLM 앞에 둘 수도 있습니다. 책임이 다르기 때문에 경계가 깨끗합니다.
> The key insight: these approaches compose. You can run SMG in front of vLLM managed by llm-d, or in front of TensorRT-LLM with Dynamo handling GPU orchestration. The boundaries are clean because the responsibilities are different.

## 프로덕션 도입 / Production Adoption

SMG는 다음의 프로덕션 배포를 지원합니다.

1. Google Cloud Platform — 멀티 테넌트 AI 인프라
2. Oracle Cloud Infrastructure — 엔터프라이즈 GenAI 서비스
3. Alibaba Cloud — 클라우드 네이티브 AI 워크로드
4. TogetherAI — 분산 추론 인프라

> SMG powers production deployments at:
>
> - Google Cloud Platform — multi-tenant AI infrastructure
> - Oracle Cloud Infrastructure — enterprise GenAI services
> - Alibaba Cloud — cloud-native AI workloads
> - TogetherAI — distributed inference infrastructure

스타트업부터 하이퍼스케일러까지.
> From startups to hyperscalers.

## 다음 단계 / What's Next

1. 배치 API 스케줄링 — 오프라인 워크로드를 위한 Job Scheduler와 Capacity Governor의 2계층 아키텍처.
2. 시맨틱 라우팅 — 정적 규칙이 아닌 콘텐츠에 기반해 서로 다른 백엔드로 분배하는 경량 분류 기반 디스패치.
3. Mixture of Vendors (MoV) — 동일 모델을 여러 제공자에 라우팅하여 A/B 테스트, 비용 최적화, 품질 비교 수행.
4. MCP 시맨틱 검색 — 수백 개의 등록된 도구를 보유한 서버들 사이의 효율적인 도구 탐색.
5. 커스텀 메트릭 부하 분산 — 임의 메트릭에 대한 CEL 표현식으로, 밀리초 미만의 라우팅 오버헤드.

> 1. Batch API scheduling — two-tier architecture with Job Scheduler and Capacity Governor for offline workloads.
> 2. Semantic routing — lightweight classification-based dispatch to different backends based on content, not static rules.
> 3. Mixture of Vendors (MoV) — route the same model across multiple providers for A/B testing, cost optimization, and quality comparison.
> 4. MCP Semantic Search — efficient tool discovery across servers with hundreds of registered tools.
> 5. Custom metrics load balancing — CEL expressions over arbitrary metrics with sub-millisecond routing overhead.

**GitHub:** [github.com/lightseekorg/smg](http://github.com/lightseekorg/smg)

**설치(Install):** `pip install smg --upgrade`

**문서(Docs):** [lightseekorg.github.io/smg](http://lightseekorg.github.io/smg)

## 감사의 글 / Acknowledgement

SMG의 개발은 업계 전반의 엔지니어링 팀 및 오픈소스 커뮤니티와의 긴밀한 협업을 통해 발전해 왔습니다. 다음 분들의 기여, 피드백, 파트너십에 감사드립니다.

- Oracle Generative AI Service — Jun Qian, Jingqiao Zhang, Wei Gao, Keyang Ru, Xinyue Zhang, Yifeng Liu, Ziwen Zhao, Daisy Zhou, Khoa Tran.
- TogetherAI — Yineng Zhang, Wei Gong, Chandra Mourya, Connor Li.
- Thinking Machines Lab — Eric Zhang, Rajat Goel, Jeff Hanson.

> SMG's development has been shaped by close collaboration with engineering teams and open-source communities across the industry. We're grateful for the contributions, feedback, and partnership of:
>
> - Oracle Generative AI Service — Jun Qian, Jingqiao Zhang, Wei Gao, Keyang Ru, Xinyue Zhang, Yifeng Liu, Ziwen Zhao, Daisy Zhou, Khoa Tran.
> - TogetherAI — Yineng Zhang, Wei Gong, Chandra Mourya, Connor Li.
> - Thinking Machines Lab — Eric Zhang, Rajat Goel, Jeff Hanson.

또한 업스트림 협업과 프로토콜 채택에 대해 SGLang, vLLM, TensorRT-LLM 커뮤니티에 감사드리며, 파트너십과 피드백을 제공해준 radixArk와 Inferact 팀에도 감사드립니다.
> We also thank the SGLang, vLLM, and TensorRT-LLM communities for upstream collaboration and protocol adoption, and the teams at radixArk and Inferact for their partnership and feedback.

이들의 프로덕션 배포, 코드 기여, 기술적 통찰이 오늘날의 SMG를 만들었습니다.
> Their production deployments, code contributions, and technical insights have shaped what SMG is today.
