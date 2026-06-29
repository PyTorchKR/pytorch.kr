---
layout: blog_detail
title: "TokenSpeed-Kernel: 멀티 실리콘 LLM 추론을 위한 이식 가능한 API와 고성능 커널"
author: AMD Triton Team, TokenSpeed Team
ext_author: Junghwan Park (박정환)
category: ["pytorch.org", "translation"]
date: 2026-06-25 12:00:00
org_title: "TokenSpeed-Kernel: Portable APIs and High-Performance Kernels for Multi-Silicon LLM Inference"
org_link: https://pytorch.org/blog/lightseek-tokenspeed-kernel/
---

![TokenSpeed-Kernel: 멀티 실리콘 LLM 추론을 위한 이식 가능한 API와 고성능 커널 / TokenSpeed-Kernel: Portable APIs and High-Performance Kernels for Multi-Silicon LLM Inference](/assets/blog/2026-06-25-lightseek-tokenspeed-kernel/00-hero.png){:style="width:100%"}

### TL;DR

TokenSpeed-kernel은 LLM 추론에서 발생하는 백엔드(backend) 복잡성을 해결하기 위해 설계된 독립형(standalone) 오픈소스 서브시스템입니다. 깔끔한 계층형(layered) API와 레지스트리(registry) 시스템을 도입하여, 고수준 런타임(runtime)을 저수준의 하드웨어별 코드로부터 분리합니다.
> The TokenSpeed-kernel is a standalone, open-source subsystem designed to solve backend complexity in LLM inference. It introduces a clean, layered API and registry system that decouples the high-level runtime from low-level, hardware-specific hardware code.

이번 글에서는 TokenSpeed-kernel을 기술적으로 분석하고, 이것이 멀티 실리콘(multi-silicon) LLM 추론을 위한 고성능 커널을 다루는 개발자에게 어떻게 도움이 되는지 보여드립니다.
> In this blog, we provide a technical breakdown of the TokenSpeed-kernel and show how it helps developers work with high-performance kernels for multi-silicon LLM inference.

## 들어가며 / Introduction

LLM 모델과 추론 하드웨어는 놀라운 속도로 발전하고 있습니다. 이러한 모델을 효율적으로 서빙하는 일은 더 이상 빠른 어텐션(attention) 커널이나 MoE 커널 하나를 찾는 문제가 아닙니다. 현대의 추론 엔진은 런타임을 특수 케이스의 미로로 만들지 않으면서도, 여러 모델·양자화(quantization) 형식·GPU 세대·벤더(vendor) 백엔드를 빠르게 오갈 수 있어야 합니다. 그러한 API는 플랫폼에 종속되지 않고 솔루션에도 종속되지 않습니다.
> LLM models and inference hardware are evolving at astonishing speed. Serving those models efficiently is no longer just a question of finding one fast attention or MoE kernel; modern inference engines need to move quickly across models, quantization formats, GPU generations, and vendor backends without turning the runtime into a maze of special cases. Those APIs are platform-agnostic and solution-agnostic.

이것이 바로 [**TokenSpeed-kernel**](https://github.com/lightseekorg/tokenspeed/tree/main/tokenspeed-kernel)의 동기입니다. *구조화된 유연성을 극대화하는 API*를 위해 깔끔한 계층형 설계를 제공하는 것입니다. 커널-런타임 인터페이스는 범용적으로 유지되는 한편, 커널 개발자는 각 플랫폼에 맞게 깊이 특화할 수 있을 만큼 충분한 구조를 갖게 됩니다.
> This is the motivation behind [**TokenSpeed-kernel**](https://github.com/lightseekorg/tokenspeed/tree/main/tokenspeed-kernel): provide a clean layered *API for maximal structured flexibility*. The kernel-runtime interface stays generic, while kernel developers get enough structure to specialize deeply for each platform.

이 설계가 실제로 어떻게 동작하는지 보여주기 위해 GPT-OSS를 구체적인 예시로 사용합니다. 런타임은 플랫폼과 무관하게 동일한 공개 TokenSpeed-kernel API를 호출하며, AMD와 NVIDIA 경로는 그 API 뒤에 꽂히는(pluggable) 커널로부터 성능을 얻습니다. AMD GPT-OSS 120B의 경우, 이 방식은 Gluon 커널을 사용해 최고 수준의 성능에 도달하며, 계층화가 백엔드 성능을 희생시키지 않음을 보여줍니다.
> We use GPT-OSS as a concrete example to showcase this design in practice. The runtime calls the same public TokenSpeed-kernel APIs regardless of platform; AMD and NVIDIA paths get their performance from pluggable kernels behind those APIs. For AMD GPT-OSS 120B, this approach reaches top-of-the-line performance using Gluon kernels, showing that the layering does not trade away backend performance.

그 결과 역할 분담이 명확해집니다.
> The result is a clear division of focus:

- TokenSpeed 런타임은 모델 실행, 스케줄링 메타데이터, 페이지 테이블(page table), 라우팅(routing) 상태를 담당합니다.
- TokenSpeed-kernel은 연산자(operator) API, 백엔드 등록, 선택, 수치(numerics), 벤치마킹, 프로파일링을 담당합니다.
- 플랫폼별 성능 작업은 모델 코드 곳곳에 흩어지지 않고, 플랫폼별 커널 안에 국소화됩니다.

> - TokenSpeed runtime owns model execution, scheduling metadata, page table, and routing state;
> - TokenSpeed-kernel owns operator APIs, backend registration, selection, numerics, benchmarking, and profiling;
> - platform-specific performance work stays localized in platform-specific kernels, not scattered through model code.

이렇게 깔끔하게 분리한 덕분에, TokenSpeed-kernel을 TokenSpeed에 얽혀 있는 구성 요소로서뿐만 아니라, 단독으로 설치해 사용할 수 있는(전체로든, 커널별로 따로따로든) 독립형 패키지로 공개하는 것도 가능해졌습니다. 목표는 이 커널 패키지가 더 넓은 생태계에도 유용하게 쓰이는 것입니다. 즉, 범용 공개 인터페이스를 갖춘 *이식 가능하고 고성능인 커널의 멀티 실리콘 모음*입니다. 여기에는 뒤에서 다룰 Gluon 커널도 포함되며, AMD는 생태계의 모든 구성원을 지원합니다. 건강한 생태계는 AMD에게도, 커뮤니티에게도 좋기 때문입니다.
> The clean separation has also made it possible to publish TokenSpeed-kernel as standalone packages that can be installed and used on their own (either as a whole or separately for different kernels), not only as an intertwined TokenSpeed component. The goal is for the kernel packages to be useful to the broader ecosystem as well: *a multi-silicon collection of portable and performant kernels* with a generic public surface. This includes the Gluon kernels we will discuss later, as AMD supports everyone in the ecosystem–a healthy ecosystem is good for AMD and the community.

## 현대 추론에서의 커널 / Kernels in Modern Inference

커널은 서빙 스택의 속도를 좌우합니다. 어텐션, MoE 라우팅, 전문가(expert) GEMM, 통신, 양자화, 샘플링이 모두 커널 위에서 실행되며, 이 커널들이 시스템 전체의 지연 시간(latency), 처리량(throughput), 하드웨어 효율을 결정합니다.
> Kernels decide whether a serving stack is fast or slow. Attention, MoE routing, expert GEMMs, communication, quantization, and sampling all run on kernels, and those kernels set the latency, throughput, and hardware efficiency of the whole system.

어려운 점은 "가장 좋은 커널"이 좀처럼 고정된 답이 아니라는 것입니다. 이는 모델 아키텍처, 텐서 형상(shape), 양자화 형식, GPU 세대, 벤더 라이브러리 가용성, 배포 제약, 그리고 그 호출이 디코드(decode) 트래픽을 처리하는지 프리필(prefill) 트래픽을 처리하는지에 따라 달라집니다. 시간이 지나면서 엔진은 이 모든 것을 감당하기 위해 여러 경로를 쌓아 올립니다. 내장(in-tree) 커널, 벤더 라이브러리 래퍼(wrapper), 실험적 커널, 아키텍처별 빠른 경로(fast path), 그리고 과거에 쓰던 폴백(fallback)이 그것입니다. *명확한 커널 시스템과 그것을 둘러싼 단단한 경계가 없으면, 백엔드 선택 로직이 모델 코드와 런타임 코드로 새어 나옵니다.*
> The hard part is that “the best kernel” is rarely a fixed answer. It depends on the model architecture, tensor shape, quantization format, GPU generation, vendor library availability, deployment constraints, and whether a call is serving decode or prefill traffic. Over time, engines accumulate paths to cover all of that: in-tree kernels, vendor library wrappers, experimental kernels, architecture-specific fast paths, and historical fallbacks. *Without a clear kernel system and a hard boundary around it, backend selection logic leaks into model code and runtime code.*

이 누수는 비용이 큽니다. 새 모델을 추가하려면 관련 없는 런타임 경로를 건드려야 할 수 있습니다. 새 실리콘 타깃(silicon target)을 추가하려면 모델 계층 곳곳에 디바이스 검사를 끼워 넣어야 할 수 있습니다. 모델 동작, 런타임 디스패치(dispatch), 백엔드 선택, 커널 구현 세부사항이 불분명한 경계 뒤에 뒤엉키면서 커널 개발은 더 어려워집니다.
> That leakage is costly. Adding a new model can require touching unrelated runtime paths. Adding a new silicon target can mean threading device checks through model layers. Kernel development becomes harder because model behavior, runtime dispatch, backend selection, and kernel implementation details are intertwined behind an unclear boundary.

TokenSpeed-kernel은 이러한 복잡성을 한곳에 모아 두도록 설계되었습니다.
> TokenSpeed-kernel is designed to keep that complexity in one place.

## 설계 원칙 / Design Principles

이 커널 시스템은 세 가지 실용적 원칙을 중심으로 만들어졌습니다.
> The kernel system is built around three practical principles:

첫째, **멀티 실리콘 지원은 근본적이어야 합니다**. 커널 시스템은 하드웨어 검사를 여기저기 흩어진 조건문으로 다루는 대신, 플랫폼의 능력을 직접 이해해야 합니다. 같은 연산이라도 실리콘 타깃마다 여러 해법을 가질 수 있으며, 이 모두가 하나의 선택 시스템 안에서 경쟁해야 합니다.
> First, **multi-silicon support has to be fundamental**. The kernel system should understand platform capabilities directly, instead of treating hardware checks as scattered conditionals. The same operation may have multiple solutions for different silicon targets; all should compete through one selection system.

둘째, **이식성(portability)과 성능은 공존해야 합니다**. 새 모델은 가능한 한 빨리 여러 실리콘 타깃에서 돌아갈 이식 가능한 경로를 필요로 하고, 그다음에 점진적으로 더 고도로 최적화된 커널을 받아들일 수 있습니다. TokenSpeed-kernel은 성능 중심의 선택지(AMD를 위한 Gluon, NVIDIA를 위한 CuteDSL, 그리고 적절한 경우 벤더 래퍼)와 함께 이식 가능한 Triton 경로를 나란히 유지합니다.
> Second, **portability and performance should coexist**. A new model needs a portable path to run on different silicon targets as quickly as possible, then can gradually pick up more highly optimized kernels. TokenSpeed-kernel keeps portable Triton paths alongside performance-focused choices: Gluon for AMD, CuteDSL for NVIDIA, and vendor wrappers where they are the right tool.

셋째, **빠른 커널 반복(iteration)에는 가드레일(guardrail)이 필요합니다**. 아이디어에서 도입까지의 경로가 짧을 때 커널 개발은 빠르게 진행됩니다. TokenSpeed-kernel은 가벼운 의존성, 독립 실행형 벤치마크, 그리고 선택된 커널을 가시화하는 프로파일링으로 그 반복 주기를 짧게 유지합니다. 같은 구조는 AI 에이전트를 위한 커널 개발에도 더 명확한 작업 경계를 제공합니다. 모델 코드를 다시 손대지 않고도 커널을 시도하고, 검증하고, 벤치마크하고, 등록할 수 있습니다. 또한 TokenSpeed-kernel은 빌드를 복잡하게 만들거나 반복을 막는 의존성을 적극적으로 재검토하여, 필요할 때 줄이거나 격리합니다.
> Third, **fast kernel iteration needs guardrails**. Kernel development moves quickly when the path from idea to adoption is short. TokenSpeed-kernel keeps that loop tight with lean dependencies, standalone benchmarks and profiling that makes selected kernels visible. The same structure gives kernel development for AI agents a clearer work boundary: try a kernel, verify it, benchmark it, and register it without reshaping model code. TokenSpeed-kernel also actively revisits dependencies that complicate builds or block iteration, trimming or isolating them when needed.

이러한 원칙은 계층형 설계로 이어집니다.
> These principles lead to a layered design.

## 계층형 커널 시스템 / The Layered Kernel System

높은 수준에서 보면 계층형 커널 시스템은 다음 다이어그램과 같습니다. 위에서 아래로, 이 스택은 런타임이 *무엇을* 요청하는지와 각 백엔드가 그것을 *어떻게* 실행하는지를 분리합니다. 런타임은 범용 공개 API를 통해 진입하고, 선택기(selector)는 그 요청을 호환 가능한 커널에 매핑합니다.
> At a high level, the layered kernel system is shown in the following diagram. From top to bottom, the stack separates what the runtime asks for from how each backend executes it. The runtime enters through a generic public API, the selector maps that request to a compatible kernel.

![계층형 커널 시스템 / Layered kernel system](/assets/blog/2026-06-25-lightseek-tokenspeed-kernel/01-layered-kernel-system.png){:style="width:100%"}

TokenSpeed-kernel은 LLM 추론에서 지배적인 연산들(어텐션, MoE, GEMM, 통신 등)을 위한 공개 API를 노출합니다. 런타임 코드는 가급적 `mha_prefill`, `mha_decode_with_kvcache`, `moe_apply` 같은 최상위 API를 호출합니다. 이 API들은 플랫폼에도, 솔루션에도 종속되지 않습니다. 런타임 호출은 "AMD 커널"이나 "Triton 커널"을 직접 지정하지 않습니다. 대신 연산자 문제, 즉 텐서, 형식, 모델 특성(trait), 실행 제약을 기술합니다. 그러면 TokenSpeed-kernel이 현재 플랫폼과 등록된 커널 특성을 고려해 구현을 선택합니다.
> TokenSpeed-kernel exposes public APIs for the operations that dominate LLM inference: attention, MoE, GEMM, communication, and so on. Runtime code preferably calls top-level APIs such as `mha_prefill`, `mha_decode_with_kvcache` and `moe_apply`. Those APIs are platform- and solution-agnostic. A runtime call does not directly name “the AMD kernel” or “the Triton kernel.” It describes the operator problem: the tensors, formats, model traits, and execution constraints. TokenSpeed-kernel then considers the current platform and registered kernel traits to select the implementation.

내부적으로 백엔드 구현은 `@register_kernel`을 통해 공유 레지스트리에 스스로를 등록합니다. 등록 시에는 연산자 계열(family)과 모드, 솔루션 이름, 플랫폼 능력 요구사항, 지원하는 텐서 시그니처, 특성, 우선순위를 선언합니다. 런타임에서 선택기는 호환되지 않는 커널을 걸러내고, 남은 후보의 순위를 매긴 뒤, 실행할 호출 가능 객체(callable)를 반환합니다.
> Under the hood, backend implementations register themselves with a shared registry through `@register_kernel`. A registration declares the operator family and mode, solution name, platform capability requirements, supported tensor signatures, traits, and priority. At runtime, the selector filters out incompatible kernels, ranks the remaining candidates, and returns the callable to execute.

이 구조는 TokenSpeed에 동시에 얻기 어려운 두 가지 속성을 부여합니다. 첫째, 모델과 런타임은 이식 가능한 상태로 유지됩니다. 각 GPU 백엔드의 세부사항을 알 필요가 없습니다. 둘째, 커널 계층은 고도로 특화된 상태로 유지됩니다. 커널은 정확한 아키텍처, 데이터 타입, 텐서 형상으로 한정될 수 있습니다.
> This structure gives TokenSpeed two properties that are hard to get at the same time. First, the model and runtime remain portable: they do not need to know the details of each GPU backend. Second, the kernel layer remains highly specialized: a kernel can be gated to a precise architecture, data type, tensor shape.

같은 계층화는 개발도 실용적으로 유지합니다. 모델은 한 플랫폼을 가장 빠르게 가동하기 위한 방법으로 특정 솔루션을 사용하다가, 경로가 여러 실리콘 타깃으로 넓어지면 공개 API로 옮겨 갈 수 있습니다. 개발자가 특정 경로를 테스트하고 싶다면, 디버깅과 벤치마킹을 위해 솔루션이나 커널을 강제로 지정(override)할 수도 있습니다.
> The same layering also keeps development pragmatic. A model can use a specific solution when that is the fastest way to bring one platform online, then move to the public APIs as the path broadens across silicon targets. If a developer wants to test a specific path, they can still force a solution or kernel override for debugging and benchmarking.

## 레지스트리와 선택 메커니즘 / Registry and Selection Mechanism

이 유연성을 떠받치는 메커니즘은 레지스트리-선택 루프입니다. 공개 API는 런타임이 연산자 요청을 기술할 안정적인 방법을 제공합니다. 커널 등록은 각 백엔드가 자신이 안전하고 효율적으로 실행할 수 있는 것을 구조적으로 선언할 방법을 제공합니다. 선택기는 이 둘을 연결합니다.
> The mechanism behind this flexibility is the registry-selection loop. Public APIs give the runtime a stable way to describe an operator request. Kernel registrations give each backend a structured way to declare what it can safely and efficiently run. The selector connects the two.

실제로 레지스트리는 사용 가능한 구현에 대한 단일 진실 공급원(single source of truth)입니다. 등록된 각 커널은 메타데이터로 기술됩니다. 어떤 연산자 계열과 모드를 구현하는지, 어떤 솔루션에 속하는지, 어떤 플랫폼 능력을 요구하는지, 어떤 텐서 형식 시그니처를 지원하는지, 어떤 기능 특성이 일치해야 하는지, 그리고 다른 후보 대비 어떤 우선순위를 가져야 하는지가 그것입니다.
> In practice, the registry is the single source of truth for available implementations. Each registered kernel is described by metadata: which operator family and mode it implements, which solution it belongs to, which platform capabilities it requires, which tensor format signatures it supports, which feature traits must match, and what priority it should have relative to other candidates.

선택은 런타임 요청을 호출 가능 객체로 변환합니다. 공개 API는 연산자 입력과 옵션으로부터 요청을 구성합니다. 어텐션의 경우 데이터 타입, 헤드 차원(head dimension), 페이지 크기, 슬라이딩 윈도우(sliding-window) 동작, 어텐션 싱크(attention sink)를 포함할 수 있습니다. MoE의 경우 가중치 형식, 활성화 타입, 내부 활성화 데이터 타입, 전문가 병렬(expert-parallel) 제약을 포함할 수 있습니다.
> Selection then turns a runtime request into a callable. The public API builds the request from the operator inputs and options. For attention, that can include data type, head dimension, page size, sliding-window behavior, and attention sinks. For MoE, it can include weight format, activation type, internal activation data type, and expert-parallel constraints.

선택기는 등록된 커널을 플랫폼 능력, 형식 시그니처, 특성으로 걸러낸 뒤 남은 후보의 순위를 매깁니다. 고정된 모델, 플랫폼, 데이터 타입, 특성 집합에 대해 선택된 구현은 대개 안정적이므로, TokenSpeed-kernel은 해결된 호출 가능 객체를 캐싱합니다. 개발자는 디버깅과 벤치마킹을 위해 여전히 솔루션이나 정확한 커널을 강제 지정할 수 있지만, 일반적인 실행은 같은 레지스트리 경로를 거칩니다.
> The selector filters registered kernels by platform capability, format signature, and traits, then ranks the remaining candidates. For a fixed model, platform, data type, and set of traits, the selected implementation is usually stable, so TokenSpeed-kernel caches the resolved callable. Developers can still force a solution or exact kernel for debugging and benchmarking, but normal execution goes through the same registry path.

다음의 단순화된 등록 스니펫은 NVIDIA와 AMD에서 GPT-OSS와 관련된 어텐션 경로에 대해 이 메타데이터가 어떤 모습인지 보여줍니다.
> The following simplified registration snippets show what this metadata looks like for GPT-OSS-relevant attention paths on NVIDIA and AMD:

![등록과 선택 / Registration and selection](/assets/blog/2026-06-25-lightseek-tokenspeed-kernel/02-registration-and-selection.png){:style="width:100%"}

### 수치, 벤치마킹, 플러그인 / Numerics, Benchmarking, and Plugins

커널 시스템은 단순한 디스패치 이상입니다. 커널 작성자에게 안전하고 빠른 반복을 위한 작업 흐름, 즉 수치 검증(numerics check), 독립 실행형 벤치마크, 프로파일링 스코프(scope)를 제공합니다. 참조 구현(reference implementation)은 공유된 정확성 목표를 제공하고, 벤치마크는 전체 서버 바깥에서 커널의 타이밍과 리포팅 경로를 제공하며, 프로파일링은 선택된 커널 이름과 핵심 매개변수를 종단 간(end-to-end) 모델 트레이스에서 가시화합니다.
> The kernel system is not just dispatch. It also gives kernel authors a workflow for safe, fast iteration: numerics checks, standalone benchmarks, and profiling scopes. Reference implementations provide a shared correctness target, benchmarks give kernels a timing and reporting path outside the full server, and profiling makes selected kernel names and key parameters visible in end-to-end model traces.

같은 경계는 트리 밖(out-of-tree) 플러그인도 지원합니다. 플러그인은 같은 데코레이터(decorator)를 통해 커널을 등록하고, 자체 우선순위를 부여하며, 내장 구현과 나란히 일반적인 선택 과정에 참여합니다. 이렇게 하면 코어 패키지는 깔끔하게 유지되면서도, 하드웨어 벤더, 연구자, 배포 팀이 전체 시스템을 포크(fork)하지 않고도 특화된 커널을 가져올 여지가 남습니다.
> The same boundary supports out-of-tree plugins. A plugin registers kernels through the same decorator, assigns its own priority, and participates in normal selection alongside in-tree implementations. This keeps the core package clean while leaving room for hardware vendors, researchers, and deployment teams to bring specialized kernels without forking the entire system.

일상적인 커널 개발에서는 이러한 사용성(ergonomics)이 디스패치만큼이나 중요합니다. 이것이 바로 이 패키지를 pip로 설치 가능하고 의존성을 신중하게 관리하도록 유지하는 이유이기도 합니다. 특화된 커널은 설치, 검증, 벤치마크, 교체가 쉬워야 합니다.
> For day-to-day kernel development, these ergonomics matter as much as dispatch. They are also why the package is kept pip-installable and dependency-conscious: specialized kernels should be easy to install, verify, benchmark, and replace.

이 작업 흐름을 쉽게 사용할 수 있도록, TokenSpeed-kernel은 주요 개발 작업을 위한 CLI와 프로그래밍 인터페이스를 모두 제공하며, 아래와 같이 수치 검증과 독립 실행형 벤치마킹을 포괄합니다. 이는 CI 작업이나 커스텀 튜닝 파이프라인에서 사용할 수 있습니다. 이 도구들은 별도의 일회성 하니스(harness)가 아닙니다. 서빙이 커널 선택에 사용하는 것과 동일한 레지스트리 메타데이터를 재사용합니다. 따라서 등록된 커널은 참조 구현과 대조해 검증하고, 표준 또는 커스텀 형상에서 측정하고, 선택적으로 프로파일링한 뒤, 그 능력과 특성이 런타임 요청과 일치할 때 자동으로 선택될 수 있습니다.
> To make this workflow easy to use, TokenSpeed-kernel provides both CLIs and programmatic interfaces for the main development tasks, covering numerics verification and standalone benchmarking as shown below. They can be used in CI jobs, or custom tuning pipelines. These tools are not separate one-off harnesses: they reuse the same registry metadata that serving uses for kernel selection, so a registered kernel can be verified against a reference implementation, measured on standard or custom shapes, optionally profiled, and then selected automatically when its capabilities and traits match the runtime request.

![수치 검증 및 벤치마킹 CLI / Numerics and benchmarking CLI](/assets/blog/2026-06-25-lightseek-tokenspeed-kernel/03-numerics-benchmarking-cli.png){:style="width:100%"}

## AMD MI355X에서의 GPT-OSS 120B / GPT-OSS 120B on AMD MI355X

GPT-OSS 120B는 이 설계를 검증하기에 좋은 초기 타깃입니다. 단일 GPU에서 여전히 실행할 수 있는 현대적 LLM이기 때문입니다. 덕분에 실험을 실용적으로 유지하면서도, 현재 추론 워크로드에 중요한 커널 시스템의 핵심 부분을 충분히 작동시킬 수 있습니다.
> GPT-OSS 120B is a good initial target for validating this design given it is a modern LLM that can still be run on a single GPU. That keeps experimentation practical while still exercising the parts of the kernel system that matter for current inference workloads.

GPT-OSS는 어텐션과 MoE를 모두 압박합니다. 어텐션 경로는 어텐션 싱크를 갖춘 일반 MHA와 슬라이딩 윈도우 계층 및 풀 어텐션(full-attention) 계층의 혼합을 사용하고, 대규모 AMD 배포에서는 MoE를 위해 MXFP4 전문가 가중치와 FP8 활성화 흐름을 사용합니다.
> GPT-OSS stresses both attention and MoE: its attention path uses regular MHA with attention sinks and a mix of sliding-window and full-attention layers, while its large AMD deployment uses MXFP4 expert weights and FP8 activation flow for MoE.

이것들은 커널 경계가 너무 느슨하면 런타임으로 새어 들어갈 수 있는 종류의 세부사항입니다. TokenSpeed는 이를 공개 API 아래에 가둬 둡니다.
> Those are exactly the kinds of details that can leak into a runtime if the kernel boundary is too loose. TokenSpeed keeps them below the public API:

![GPT-OSS 커널 API 경계 / GPT-OSS kernel API boundary](/assets/blog/2026-06-25-lightseek-tokenspeed-kernel/04-gpt-oss-kernel-api-boundary.png){:style="width:100%"}

모델 코드는 MI355X 아키텍처 세부사항, [MXFP4](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) 스케일을 [CDNA4](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)(MI355X의 아키텍처)에 맞게 어떻게 배치해야 하는지, 또는 특정 프리필/디코드 어텐션 케이스에서 어떤 AMD 커널이 가장 빠른지를 알 필요가 없습니다. 공개 API에 올바른 텐서와 메타데이터를 전달하기만 하면 됩니다.
> The model code does not need to know MI355X architecture details, how [MXFP4](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) scales should be arranged for [CDNA4](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf) (MI355X’s architecture), or which AMD kernel is fastest on a specific prefill/decode attention case. It only needs to pass the right tensors and metadata to the public API.

### AMD 커널 경로로서의 Gluon / Gluon as the AMD Kernel Path

이 글에서 다루는 AMD 경로의 경우, 성능에 중요한 어텐션 및 MoE 커널은 Gluon으로 구현됩니다. Gluon은 Triton 계열 DSL로, 성능을 위한 명시적 제어를 노출하면서도 블록 수준(block-level) 프로그래밍의 단순함을 유지합니다. 자세한 내용은 Triton 콘퍼런스 발표 [“Gluon Tile Based GPU Programming with Low level Control”](http://youtube.com/watch?v=KqeI23SpJx8)을 참고하세요.
> For the AMD path discussed in this post, the performance-critical attention and MoE kernels are implemented in Gluon, a Triton-family DSL exposing explicit controls for performance, yet still maintaining the simplicity of block-level programming. See the [“Gluon Tile Based GPU Programming with Low level Control” Triton conference talk](http://youtube.com/watch?v=KqeI23SpJx8) for details.

AMD MI355X의 경우, Gluon은 커널 작성자에게 CDNA4 기능에 대한 직접 접근을 제공합니다. 비동기 복사(async copy), 공유 메모리 레이아웃, FP8/MXFP 형식을 위한 스케일드 MFMA(AMD 매트릭스 코어용) 연산, 그리고 효율적인 버퍼/전역 메모리 연산이 그것입니다. 이 모든 기능은 숨겨진 컴파일러 최적화가 아니라 명시적 프로그래밍 기본 요소(primitive)입니다. 커널 작성자는 메모리 접근 방식을 기술하기 위해 단순한 `BlockedLayout`이나 범용 `DistributedLinearLayout` 같은 레이아웃을 고를 수 있고, 공유 메모리의 뱅크 충돌(bank conflict)을 피하기 위해 `SwizzledSharedLayout`이나 `PaddedSharedLayout`으로 공유 메모리를 할당할 수 있으며, `AMDMFMALayout`을 통해 AMD 매트릭스 코어 레이아웃을 선택할 수 있습니다. AMD Gluon 모듈은 하드웨어에 밀접하게 대응되는 연산들을 노출하며, 여기에는 `mfma`, `mfma_scaled`, `buffer_load`, `buffer_store`, 그리고 공유 메모리로의 비동기 전역/버퍼 로드가 포함됩니다.
> For AMD MI355X, Gluon gives kernel authors direct access to CDNA4 features such as async copies, shared-memory layouts, and scaled MFMA (for AMD matrix core) operations for FP8/MXFP formats, and efficient buffer/global memory operations. All of those features are explicit programming primitives rather than hidden compiler optimizations: Kernel authors can choose layouts such as simple `BlockedLayout`, or generic `DistributedLinearLayout` to describe how to access memory; allocate shared memory with `SwizzledSharedLayout` or `PaddedSharedLayout` to avoid bank conflict in shared memory; select AMD matrix-core layouts through `AMDMFMALayout`. The AMD Gluon modules expose operations that map closely to the hardware, including `mfma`, `mfma_scaled`, `buffer_load`, `buffer_store`, and async global-or-buffer loads into shared memory.

Gluon은 또한 소프트웨어 파이프라이닝(software pipelining)을 암묵적인 컴파일러 변환이 아니라 커널의 명시적인 일부로 만듭니다. 커널은 여러 개의 공유 메모리 버퍼를 할당하고, 앞으로 쓸 텐서 타일에 대해 비동기 로드를 발행하고, `async_wait`로 그 타일이 언제 보이게 될지 제어한 뒤, 서로 다른 스케줄을 위해 버퍼를 순환시킬 수 있습니다. 이 수준의 제어는 디코드 단계 커널에서 특히 중요합니다. 디코드 단계의 성능은 메모리 지연을 숨기고 파이프라인 세부사항을 TokenSpeed 런타임으로 밀어 넣지 않으면서 매트릭스 코어를 바쁘게 유지하는 데 달려 있기 때문입니다.
> Gluon also makes software pipelining an explicit part of the kernel rather than an implicit compiler transformation. A kernel can allocate multiple shared-memory buffers, issue asynchronous loads for future tensor tiles, and use `async_wait` to control when those tiles become visible, and then rotate through the buffers for different schedules. This level of control is especially important for decode-phase kernels, where performance depends on hiding memory latency and keeping matrix cores busy without pushing pipeline details into the TokenSpeed runtime.

### 어텐션 / Attention

AMD 경로는 GPT-OSS가 필요로 하는 어텐션 변형들에 대해 CDNA4 Gluon 커널을 등록합니다. 프리필과 페이지 단위 디코드(paged decode)를 포함하며, 슬라이딩 윈도우 사용 여부, 어텐션 싱크 사용 여부 등 서로 다른 변형을 위한 추가 옵션을 갖습니다. 등록 특성이 이러한 선택을 명시적으로 만들어 주므로, 런타임은 여전히 MHA를 요청하는 한편 커널 시스템은 일치하는 Gluon 구현을 선택합니다.
> The AMD path registers CDNA4 Gluon kernels for attention variants GPT-OSS needs: prefill and paged decode, with extra options for different variants like whether using sliding-window, whether using attention sinks, etc. The registration traits make those choices explicit, so the runtime still asks for MHA while the kernel system chooses the matching Gluon implementation.

이 커널 구현은 타일드 QK/PV, 온라인 소프트맥스(online softmax) 같은 표준 어텐션 기법을 사용합니다. 또한 행렬 곱을 위한 매트릭스 코어, 소프트맥스를 위한 팩드 연산(packed math) 명령어, K와 V 타일을 로드하기 위한 버퍼 로드 명령어 같은 CDNA4 고유 기능을 사용합니다.
> The kernel implementation uses standard attention techniques such as tiled QK/PV and online softmax. It also uses CDNA4-specific features such as matrix cores for matrix multiply, packed math instructions for softmax, and buffer load instructions for loading K and V tiles.

![Gluon 어텐션 커널 스니펫 / Gluon attention kernel snippet](/assets/blog/2026-06-25-lightseek-tokenspeed-kernel/05-gluon-attention-kernel-snippet.png){:style="width:100%"}

이 커널은 LLM에서의 인과적 프리필(causal prefill)이라는 워크로드 특성을 한층 더 활용하며, XCD 전반에 워크로드를 균형 있게 유지하기 위한 특수 스케줄링 로직을 갖춘 새로운 퍼시스턴트 커널(persistent kernel)을 설계합니다.
> The kernel further exploits the workload characteristics of causal prefill in LLMs and designs a new persistent kernel with special scheduling logic to keep workload balanced across XCDs.

![어텐션 퍼시스턴트 스케줄러 / Attention persistent scheduler](/assets/blog/2026-06-25-lightseek-tokenspeed-kernel/06-attention-persistent-scheduler.png){:style="width:100%"}

현재 Gluon 어텐션 구현은 측정된 15개의 GPT-OSS 프리필 형상 중 14개에서 가장 빠른 MI355X 백엔드입니다. 전체 그리드에 걸쳐, Triton 기준선(baseline)보다 1.4~2.3배 빠릅니다. 또한 AITER를 벤더 솔루션으로 통합하여 프리필 커널을 평가했습니다. 이 환경에서 AITER는 BF16 프리필 케이스를 자신의 [CK](https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel) 기반 MHA 경로로 디스패치하며, 패키지 내 Triton 폴백을 갖춥니다. AITER와 비교하면 Gluon은 1.1~1.3배의 성능 향상을 제공합니다.
> The current Gluon attention implementation is the fastest evaluated MI355X backend on 14 of the 15 measured GPT-OSS prefill shapes. Across the full grid, it is 1.4-2.3x faster than the Triton baseline. We also evaluate the prefill kernel by integrating AITER as a vendor solution. In this environment, AITER dispatches the BF16 prefill case to its [CK](https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel)-backed MHA path, with an in-package Triton fallback. Compared to AITER, Gluon provides a 1.1-1.3x performance uplift.

![어텐션 프리필 벤치마크 / Attention prefill benchmark](/assets/blog/2026-06-25-lightseek-tokenspeed-kernel/07-attention-prefill-benchmark.png){:style="width:100%"}
*단일 MI355X(CDNA4) GPU에서 GPT-OSS 120B의 어텐션 프리필 처리량: 형상은 bf16 Q/K/V, head_dim = 64, Q 헤드 64개, KV 헤드 8개, 어텐션 싱크가 활성화된 풀 인과적 프리필을 사용합니다. 시퀀스 길이는 1K/4K/8K이며 Q/K/V 모두 동일하고, 배치 크기는 1/2/4/8/16입니다. 막대는 TFLOP/s를 나타내며 높을수록 좋습니다. 타이밍은 어텐션 커널 호출 주위에 HIP 이벤트를 사용해 측정하며, 추가 래퍼 전치(transpose), repeat_interleave, 출력 reshape 등은 제외합니다. TFLOP/s는 인과적 QK+PV 행렬곱 FLOP만 계산하고 인과적 마스킹을 위해 2로 나눕니다. TokenSpeed 커밋 [1492030](https://github.com/lightseekorg/tokenspeed/commit/1492030a2a02d32bc7011645a74d2d691e99c2e6), AITER 버전 0.1.13, ROCm 7.2.1에서 측정. / Attention prefill throughput for GPT-OSS 120B on one MI355X (CDNA4) GPU: Shapes use bf16 Q/K/V, head_dim = 64, 64 Q heads, 8 KV heads, full causal prefill with attention sinks enabled. Sequence lengths are 1K/4K/8K, same for Q/K/V and batch sizes are 1/2/4/8/16. Bars report TFLOP/s; higher is better. Timing uses HIP events around the attention kernel calls excluding extra wrapper transposes, repeat_interleave, output reshape etc. TFLOP/s counts causal QK+PV matmul FLOPs only and divided by 2 for causal masking. Measured at TokenSpeed commit [1492030](https://github.com/lightseekorg/tokenspeed/commit/1492030a2a02d32bc7011645a74d2d691e99c2e6) and AITER version 0.1.13 on ROCm 7.2.1.*

### MoE

MoE는 계층화가 한층 더 유용해지는 지점입니다. GPT-OSS의 MoE 계층은 단일 밀집(dense) 행렬 곱이 아닙니다. 토큰을 전문가로 라우팅하고, 토큰 행을 모으거나(gather) 분배하고(dispatch), 전문가 GEMM을 실행하고, 활성화를 적용하고, top-k 전문가 출력을 라우팅 가중치와 결합(combine)하는 과정을 포함합니다.
> MoE is where the layering becomes even more useful. A GPT-OSS MoE layer is not a single dense matrix multiply. It includes routing tokens to experts, gathering or dispatching token rows, running expert GEMMs, applying the activation, and combining top-k expert outputs with routing weights.

AMD Gluon MoE 경로는 MoE를 두 개의 고립된 GEMM으로 다루는 대신, 이 전체 구조를 중심으로 구축됩니다. 런타임은 하나의 MoE 계층 동작을 보는 한편, 커널 구현은 그 단계들을 함께 튜닝할 자유를 가집니다.
> The AMD Gluon MoE path is built around that full structure rather than treating MoE as two isolated GEMMs. The runtime sees one MoE layer behavior, while the kernel implementation is free to tune those stages together.

프리필의 경우, 핵심 과제는 라우팅된 토큰이 전문가들에 고르지 않게 분포할 때 CDNA4 연산 유닛(Compute Unit, CU)을 바쁘게 유지하는 것입니다. 구현은 비정형 블록 스케줄(ragged block schedule)을 사용해 작업이 실제 전문가 분포를 따라가게 하고, 논리적 토큰 수와 전문가별 슬라이스 크기 양쪽에서 타일 형상을 고릅니다. 큰 프리필 타일은 M/N 또는 N 방향으로 분할할 수 있으며, 스케일드 MFMA 작업이 더 잘 인터리빙(interleave)되도록 작업을 타일 그룹과 XCD 전반에 걸쳐 스위즐(swizzle)합니다. 가중치 경로 역시 CDNA4에 친화적인 MXFP4 스케일 스위즐링과, 메모리 접근에 도움이 되는 경우 호스트에서 미리 셔플(preshuffle)한 가중치를 사용합니다.
> For prefill, the key challenge is keeping CDNA4 Compute Unit (CU) busy when routed tokens are distributed unevenly across experts. The implementation uses ragged block schedules so work follows the actual expert distribution, then chooses tile shapes from both the logical token count and the per-expert slice size. Large prefill tiles can be split along M/N or N, and work is swizzled across tile groups and XCDs so scaled MFMA work is better interleaved. The weight path also uses CDNA4-friendly MXFP4 scale swizzling and host-preshuffled weights where it helps memory access.

디코드는 병목이 다릅니다. 작은 배치는 실행(launch)과 라우팅에 묶이므로, 배치 크기에 따라 선택되는 두 가지 경로를 사용합니다. 가장 작은 배치 크기에서는 [“Better MoE model inference with warp decode” 블로그 글](https://cursor.com/blog/warp-decode)에서 영감을 얻은 워프 디코드(warp-decode) 구현이 top-k 라우팅을 게이트/업 프로젝션(gate/up projection)에 융합(fuse)하여 라우팅과 첫 GEMM이 하나의 실행을 공유하도록 합니다. 여기서의 한계는 점유율(occupancy)입니다. 진행 중인 토큰이 너무 적어 머신을 채우지 못하므로, 이를 다중 버퍼 소프트웨어 파이프라인으로 타일을 공유 메모리에 단계적으로 올리는 협력적(cooperative) 다중 워프 GEMM으로 실행합니다. 한 전문가를 충분히 많은 토큰이 공유해 로드된 가중치 타일이 그들 사이에서 재사용되는 중간(medium) 배치의 경우, 중간 배치 크기를 위한 직접 그룹 GEMM(direct grouped GEMM)으로 전환합니다. 이 경로는 타일을 공유 메모리에 단계적으로 올리되, 파이프라인 대신 단일 버퍼 직접 로드 스케줄을 사용하여, 파이프라인 깊이를 낮은 레지스터·공유 메모리 압력과 맞바꿔 점유율을 높게 유지합니다. 라우팅은 자체적인 작은 융합 커널로 실행됩니다.
> Decode has a different bottleneck: small batches are launch- and routing-bound, so we use two paths selected by batch size. At the smallest batch sizes, the warp-decode implementation, originally inspired by the [“Better MoE model inference with warp decode” blog post](https://cursor.com/blog/warp-decode), fuses top-k routing into the gate/up projection so routing and the first GEMM share a single launch. Here the limit is occupancy: too few tokens are in flight to fill the machine, so we run it as a cooperative multi-warp GEMM that stages tiles through shared memory with a multi-buffer software pipeline. For the medium batch, where enough tokens share an expert that a loaded weight tile is reused across them, we switch to a direct grouped GEMM for medium batch sizes. This path stages tiles through shared memory but uses a single-buffer direct-load schedule instead of a pipeline, trading pipeline depth for the lower register and shared memory pressure that keeps occupancy high; routing runs as its own small fused kernel.

이상의 방법으로, Triton 구현 대비 큰 성능 향상을 달성할 수 있습니다. 가장 작은 배치 크기에서 Gluon 커널은 Triton과 AITER MoE 구현 모두에 대해 큰 향상을 제공합니다. Triton보다 1.7~2.1배, AITER보다 1.1~1.6배 빠릅니다. 중간 디코드 구간에서는 AITER가 약간 앞서지만, Gluon은 가장 빠른 것의 0.9배 이내에 머물면서 Triton보다 1.3~1.4배 빠릅니다. 이곳은 앞으로 계속 개선해 나갈 영역입니다.
> With the above, we are able to achieve great perf uplift against the Triton implementation. At the smallest batch sizes, the Gluon kernels deliver a large uplift over both the Triton and AITER MoE implementations: 1.7 – 2.1× faster than Triton and 1.1 – 1.6× faster than AITER. In the medium decode band, AITER pulls slightly ahead, but gluon stays within 0.9x of the fastest while remaining 1.3 – 1.4× faster than Triton. This is a place we will continue improving.

![MoE 벤치마크 / MoE benchmark](/assets/blog/2026-06-25-lightseek-tokenspeed-kernel/08-moe-benchmark.png){:style="width:100%"}
*단일 MI355(CDNA4) GPU에서 GPT-OSS-120B(MXFP4 가중치, FP8 활성화)의 MoE 지연 시간: Gluon vs AITER vs Triton. 전문가 128개, top-4, D = I = 2880, clamped SwiGLU. M은 MoE 배치 크기(순전파당 토큰 수)이며, "(N experts)"는 해당 M에서 라우팅이 활성화하는 전문가 수입니다. 막대는 전체 MoE 지연 시간(라우팅 + 두 GEMM + SwiGLU + 결합)을 동일한 라우팅에 대한 rocprofv3 GPU 시간으로 보여주며, 모두 torch 참조 대비 cos = 1.0으로 검증되었습니다. 낮을수록 좋습니다. (a) 디코드, M = 1 ~ 16(활성 전문가 4/8/15/31/53개). (b) 프리필, M = 512 ~ 8192(128개 전부 활성). TokenSpeed 커밋 [1492030](https://github.com/lightseekorg/tokenspeed/commit/1492030a2a02d32bc7011645a74d2d691e99c2e6), AITER 버전 0.1.13, ROCm 7.2.1에서 측정. / MoE latency for GPT-OSS-120B (MXFP4 weights, FP8 activations) on one MI355 (CDNA4) GPU: Gluon vs AITER vs Triton. 128 experts, top-4, D = I = 2880, clamped SwiGLU. M is the MoE batch size (tokens per forward); “(N experts)” is the number of experts the routing activates at that M. Bars show full-MoE latency (routing + both GEMMs + SwiGLU + combine), as rocprofv3 GPU time on identical routing, all validated to cos = 1.0 vs a torch reference; lower is better. (a) Decode, M = 1 to 16 (4/8/15/31/53 experts active). (b) Prefill, M = 512 to 8192 (all 128 active). Measured at TokenSpeed commit [1492030](https://github.com/lightseekorg/tokenspeed/commit/1492030a2a02d32bc7011645a74d2d691e99c2e6) and AITER version 0.1.13 on ROCm 7.2.1.*

커널 변형 전반에 걸쳐 중요한 주제는 동일합니다. 백엔드는 CDNA4 스케일드 MFMA, 소프트웨어 파이프라인된 로드와 연산, 융합된 SwiGLU, FP8 출력 양자화, 바이어스 처리, 스케일 스위즐링, 가중치 미리 셔플링, 비정형 스케줄링을 그러한 선택을 모델 코드로 밀어 넣지 않고도 사용할 수 있다는 것입니다.
> Across kernel variants, the important theme is the same: the backend can use CDNA4 scaled MFMA, software-pipelined loads and compute, fused SwiGLU, FP8-output quantization, bias handling, scale swizzling, weight preshuffling, and ragged scheduling without pushing those choices into model code.

## 멀티 실리콘 지원 / Multi-Silicon Support

위에서는 AMD MI355X에서의 GPT-OSS를 다뤘습니다. 같은 커널 API는 NVIDIA 경로도 지원합니다. 현재 GPT-OSS Blackwell 구성에서 어텐션은 FlashInfer가 노출하는 TensorRT-LLM 래퍼를 통해 `trtllm` MHA 백엔드를 사용하고, MXFP4 MoE는 `flashinfer_trtllm` 솔루션을 사용합니다. 런타임은 여전히 순수하게 `mha_prefill`, `mha_decode_with_kvcache`, `moe_apply`를 호출합니다.
> The above talks about GPT-OSS on AMD MI355X. The same kernel API also supports NVIDIA paths. In the current GPT-OSS Blackwell configuration, attention uses the `trtllm` MHA backend through FlashInfer-exposed TensorRT-LLM wrappers, and MXFP4 MoE uses the `flashinfer_trtllm` solution. The runtime still purely calls `mha_prefill`, `mha_decode_with_kvcache` and `moe_apply`.

따라서 멀티 실리콘 지원은 서로 무관한 두 개의 스택이 아닙니다. AMD와 NVIDIA 지원은 같은 커널 API, 레지스트리, 선택 모델 뒤에 있는 형제 구현(sibling implementation)입니다. 플랫폼별 커널은 각 실리콘 타깃에 가용한 최선의 백엔드를 사용할 수 있는 한편, TokenSpeed 런타임은 모델에 대해 일관된 실행 경로를 유지합니다.
> Multi-silicon support is therefore not two unrelated stacks. AMD and NVIDIA support are sibling implementations behind the same kernel API, registry, and selection model. Platform-specific kernels can use the best available backend for each silicon target, while the TokenSpeed runtime keeps a consistent execution path for the model.

## 종단 간 성능 / End-to-end performance

아래 그림은 AMD MI355X에서 측정한 GPT-OSS 120B 출력 처리량 성능을 보여줍니다. 두 가지 TokenSpeed 구성을 비교합니다. 원래의 이식 가능한 Triton 기반 어텐션 및 MoE 경로와, 최적화된 Gluon 기반 경로입니다. 측정된 20개 지점 전반에 걸쳐, Gluon 기반 경로는 모든 입력/출력 길이와 동시성(concurrency) 설정에서 출력 처리량을 개선합니다. 속도 향상은 이식 가능한 Triton 경로 대비 1.6~3.6배에 이릅니다. 전체적으로, 이 핵심 Gluon 커널들은 AMD MI355X에서 GPT-OSS 120B에 대해 TokenSpeed를 경쟁력 있는 성능으로 끌어올립니다.
> The figure below shows the GPT-OSS 120B output throughput performance measured on AMD MI355X. It compares two TokenSpeed configurations: the original portable Triton-backed attention and MoE path, and the optimized Gluon-backed path. Across the 20 measured points, the Gluon-backed path improves output throughput at every input/output length and concurrency setting. The speedups range from 1.6x to 3.6x over the portable Triton path. Overall, these key Gluon kernels bring TokenSpeed to competitive performance for GPT-OSS 120B on AMD MI355X.

![종단 간 출력 처리량 / End-to-end output throughput](/assets/blog/2026-06-25-lightseek-tokenspeed-kernel/09-end-to-end-output-throughput.png){:style="width:100%"}
*단일 MI355X(CDNA4) GPU에서 GPT-OSS-120B의 종단 간 출력 처리량: TokenSpeed Triton(어텐션 및 MoE) 백엔드 vs TokenSpeed Gluon 백엔드. 벤치마크는 [amd/gpt-oss-120b-w-mxfp4-a-fp8](https://huggingface.co/amd/gpt-oss-120b-w-mxfp4-a-fp8)을 TokenSpeed의 OpenAI 호환 HTTP 서버를 통해 단일 GPU에서 TP 크기 1로 서빙하며, 접두사 캐싱(prefix caching)은 비활성화했습니다. 모든 수치는 랜덤 프롬프트로 수집했으며 [EvalScope](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/index.html)로 측정했습니다. TokenSpeed 커밋 [1492030](https://github.com/lightseekorg/tokenspeed/commit/1492030a2a02d32bc7011645a74d2d691e99c2e6), ROCm 7.2.1에서 측정. 더 자세한 내용은 성능 CI 작업 [perf-gpt-oss-120b-mxfp4-mi35x](https://github.com/lightseekorg/tokenspeed/actions/runs/27936591141/job/82821261346)를 참고하세요. / End-to-end output throughput for GPT-OSS-120B on one MI355X (CDNA4) GPU: TokenSpeed Triton (attention and MoE) backend vs TokenSpeed Gluon backend. The benchmark serves [amd/gpt-oss-120b-w-mxfp4-a-fp8](https://huggingface.co/amd/gpt-oss-120b-w-mxfp4-a-fp8) through the TokenSpeed OpenAI-compatible HTTP server with TP size 1 on a single GPU, with prefix caching disabled. All numbers are collected using random prompts and measured by [EvalScope](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/index.html). Measured at TokenSpeed commit [1492030](https://github.com/lightseekorg/tokenspeed/commit/1492030a2a02d32bc7011645a74d2d691e99c2e6) on ROCm 7.2.1. For more detail, please refer to our performance CI job: [perf-gpt-oss-120b-mxfp4-mi35x](https://github.com/lightseekorg/tokenspeed/actions/runs/27936591141/job/82821261346).*

이 결과는 TokenSpeed-kernel 설계의 역할을 부각합니다. 이러한 성능 향상은 별도의 AMD 전용 GPT-OSS 서빙 경로를 필요로 하지 않았습니다. 대신 AMD 성능은 동일한 공개 어텐션 및 MoE 계약(contract)을 특화된 Gluon 커널로 구현하고, 그 플랫폼과 형상 제약을 등록하고, 요청이 일치할 때 선택기가 그것들로 디스패치하도록 함으로써 얻어졌습니다. 이 계층형 설계는 이식 가능한 기준선을 그대로 유지하면서 최적화 주기를 단축합니다. 개발자는 중요한 프로덕션 형상을 포착하고, 그 형상에 맞게 커널을 특화하고, 동일한 수치 및 벤치마크 도구로 검증하고, 선택 메타데이터를 통해 런타임을 최적화된 구현으로 라우팅할 수 있습니다.
> The result highlights the role of the TokenSpeed-kernel design. These gains did not require a separate AMD-specific GPT-OSS serving path. Instead, AMD performance was acquired by implementing the same public attention and MoE contracts with specialized Gluon kernels, registering their platform and shape constraints, and letting the selector dispatch to them when a request matches. This layered design keeps a portable baseline in place while shortening the optimization cycle: developers can capture important production shapes, specialize kernels for those shapes, validate them with the same numerics and benchmark tools, and route the runtime to the optimized implementation through selection metadata.

나아가, 이 설계 덕분에 AMD에서의 이 최적화된 커널들은 TokenSpeed를 넘어 재사용될 수도 있습니다. AMD 전용 어텐션 및 MoE 커널을 TokenSpeed 런타임과 분리하여 [tokenspeed-kernel-amd](https://pypi.org/project/tokenspeed-kernel-amd/)로 공개했으므로, 다른 추론 엔진도 전체 TokenSpeed 서빙 스택에 의존하지 않고 이를 채택할 수 있습니다. 이는 [vLLM에 채택](https://github.com/vllm-project/vllm/pull/46742)되었습니다.
> Moreover, benefit from this design, these optimized kernels on AMD can also be reused beyond TokenSpeed. We released the AMD-specific attention and MoE kernels as [tokenspeed-kernel-amd](https://pypi.org/project/tokenspeed-kernel-amd/), separate from the TokenSpeed runtime, so other inference engines can adopt them without taking a dependency on the full TokenSpeed serving stack. It has been [adopted by vLLM](https://github.com/vllm-project/vllm/pull/46742).

## 맺으며 / Conclusion

TokenSpeed-kernel은 커널을 숨겨진 빠른 경로들의 모음이 아니라 일급(first-class) 서브시스템으로 만들도록 설계되었습니다. 고수준 기능으로는 깔끔한 공개 API, 구조화된 형식 및 특성 메타데이터, 중앙집중식 등록과 선택, 이식 가능하면서도 특화된 구현 경로, 플러그인 지원이 있습니다. 이 모두가 확정된 것은 아니며, 이를 검증하고 개선하는 작업을 활발히 진행 중입니다.
> TokenSpeed-kernel is designed to make kernels a first-class subsystem rather than a collection of hidden fast paths. Its high-level features include a clean public API, structured format and trait metadata, centralized registration and selection, portable and specialized implementation paths, and plugin support. Not all of them have been finalized; we are actively working on validating and improving them.

이점은 단지 더 깔끔한 코드만이 아닙니다. 새로운 하드웨어 지원이 도입되는 방식을 바꿉니다. 이 설계에서는 NVIDIA GPU와 AMD GPU가 모두 일급(first-party) 타깃입니다. AMD에서의 GPT-OSS 120B는 이 모델이 실제로 어떻게 동작하는지 보여줍니다. 이는 추론이 모델·형식·GPU 세대 전반에 걸쳐 점점 이질적(heterogeneous)으로 되어 가는 상황에서 중요합니다. 더 많은 TokenSpeed 모델이 공개 TokenSpeed-kernel API로 옮겨 갈수록, 같은 메커니즘은 런타임 로직을 중복하거나 전환하지 않고도 그 모델들을 AMD GPU에서 가동하고 계속 개선하기 쉽게 만들어 줄 것입니다.
> The benefit is not only cleaner code. It changes how new hardware support can land. NVIDIA GPUs and AMD GPUs are both first-party targets in this design. GPT-OSS 120B on AMD demonstrates how this model works in practice. That matters as inference becomes more heterogeneous across models, formats, and GPU generations. As more TokenSpeed models move to the public TokenSpeed-kernel APIs, the same mechanism will make it easier to bring them up on AMD GPUs and keep improving them without duplicating/switching runtime logic.

### 감사의 글 / Acknowledgements

이 작업은 PyTorch, Triton, 그리고 서빙 시스템과 GPU 커널의 수준을 계속 끌어올리는 그 밖의 많은 프로젝트를 포함한, 더 넓은 오픈소스 추론 생태계 위에 세워졌습니다.
> This work builds on the broader open-source inference ecosystem, including PyTorch, Triton, and many other projects that continue to raise the bar for serving systems and GPU kernels.

이 노력의 배경이 된 런타임 및 시스템 작업에 대해 TokenSpeed 팀과 [LightSeek Foundation](https://lightseek.org/)에 감사드립니다. 또한 협력과 컴퓨팅 지원을 제공해 준 AMD에 감사드립니다. 덕분에 AMD에서의 GPT-OSS 120B 최적화 작업이 가능했고, 그 혜택을 커뮤니티 전체로 확장할 수 있었습니다.
> We thank the TokenSpeed team and the [LightSeek Foundation](https://lightseek.org/) for the runtime and systems work behind this effort. We also thank AMD for its collaboration and compute support, which made the GPT-OSS 120B on AMD optimization work possible that can extend benefits to the whole community.
