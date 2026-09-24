---
layout: blog_detail
title: "vLLM의 하드웨어에 구애받지 않는 모델"
author: Thomas Parnell (IBM), Thomas Ortner (IBM), Richard Zou (Meta), Harry Mellor (Hugging Face)
ext_author: Junghwan Park (박정환)
category: ["pytorch.org", "translation"]
date: 2026-09-22 12:00:00
org_title: "Hardware-Agnostic Models in vLLM"
org_link: https://pytorch.org/blog/hardware-agnostic-models-in-vllm/
---

![vLLM의 하드웨어에 구애받지 않는 모델 대표 이미지 / Hardware-Agnostic Models in vLLM](/assets/blog/2026-09-22-hardware-agnostic-models-in-vllm/hero.png){:style="width:100%"}

**TL;DR**

최전선(frontier)에서 최고 수준의 성능을 달성하기 위해, vLLM은 내부 구현을 fullgraph torch.compile과 호환되지 않는 방향으로 바꾸고 있습니다. 이 변화는 트리 외부(out-of-tree) 가속기, 구형 GPU, 또는 좀 더 특이한 모델을 중요하게 여기는 사용자에게 영향을 줄 수 있습니다. 이 문제를 해결하기 위해 vLLM에 "HW agnostic"(하드웨어에 구애받지 않는, hardware-agnostic) 레이어 세트를 새로 도입합니다. 이 레이어 덕분에 vLLM은 최대한 빠른 속도로 계속 발전하면서도, 이식성(portability)을 중요하게 여기는 사용자의 요구를 함께 충족할 수 있습니다. NVIDIA H100 GPU에서 HW agnostic 레이어의 전체 토큰 처리량(total token throughput)은 네이티브 구현의 3.4% 이내입니다(최근 모델 3종에 대한 기하 평균 기준).
> To achieve state-of-the-art performance at the frontier, vLLM is changing its internal implementation in ways that make it incompatible with fullgraph torch.compile. This may have consequences for users who care about out-of-tree accelerators, older GPUs, or more exotic models. To address this, we are introducing a new set of “HW agnostic” layers in vLLM. These layers will ensure vLLM can continue to move at the speed of light, while at the same time meeting the needs of users who care about portability. On NVIDIA H100 GPUs, the HW agnostic layers achieve total token throughput within 3.4% of the native implementation (geometric mean across three recent models).

## 최전선의 vLLM / vLLM at the frontier

vLLM은 다양한 하드웨어에서 다양한 모델을 지원하는 추상화 레이어로 자리매김하며 전례 없는 성공을 거두었습니다. 잘 설계된 추상화 세트와 [최적화 및 융합을 위한 torch.compile](https://vllm.ai/blog/2025-08-20-torch-compile)을 활용한 덕분에, 이 프로젝트는 실제 모델 로직을 정의하는 코드(흔히 "모델 정의(model definitions)"라고 부릅니다)를 비교적 단순하게 유지하면서도 NVIDIA GPU, AMD GPU, Intel XPU, Google TPU, IBM Spyre, Huawei Ascend 등 폭넓은 하드웨어에서 높은 성능을 달성할 수 있었습니다.
> vLLM has achieved unprecedented success by positioning itself as the abstraction layer supporting a wide variety of models on a wide variety of hardware. By using a set of well-designed abstractions and [torch.compile for optimization and fusion](https://vllm.ai/blog/2025-08-20-torch-compile), the project has been able to keep the code defining the actual model logic (commonly known as “model definitions”) relatively simple, whilst still achieving high performance across NVIDIA GPUs, AMD GPUs, Intel XPUs, Google TPUs, IBM Spyre, Huawei Ascend and more.

그러나 가중치가 공개된 최전선(open-weight) 모델의 아키텍처가 빠르게 갈라지면서, 커뮤니티는 기존 추상화 중 일부가 여전히 목적에 맞는지 다시 검토하게 되었습니다. 모델은 갈수록 자체 제작한 레이어와 최적화된 커널을 함께 내놓습니다. 이는 핵심 어텐션 메커니즘에도 해당되어, DeepSeek V4와 Kimi K3는 완전히 다른 접근으로 백만 토큰 컨텍스트를 달성합니다. 이런 모델을 vLLM에 맞추려면 `model_executor/layers`의 공통 레이어로 모델을 구성하고 모델 전체를 fullgraph로 컴파일할 수 있게 유지해야 합니다. 다시 말해 Dynamo가 추적(trace)할 수 있도록 모델을 작성하고, 새 커널마다 가짜(fake) 구현과 올바른 변경(mutation) 어노테이션을 갖춘 torch 라이브러리 연산으로 등록해야 합니다. 이 작업은 모델 개발에 따르는 추가 부담이며, 그 부담은 모델을 추가하는 사람이 집니다. 직접 모델을 가져오는 고급 사용자도 예외가 아닙니다.
> However, the architectures of frontier open-weight models are rapidly diverging, which has led the community to revisit whether some of the existing abstractions are fit for purpose. Models increasingly ship with bespoke layers and optimized kernels. This even extends to the core attention mechanism: DeepSeek V4 and Kimi K3 achieve million-token context via completely different approaches. Fitting one of these into vLLM means composing it from the shared layers in `model_executor/layers` and keeping the whole model fullgraph compilable: written so that Dynamo can trace it, with each new kernel registered as a torch library op with a fake implementation and correct mutation annotations. That work is a tax on model development, and it is paid by whoever adds the model, including the advanced users who bring their own.

동시에 NVIDIA Blackwell GPU와 NVIDIA GB300 NVL72 같은 랙 규모 시스템은 새로운 기능을 활용하고 연산과 통신을 효과적으로 겹치려면(overlap) 세심한 커널 엔지니어링이 필요합니다.
> At the same time, NVIDIA Blackwell GPUs and rack-scale systems like NVIDIA GB300 NVL72 require careful kernel engineering to exploit new features and effectively overlap computation with communication.

이런 흐름 속에서 Claude Code나 OpenAI Codex 같은 코딩 에이전트가 부상하면서 코드 생성이 훨씬 쉬워졌습니다. 특히 이런 에이전트는 특정 하드웨어에서 특정 모델을 위한 최적화를 설계하는 데 매우 효과적입니다. 다만 어떤 변경이 다른 가속기에서 다른 모델의 성능을 떨어뜨리지 않을지 신경 쓸 필요가 없을 때 가장 잘 작동합니다.
> While all this is happening, we have seen the rise of coding agents like Claude Code and OpenAI Codex, which make generating code much easier. In particular, these agents are very effective at designing optimizations for a specific model on specific hardware. However, they work best if they do not need to worry about whether a particular change will make things worse for a different model, on a different accelerator.

이런 추세가 겹치면서, 최신 GPU 하드웨어에서 최고 수준의 성능을 얻으려면 커뮤니티는 vLLM의 기존 추상화 일부를 해체하고자 합니다. 특히 vLLM은 "플랫(flat)" 모델이라고도 부르는 [하드웨어별 모델 정의](https://github.com/vllm-project/vllm/issues/42770)를 유지하기 시작했습니다. 플랫 모델은 torch.compile 대신 맞춤형 융합(custom fusion)과 그 밖의 모델별, 하드웨어별 최적화를 사용합니다. 최근 몇 달간 vLLM에 추가된 새로운 최전선 모델은 모두 이 플랫 모델 정의를 따릅니다. 결정적으로, 모델 정의가 사용하는 기존 레이어와 연산은 torch.compile과 근본적으로 호환되지 않는 방식으로 리팩터링될 가능성이 높습니다.
> These trends come together and mean that, to achieve state-of-the-art performance on the latest GPU hardware, the community would like to dismantle some of the existing abstractions in vLLM. In particular, vLLM is starting to maintain [hardware-specific model definitions](https://github.com/vllm-project/vllm/issues/42770), otherwise known as “flat” models. Rather than using torch.compile, flat models use custom fusions and other model-specific and hardware-specific optimizations. New frontier models added to vLLM in recent months all use this flat model definition. Critically, the existing layers and ops consumed by the model definitions are likely to be refactored in a way that makes them fundamentally incompatible with torch.compile.

이 작업은 vLLM이 최신 GPU 벤치마크에서 경쟁력을 유지하려면 필요합니다. 하지만 구형 GPU나 트리 외부(out-of-tree, OOT) 가속기처럼 다양한 하드웨어에서 다양한 모델을 서빙하려는 사용자에게 vLLM이 계속 도움이 되는 것도 중요합니다.
> This effort is necessary to enable vLLM to stay competitive on the latest GPU benchmarks. However, it is also important that vLLM continues to serve its users who care about serving diverse models on diverse hardware like older GPUs or out-of-tree (OOT) accelerators.

그렇다면 무엇을 할 수 있을까요? 먼저 vLLM이 오늘날 모델 정의를 어떻게 다루는지 살펴보겠습니다.
> So, what can we do about it? Let’s start by reviewing how vLLM handles model definitions today.

## vLLM에서 모델 정의는 어떻게 동작하나요? / How do model definitions work in vLLM?

오늘날 vLLM은 세 가지 방식의 모델 정의를 제공합니다:

1. `vllm/models/` 아래에 있는 새로운 "플랫" 모델
2. `vllm/model_executor/models` 아래에 있는 레거시 모델
3. transformers에서 모델을 가져오는 transformers 모델링 백엔드

> Today, vLLM offers three flavours of model definitions.
>
> 1. The new “flat” models which live under `vllm/models/`
> 2. The legacy models which live under `vllm/model_executor/models`
> 3. The transformers modeling backend, which imports models from transformers.

현재 상태를 개략적으로 그리면 아래와 같습니다.
> A high-level sketch of the current state is shown below.

![vLLM 모델 정의의 현재 상태 / The current state of model definitions in vLLM](/assets/blog/2026-09-22-hardware-agnostic-models-in-vllm/fig1-model-definitions.png){:style="width:100%"}
*그림 1: vLLM 모델 정의의 현재 상태. 세 방식 모두 공통 레이어마다 하나의 구현으로 귀결되며, 여기서는 RowParallelLinear를 예로 들었습니다. SpyreRowParallelLinear는 한 가속기에서 해당 레이어를 재정의하는 트리 외부(OOT) 플러그인입니다. / Figure 1: The current state of model definitions in vLLM. All three flavours resolve to a single implementation of each common layer, shown here for RowParallelLinear. SpyreRowParallelLinear is an out-of-tree plugin overriding that layer on one accelerator.*

모델링 로직이 서로 다른 곳에 있더라도 대부분의 모델은 어텐션, 전문가 혼합(mixture-of-experts), 선형 투영(linear projection), 정규화(norm), 활성화 함수 같은 공통 레이어로 구성됩니다. 위 세 경우 모두에서 이 공통 레이어가 여전히 한 곳에서 구현된다는 점을 이해하는 것이 중요합니다. (1)과 (2)에서는 이 레이어를 `vllm/model_executor/layers`에서 명시적으로 임포트하며, (3)에서는 transformers 모델이 자동으로 융합되고 vLLM 레이어를 쓰도록 재배선(rewire)됩니다. 따라서 모델 정의가 어디에서 오든, 그 바탕을 이루는 레이어 대부분에는 여전히 공통 구현을 사용합니다.
> While the modeling logic may live in different places, most models are composed of common layers like attention, mixture-of-experts, linear projections, norms and activations. It is important to understand that, in all 3 cases above, these common layers are still implemented in a single place. In case (1) and (2) these layers are explicitly imported from `vllm/model_executor/layers`. In case (3), the transformers model gets automatically fused and re-wired to use the vLLM layers. Thus, wherever the model definition is coming from, we are still using a common implementation of the majority of the layers that underpin it.

vLLM의 레이어 구현은 수년에 걸쳐 발전해 왔으며, 지금부터 더 자세히 살펴볼 두 가지 중요한 기능을 제공합니다: (a) torch compile 지원, (b) OOT 확장성(extensibility).
> vLLM’s layer implementations have evolved over several years and offer two important features that we will now discuss in more detail: (a) torch compile support, and (b) OOT extensibility.

fullgraph torch compile은 플랫 모델에서는 쓰이지 않지만, [IBM Spyre](https://github.com/torch-spyre/spyre-inference) 같은 OOT 플러그인에는 여전히 핵심 기능입니다. Spyre는 모델 그래프를 추적하는 데 TorchDynamo에, 그래프를 대상 하드웨어에서 최적으로 실행되는 표현으로 저수준화(lowering)하는 데 TorchInductor에 의존합니다. 결정적으로, torch compile은 vLLM의 transformers 백엔드가 NVIDIA GPU에서 Qwen3 같은 모델에 [네이티브 속도](https://huggingface.co/blog/native-speed-vllm-transformers-backend)를 내는 데에도 꼭 필요한 구성 요소입니다.
> While fullgraph torch compile is not used by the flat models, it remains a critical feature for OOT plugins like [IBM Spyre](https://github.com/torch-spyre/spyre-inference). Spyre relies on TorchDynamo to trace the model graph, and TorchInductor to lower the graph down to representations that run optimally on the target hardware. Crucially, torch compile is also a necessary component for enabling vLLM’s transformers backend to achieve [native speed](https://huggingface.co/blog/native-speed-vllm-transformers-backend) for models like Qwen3 on NVIDIA GPUs.

다만 OOT 플러그인에는 torch compile이 전부가 아닙니다. Spyre 같은 가속기는 최적의 성능을 내기 위해 레이어에 동작을 주입해야 할 때도 있습니다(예: 맞춤형 메모리 레이아웃). vLLM의 레이어는 맞춤 동작을 주입하는 두 가지 메커니즘을 제공합니다. **CustomOp**(플러그인이 forward 함수를 재정의할 수 있게 함)과 **PluggableLayer**(플러그인이 레이어 전체를 재정의할 수 있게 함)입니다. 이런 확장성이 없으면 OOT 플러그인은 레이어 상당수를 직접 다시 구현해야 합니다.
> However, for OOT plugins torch compile is not the whole story. Accelerators like Spyre also occasionally need to inject behaviour into the layers (e.g., custom memory layouts) to achieve optimal performance. vLLM’s layer offers two different mechanisms for injecting custom behaviour: **CustomOp** (which enables the plugin to override the forward function) and **PluggableLayer** (which enables the plugin to override the entire layer). Without this extensibility, OOT plugins would need to re-implement many of the layers themselves.

## 그렇다면 무엇이 문제일까요? / So, what is the problem here?

모델 정의가 세 곳에 나뉘어 있어 꽤 헷갈린다는 점은 차치하더라도, 위 설계에는 더 시급한 문제가 있습니다.
> Aside from the fact that having model definitions in three places is pretty confusing, there is a more pressing issue with the above design.

플랫 모델 작업은 모델 정의와 그 바탕이 되는 레이어 구현을 바꿔서 **torch compile과의 호환성을 깨고** **CustomOp을 통한 확장성 지원을 없애야** 합니다. 그래야 하드웨어별, 모델별 성능 최적화를 더 빠르게 개발할 수 있지만, 몇 가지 우려도 생깁니다.
> The flat model workstream needs to change the model definitions, and their underlying layer implementations, to **break compatibility with torch compile** and **remove support for extensibility via CustomOp**. This will unlock them to move faster on developing hardware-specific and model-specific performance optimizations, but it also raises some concerns.

첫째, OOT 플러그인은 자체 모델 정의와 레이어 세트를 직접 유지 관리해야 할 처지에 놓이게 되어 유지 보수 부담이 커집니다. 새 모델을 지원하려면 transformers, vLLM, 그리고 잠재적으로는 그 모델을 지원하려는 모든 OOT 플러그인에 풀 리퀘스트를 보내야 합니다. 코딩 에이전트가 이 일을 쉽게 만들어 주는 것은 사실이지만, 결국 실질적인 이득 없이 여러 조직에 걸쳐 토큰 예산만 소진하게 됩니다.
> Firstly, it leaves OOT plugins facing the prospect of maintaining their own set of model definitions and layers, creating a large maintenance burden. Supporting a new model will involve making pull requests to transformers, vLLM, and then potentially every OOT plugin that wants to support it. Yes, coding agents make this easier but this will still require burning through token budgets across multiple different organizations for ultimately no real benefit.

둘째, vLLM은 구형 모델이나 특이한 모델을 지원하는 데 점점 더 transformers 백엔드에 의존하고 있습니다. 레거시 모델 정의는 `model_executor/models`에서 적극적으로 제거되고 있으며, 레지스트리 항목은 transformers 모델링 백엔드를 직접 가리키도록 갱신되고 있습니다. torch compile이 가능한 레이어가 없으면 GPU에서 이런 모델의 성능이 크게 떨어질 것입니다.
> Second, vLLM is increasingly relying on the transformers backend to provide support for older or more exotic models. Legacy model definitions are actively being removed from `model_executor/models` and their registry entries updated to point directly at the transformers modeling backend. Without torch compilable layers, performance for these models on GPU will regress significantly.

마지막으로, 플랫 모델과 레이어는 최전선 GPU에 최적화되겠지만 구형 GPU나 소비자용/프로슈머용 GPU까지 지원하리라고는 기대하기 어렵습니다. vLLM 자체 [사용 통계](https://app.hex.tech/019c4540-72b8-7005-9d68-08e0191ac583/app/vLLM-Weekly-Usage-Stats-032Vh7ZNLdI3OI2hNYJaPv/latest)를 보면 사용자층의 상당 부분이 여전히 그런 하드웨어를 쓰고 있습니다. 프로젝트가 이들의 요구도 충족하는 방향으로 발전해야 한다고 생각합니다.
> Finally, while the flat model and layers will be optimized for frontier GPUs, we do not expect them to provide support for older GPUs or consumer/prosumer GPUs. vLLM’s own [usage statistics](https://app.hex.tech/019c4540-72b8-7005-9d68-08e0191ac583/app/vLLM-Weekly-Usage-Stats-032Vh7ZNLdI3OI2hNYJaPv/latest) show that a significant portion of the user base continues to use such hardware. We believe the project should also evolve in a way that meets their needs.

## 해결책은 무엇인가요? / What is our solution?

vLLM 트리 안에 하드웨어에 구애받지 않는 레이어 세트를 만들고 있습니다. 이 레이어의 목표는 다양한 하드웨어에서 다양한 모델을 실행하려는 사용자층을 vLLM이 계속 지원할 수 있게 하는 것입니다.
> We are building a set of hardware-agnostic layers in-tree in vLLM. The aim of these layers is to ensure that vLLM can continue to support its user base that cares about running diverse models on diverse hardware.

하드웨어에 구애받지 않는 레이어는 다음 네 가지 설계 원칙을 따릅니다:

1. **컴파일 가능성**. 모델 정의는 전체 그래프를 torch compile로 컴파일할 수 있습니다. 성능을 위해 컴파일이 필요한 가속기는 지금처럼 계속 컴파일을 사용할 수 있습니다.
2. **확장성**. vLLM의 CustomOp과 PluggableLayer 같은 메커니즘을 유지하여, 필요할 때 OOT 플러그인이 구현을 재정의할 수 있게 합니다.
3. **격리성**. 모델 정의는 하드웨어별 경로에서 쓰는 레이어와 연산으로부터 분리되고 격리된, 자체 레이어와 연산 세트로 만듭니다. 이렇게 하면 두 방향의 개발이 서로를 방해하지 않고 빠르게 진행될 수 있습니다.
4. **이식성**. 모든 레이어와 연산을 네이티브 PyTorch 코드나 Triton, Helion 같은 이식 가능한 DSL로 구현하기 위해 노력합니다. 그러면 이런 프레임워크를 지원하는 모든 가속기에서 모델을 이식할 수 있습니다. 지원하지 않는 가속기는 필요할 때 (2)에 의존할 수 있습니다.

> The Hardware-agnostic layers adhere to the following four design principles:
>
> 1. **Compilable**. The model definitions will be full-graph torch compilable; Accelerators that require compile for performance can continue using it as they do today.
> 2. **Extensible**. We will keep mechanisms like vLLM’s CustomOp and PluggableLayer to ensure that OOT plugins can override the implementation when necessary.
> 3. **Isolated**. The model definitions will be built with their own set of layers and ops that are separate and isolated from the layers and ops used by the hardware-specific paths. This will ensure that development in both directions can move fast without impeding the other.
> 4. **Portable**. We will strive to implement all layers and ops using either native PyTorch code or portable DSLs like Triton and Helion. This will make the models portable across all accelerators that support these frameworks. Those that do not can still rely on (2) when necessary.

우리가 추구하는 설계는 아래와 같습니다:
> The design we are working towards is illustrated below:

![vLLM의 하드웨어에 구애받지 않는 레이어 / Hardware-agnostic Layers in vLLM](/assets/blog/2026-09-22-hardware-agnostic-models-in-vllm/fig2-hw-agnostic-layers.png){:style="width:100%"}
*그림 2: vLLM의 하드웨어에 구애받지 않는 레이어. / Figure 2: Hardware-agnostic Layers in vLLM.*

레거시 모델 정의가 점진적으로 제거되면, 모델은 (NVIDIA, AMD, XPU 등에 각각 다른 구현을 두는 식의) 플랫 방식으로 다시 구현되거나 transformers 백엔드로 대체(fallback)됩니다. 이 두 경우 모두에서 하드웨어에 구애받지 않는 지원을 제공하고자 합니다.
> As the legacy model definitions are gradually removed, models will either be re-implemented in the flat way (e.g., a different implementation for NVIDIA, AMD, XPU etc), or they will fallback to the transformers backend. We intended to offer hardware-agnostic support in both of these cases.

transformers 백엔드에서는 "재배선(rewiring)" 과정이 기존 `model_executor/layers`의 레이어 대신 `model_executor/hw_agnostic`에 있는 새로운 HW agnostic 레이어를 대상으로 하도록 수정했습니다. 이 지원은 (제한된 수의 레이어에 대해) vLLM 메인 브랜치에 [이미 반영](https://github.com/vllm-project/vllm/pull/49458)되었으며, transformers 백엔드로 vLLM을 실행할 때 `USE_HW_AGNOSTIC=1`을 설정하면 사용할 수 있습니다:
> For the transformers backend, we have modified the “rewiring” process to target the new HW-agnostic layers which reside at `model_executor/hw_agnostic`, instead of the existing layers at `model_executor/layers`. This support has [already landed](https://github.com/vllm-project/vllm/pull/49458) in the main branch of vLLM (for a limited number of layers), and can be enabled by setting `USE_HW_AGNOSTIC=1` when running vLLM with the transformers backend:

```sh
USE_HW_AGNOSTIC=1 vllm serve google/gemma-4-31B --model-impl=transformers
```

Gemma 4, Qwen3, Granite 4.2 같은 모델에서 Spyre OOT 플러그인으로 이 새 경로를 검증했습니다. 곧 CI에 HW agnostic 모델을 포함하기 시작하고, 점차 Spyre에서 모델을 서빙하는 기본 경로로 전환할 것입니다.
> We have validated this new pathway using the Spyre OOT plugin for models like Gemma 4, Qwen3, and Granite 4.2. Very soon, we will start to include HW agnostic models in our CI, and gradually switch over to using this as our default pathway for serving models on Spyre.

아직 반영되지는 않았지만, 각 플랫 모델에 대해 HW agnostic 레이어로 모델을 구현하는 새 `model.py`를 제공할 계획입니다. 여러 플랫 모델에서 재사용되는 레이어는 공유 위치(`model_executor/hw_agnostic`)에 두고, 모델별 레이어(예: `DeepSeekV4FlashMLAAttention`)는 `model.py`와 같은 로컬 디렉터리에 두되 위의 네 가지 설계 원칙을 그대로 따릅니다. 한 예로, 현재 리뷰 중인 하드웨어에 구애받지 않는 [DeepSeek V4 PR](https://github.com/vllm-project/vllm/pull/45470)을 확인해 보세요.
> While not yet landed, we plan to provide a new `model.py` for each flat model, that implements the model using the HW-agnostic layers. Layers that are re-used across multiple flat models will reside in a shared place (`model_executor/hw_agnostic`), whereas model-specific layers (e.g., `DeepSeekV4FlashMLAAttention`) will reside in the local directory with the `model.py` but still adhere to the 4 design principles above. As one example of this, please check out the hardware agnostic [DeepSeek V4 PR](https://github.com/vllm-project/vllm/pull/45470) that is currently under review.

## 그렇다면 GPU에서는 성능이 어떨까요? / But, how will it perform on GPUs?

Blackwell, CDNA 4 및 그 이후 세대에서 최고 수준의 성능을 내는 것은 이 모델 정의의 목표가 아님을 강조합니다. 목표는 OOT 가속기, 구형 GPU, 프로슈머급 GPU를 포함한 다양한 하드웨어에서 플랫폼 이식성과 성능 이식성을 달성하는 것입니다.
> We stress that state-of-the-art performance on Blackwell, CDNA 4, and beyond is not the goal of these model definitions. Our aim is to achieve platform and performance portability across diverse hardware, including OOT accelerators, older GPUs, as well as prosumer-grade GPUs.

이 새 경로가 널리 쓰이는 GPU에서 어떻게 동작하는지 평가하기 위해, 최근 모델 몇 가지를 대상으로 NVIDIA H100 GPU에서 실험을 진행했습니다. 그림 3에서 vLLM의 transformers 백엔드를 `USE_HW_AGNOSTIC=0`과 `USE_HW_AGNOSTIC=1`로 실행했을 때의 성능을 비교합니다.
> To evaluate how the new pathway behaves on widely-available GPUs, we ran some experiments on NVIDIA H100 GPUs, for a handful of recent models. We compare the performance of vLLM’s transformers backend using `USE_HW_AGNOSTIC=0` vs. `USE_HW_AGNOSTIC=1` in Figure 3.

보시다시피 HW agnostic 모델은 기반이 되는 레이어와 연산을 오로지 **이식 가능한 구현**만으로 만들었는데도, FlashAttention이나 CUTLASS 같은 CUDA 최적화 라이브러리를 쓰는 네이티브 모델과 비교해 상당히 근접한 성능을 내며, 경우에 따라서는 약간 더 나은 성능을 내기도 합니다.
> As we can see, despite being built solely from **portable implementations** of the underlying layers and ops, HW agnostic models achieve relatively close, and in some cases even slightly better, performance, than the native models that use CUDA-optimized libraries like FlashAttention and CUTLASS.

![H100 GPU에서 HW agnostic 레이어가 성능에 미치는 영향 / Impact of HW Agnostic layers on performance for H100 GPUs](/assets/blog/2026-09-22-hardware-agnostic-models-in-vllm/fig3-h100-performance.png){:style="width:100%"}
*그림 3: H100 GPU에서 HW agnostic 레이어가 성능에 미치는 영향. / Figure 3: Impact of HW Agnostic layers on performance for H100 GPUs.*

## 결론 / Conclusion

프로젝트가 최전선의 성능 엔지니어링 속도를 늦추지 않으면서도 다양한 하드웨어에서 다양한 모델을 계속 지원할 수 있도록, vLLM에 HW agnostic 레이어를 도입합니다. 이 노력이 vLLM이 더 넓은 오픈 소스 생태계의 요구를 계속 충족하는 데 중요하다고 믿습니다. 이 작업을 실현하기 위한 PR을 반영하기 시작했지만 아직 진행 중인 작업(work-in-progress)이며, 여러분의 피드백을 환영합니다.
> We are introducing HW agnostic layers into vLLM to ensure that the project can continue to support diverse models on diverse hardware, without slowing down performance engineering at the frontier. We believe this effort is important for vLLM to continue to serve the needs of the broader open-source ecosystem. While we have started landing PRs to realize this effort, this is still very much a work-in-progress and we welcome any feedback.

자세한 내용은 [RFC](https://github.com/vllm-project/vllm/issues/44219)를 확인하거나 vLLM Slack의 [#hw-agnostic-models](https://vllm-dev.slack.com/archives/C0B8VV3CRC7) 채널을 팔로우하세요. [vLLM.ai](http://vllm.ai/)나 [vLLM GitHub](https://github.com/vllm-project/vllm) 프로젝트 페이지에서 vLLM에 대해 더 알아볼 수도 있습니다.
> For more information, please check out the [RFC](https://github.com/vllm-project/vllm/issues/44219) or follow the slack channel [#hw-agnostic-models](https://vllm-dev.slack.com/archives/C0B8VV3CRC7) on vLLM slack. You can also learn more about vLLM at [vLLM.ai](http://vllm.ai/) or the [vLLM GitHub](https://github.com/vllm-project/vllm) project page.
