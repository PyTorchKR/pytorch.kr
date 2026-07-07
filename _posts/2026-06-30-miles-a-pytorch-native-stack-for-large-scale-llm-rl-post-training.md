---
layout: blog_detail
title: "Miles: 대규모 LLM RL 사후 학습을 위한 PyTorch 네이티브 스택"
author: Miles Team
ext_author: Junghwan Park (박정환)
category: ["pytorch.org", "translation"]
date: 2026-06-30 12:00:00
org_title: "Miles: A PyTorch-Native Stack for Large-Scale LLM RL Post-Training"
org_link: https://pytorch.org/blog/miles-a-pytorch-native-stack-for-large-scale-llm-rl-post-training/
---

## TL;DR

Miles는 대규모 LLM RL 사후 학습(post-training)을 위한 RadixArk의 오픈 소스 프레임워크입니다. 롤아웃(rollout)을 위한 SGLang, 학습을 위한 NVIDIA Megatron-LM, Ray 오케스트레이션(orchestration), PyTorch 네이티브 확장성을 작고 플러그 가능한(pluggable) 트레이너 뒤에 결합하며, 통합된 낮은 정밀도(low-precision) 레시피, MoE를 인식하는 롤아웃/학습 정렬(alignment), 빠른 NVIDIA NCCL/RDMA 가중치 동기화(weight synchronization), 관측 가능성(observability), 장애 허용(fault tolerance)을 기본으로 갖추고 있습니다 — 이를 통해 프런티어(frontier) 규모의 LLM RL을 더 쉽게 구축하고, 재현하고, 운영할 수 있게 합니다.
> Miles is RadixArk's open source framework for large-scale LLM RL post-training. It composes SGLang for rollout, NVIDIA Megatron-LM for training, Ray orchestration, and PyTorch-native extensibility behind a small, pluggable trainer, with unified low-precision recipes, MoE-aware rollout/training alignment, fast NVIDIA NCCL/RDMA weight synchronization, observability, and fault tolerance built in — making frontier-scale LLM RL easier to build, reproduce, and operate.

## 왜 Miles인가? / Why Miles?

강화 학습(reinforcement learning, RL)은 대규모 언어 모델(LLM)을 사후 학습(post-training)하는 데 핵심적인 부분이 되었습니다. 하지만 모델이 더 커지고, 밀집(dense) 구조에서 전문가 혼합(mixture-of-experts, MoE) 구조로 전환되며, 더 분산되고 특화된 하드웨어(예: NVIDIA Blackwell과 Hopper 시리즈) 전반에서 실행됨에 따라, RL 사후 학습은 더 이상 단순한 학습 루프가 아닙니다. 이는 분산 시스템 문제입니다.
> Reinforcement learning has become a central part of post-training large language models. But as models become larger, transition from dense to mixture-of-experts (MoE), and run across more distributed and specialized hardware (e.g. NVIDIA Blackwell and Hopper series), RL post-training is no longer just a training loop. It is a distributed systems problem.

현대적인 LLM RL 프레임워크는 여러 움직이는 부분들을 조율해야 합니다:
> A modern LLM RL framework needs to coordinate several moving pieces:

1. 롤아웃 워커(rollout worker)는 높은 처리량으로 샘플을 생성해야 합니다.
2. 트레이너는 그 샘플들을 효율적으로 소비하고 안정적인 정책 업데이트를 계산해야 합니다.
3. 롤아웃 정책과 학습 정책은 서로 동기화된 상태를 유지해야 합니다.
4. 대규모 MoE 모델은 롤아웃과 학습 전반에서 정렬을 유지해야 하는 라우팅(routing) 동작을 수반합니다.
5. 낮은 정밀도 레시피는 전체 파이프라인에서 일관되게 동작해야 합니다.
6. 장기 실행되는 작업은 처음부터 관측 가능성, 체크포인팅, 장애 허용을 갖춰야 합니다.

> 1. Rollout workers must generate samples at high throughput.
> 2. Trainers must consume those samples efficiently and compute stable policy updates.
> 3. The rollout policy and training policy must stay synchronized.
> 4. Large MoE models introduce routing behavior that must remain aligned across rollout and training.
> 5. Low-precision recipes need to work consistently across the full pipeline.
> 6. Long-running jobs need observability, checkpointing, and fault tolerance from the start.

Miles는 바로 이런 환경을 위해 만들어졌습니다.
> Miles was built for this setting.

Miles는 LLM 사후 학습을 위한 RadixArk의 오픈 소스 강화 학습 프레임워크입니다. 높은 처리량의 롤아웃을 위해 SGLang 위에 네이티브로 구축되었고, 확장 가능한 학습을 위해 Megatron-LM과 깊이 통합되며, 분산 시스템을 오케스트레이션하기 위해 Ray를 사용하고, 스택 전반에서 공통 프로그래밍 및 수치 계층(numerical layer)으로 PyTorch를 유지합니다.
> Miles is RadixArk's open source reinforcement learning framework for LLM post-training. It is built natively on SGLang for high-throughput rollout and integrates deeply with Megatron-LM for scalable training, uses Ray to orchestrate the distributed system, and keeps PyTorch as the common programming and numerical layer throughout the stack.

목표는 단순합니다: 대규모 LLM RL 학습을 더 조합 가능(composable)하고 재현 가능하며 확장하기 쉽게 만들면서도, 연구자와 인프라 팀이 커스터마이징할 수 있을 만큼 핵심 트레이너를 작게 유지하는 것입니다.
> The goal is simple: make large-scale LLM RL training more composable, reproducible, and easier to scale, while keeping the core trainer small enough for researchers and infrastructure teams to customize.

## Miles 아키텍처 / The Miles Architecture

Miles는 '작은 코어, 많은 엣지(edges)' 철학을 따릅니다.
> Miles follows a small-core, many-edges philosophy.

핵심 학습 루프는 의도적으로 작게 유지됩니다. 사용자가 가장 자주 바꾸고 싶어 하는 부분들 — 롤아웃 로직, 보상 계산, 손실 함수, 샘플 필터링, 지표, 학습 루프 훅(hook) — 은 실행 시점에 사용자가 제공하는 Python 모듈을 통해 연결됩니다. 이를 통해 팀은 프레임워크를 포크하지 않고도 새로운 알고리즘과 프로덕션 제약에 맞춰 시스템을 조정할 수 있습니다.
> The core training loop is intentionally compact. The pieces that users most often want to change — rollout logic, reward computation, loss functions, sample filtering, metrics, and training-loop hooks — are attached at launch time through user-supplied Python modules. This lets teams adapt the system to new algorithms and production constraints without forking the framework.

이 작은 코어 아래에서, Miles는 네 가지 주요 시스템을 조합합니다:
> Underneath that small core, Miles composes four major systems:

1. 높은 처리량의 롤아웃 생성을 위한 **SGLang**.
2. 확장 가능한 분산 학습을 위한 **Megatron-LM**.
3. 클러스터 오케스트레이션, 액터(actor) 생명주기, 스케줄링, 감독(supervision)을 위한 **Ray**.
4. 모델, autograd, 분산 프리미티브(primitive), dtype 지원, 확장성, 프로파일링을 위한 **PyTorch**.

> 1. **SGLang** for high-throughput rollout generation.
> 2. **Megatron-LM** for scalable distributed training.
> 3. **Ray** for cluster orchestration, actor lifecycle, scheduling, and supervision.
> 4. **PyTorch** for models, autograd, distributed primitives, dtype support, extensibility, and profiling.

![Miles 아키텍처 다이어그램 / Miles architecture diagram](/assets/blog/2026-06-30-miles-a-pytorch-native-stack-for-large-scale-llm-rl-post-training/architecture.jpeg){:style="width:100%"}

이 조합은 중요합니다. RL 사후 학습은 생성과 학습이 함께 동작해야 하지만, 두 단계는 매우 다른 성능 프로파일을 가집니다: 롤아웃은 메모리 대역폭에 의해 제한되는(memory-bandwidth-bound) 반면(디코딩 중에는 KV 캐시와 매개변수 읽기가 지배적입니다), 학습은 연산에 의해 제한되며(compute-bound) 통신 부담이 큽니다. 가중치 동기화, 샘플 전송, 체크포인트 변환, 라우팅 일관성, 낮은 정밀도 동작 모두가 이 경계에서 신중하게 처리되어야 합니다.
> This composition is important. RL post-training requires generation and training to work together, but the two phases have very different performance profiles: rollout is memory-bandwidth-bound (KV-cache and parameter reads dominate during decoding), while training is compute-bound and communication-heavy. Weight synchronization, sample transfer, checkpoint conversion, routing consistency, and low-precision behavior all need to be handled carefully across the boundary.

이번 글의 나머지 부분에서는 Miles가 그 경계의 각 부분을 어떻게 다루는지 — Ray를 이용한 오케스트레이션, Megatron-LM을 이용한 확장, PyTorch를 이용한 확장성, 그리고 기본으로 제공되는 기능 — 을 살펴봅니다.
> The rest of this post walks through how Miles handles each piece of that boundary — orchestration with Ray, scaling with Megatron-LM, extensibility with PyTorch, and what comes out of the box.

## Ray: 장기 실행되는 RL 작업 오케스트레이션하기 / Ray: Orchestrating Long-Running RL Jobs

Miles는 Ray 분산 런타임 위에 직접 구축됩니다. Miles 실행에서는 장기 실행되는 모든 프로세스가 Ray 액터(actor)로 표현됩니다: 트레이너 랭크(rank), SGLang 롤아웃 서버, 라우팅 프록시, 비동기 롤아웃 워커가 모두 Ray의 액터 모델 안에서 동작합니다.
> Miles is built directly on the Ray distributed runtime. In a Miles run, every long-lived process is represented as a Ray actor: trainer ranks, SGLang rollout servers, routing proxies, and asynchronous rollout workers all live inside Ray's actor model.

이는 Miles에게 클러스터 규모 RL 워크로드를 위한 자연스러운 기반을 제공합니다.
> This gives Miles a natural foundation for cluster-scale RL workloads.

### GPU에 워커 배치하기 / Placing workers on GPUs

Miles는 액터 배치를 위해 Ray의 GPU 인식 스케줄러와 배치 그룹(placement group)을 사용하며, 실행 시점의 Ray 배치 스펙(placement spec)을 통해 분리형(disaggregated, 롤아웃과 학습을 별도 노드에서 수행) 및 코로케이션형(colocated, 롤아웃과 학습을 같은 노드에서 수행) 배치 방식을 모두 지원합니다. 프로세스 배치는 신중한 코로케이션(colocation)을 돕고 여분 노드를 예약하기 위해 랙(rack)을 인식해야 하며, 이는 오류 격리에도 핵심적입니다. 랙 내부에서 문제를 격리하는 일(예: 결함이 있는 GPU와 랙 전체 문제를 구분하는 일)이 항상 간단하지는 않기 때문입니다.
> Miles uses Ray's GPU-aware scheduler and placement groups for actor placement, supporting disaggregated (rollout and training on separate nodes) and colocated (rollout and training on the same nodes) layouts via launch-time Ray placement specs. Process placement must be rack-aware to facilitate careful colocation, reserving spare nodes, and key for error isolation, since isolating problems within a rack (e.g., distinguishing a bad GPU from a full rack issue) is not always straightforward.

### RL 파이프라인 전반에서 데이터 이동하기 / Moving data across the RL pipeline

프롬프트, 샘플, 갱신된 가중치는 롤아웃 액터와 트레이너 랭크 사이를 계속 순환하며, Miles는 이 흐름을 조율하기 위해 Ray 액터와 태스크를 사용합니다. 대량 가중치 전송의 경우, Ray는 제어 경로(control path)를 처리하고 텐서 바이트는 전용 NCCL/RDMA 채널을 통해 이동하여, Miles에게 Ray 수준의 프로그래밍 가능성과 대용량 데이터를 위한 빠른 경로(fast path)를 모두 제공합니다.
> Prompts, samples, and updated weights cycle continuously between rollout actors and trainer ranks, and Miles uses Ray actors and tasks to coordinate that flow. For bulk weight transfer, Ray handles the control path while the tensor bytes move over dedicated NCCL/RDMA channels, giving Miles both Ray-level programmability and a fast path for large data.

![Ray를 통한 가중치 동기화와 작업 감독 / Weight synchronization and job supervision via Ray](/assets/blog/2026-06-30-miles-a-pytorch-native-stack-for-large-scale-llm-rl-post-training/supervising.jpeg){:style="width:100%"}

### 장기 실행 작업 감독하기 / Supervising long-running jobs

Miles 실행은 처음부터 끝까지 하나의 Ray 작업이기 때문에, 별도로 덧붙인 인프라 없이도 Ray의 운영 표면(operator surface) — 작업 제출, 워커 감독, 로그 집계, 대시보드 가시성 — 을 그대로 물려받습니다. 장애 허용을 활성화하면, Miles는 실패한 랭크를 복구하고 동일한 Ray 기반 위에서 몇 주에 걸친 워크로드를 계속 진행시킬 수 있습니다.
> Because a Miles run is a Ray job end-to-end, it inherits Ray's operator surface — job submission, worker supervision, log aggregation, and dashboard visibility — without bolt-on infrastructure. With fault tolerance enabled, Miles can recover failed ranks and keep week-long workloads moving on top of the same Ray substrate.

### 완전 비동기 RL 지원하기 / Supporting fully asynchronous RL

Ray 액터는 지속적(persistent)이고 자신만의 상태를 가지며 독립적으로 스케줄링되기 때문에, Miles는 롤아웃과 학습이 더 이상 서로를 막지 않는 완전 비동기 모드로 실행될 수 있습니다 — 롤아웃 액터는 큐(queue)에 샘플을 계속 스트리밍하고, 트레이너는 자신의 속도에 맞춰 그 큐를 소비합니다.
> Because Ray actors are persistent, hold their own state, and are scheduled independently, Miles can run a fully asynchronous mode in which rollout and training no longer block on each other — rollout actors continuously stream samples into a queue that the trainer drains at its own pace.

## Megatron-LM: 학습 백엔드 확장하기 / Megatron-LM: Scaling the Training Backend

Miles는 Megatron-LM을 프로덕션 학습 백엔드로 사용하며, 이를 블랙박스 라이브러리로 감싸는 대신 Megatron의 인자 파서(argument parser), 모델 구성 파이프라인, 학습 루프, 병렬화 프리미티브, 분산 체크포인트 형식에 직접 연결됩니다. 이를 통해 Miles는 깔끔한 사용자 워크플로우를 유지하면서도 프런티어 규모의 밀집 및 MoE 학습에 필요한 인프라를 갖추게 됩니다.
> Miles uses Megatron-LM as its production training backend, plugging directly into Megatron's argument parser, model-construction pipeline, training loop, parallelism primitives, and distributed checkpoint format rather than wrapping it as a black-box library. That gives Miles the infrastructure needed for frontier-scale dense and MoE training while preserving a clean user-facing workflow.

### 하나의 인자 표면 / One argument surface

Megatron-LM은 이미 시퀀스 길이, 로터리 임베딩(rotary embedding), 그룹화된 GEMM(grouped GEMM), 모든 종류의 병렬화, 옵티마이저 설정, 활성화 체크포인팅 등 방대한 분산 학습 설정 표면(configuration surface)을 노출하고 있으며, Miles는 이를 감싸거나 다시 선언하는 대신 그대로 재사용합니다. 사용자는 Miles 전용 옵션과 표준 Megatron 옵션을 결합한 하나의 실행 스크립트를 통해 Miles 실행을 설정하므로, 설정 계층이 중복되지 않고 학습 설정이 업스트림 Megatron 동작에 가깝게 유지됩니다.
> Megatron-LM already exposes a large distributed-training configuration surface — sequence length, rotary embeddings, grouped GEMM, all flavors of parallelism, optimizer settings, activation checkpointing, and more — and Miles reuses it directly rather than wrapping or re-declaring it. Users configure a Miles run through one launch script that combines Miles-specific options with standard Megatron options, avoiding duplicated configuration layers and keeping the training setup close to upstream Megatron behavior.

### 장기 유지되는 포크 대신 모델 스펙 / Model specs instead of long-lived forks

프런티어 아키텍처는 빠르게 변화하며, 모델 계열 전반에 걸쳐 새로운 어텐션 블록, 라우팅 메커니즘, 전문가(expert) 레이아웃이 계속 등장합니다. 그래서 Miles는 이를 플러그인 형태의 모델 스펙(model spec)으로 다룹니다 — 커스텀 PyTorch 컴포넌트(예: 게이트형 어텐션 출력 모듈, Gated-Delta-Net 블록, 모델별 MoE 라우터)를 Megatron의 모델 파이프라인에 직접 삽입하는 작은 스펙 파일입니다. 이를 통해 Miles는 업스트림에서 계속 갈라져 나가는 장기 유지 Megatron 포크를 관리하지 않고도 DeepSeek-V3/V4, GLM-4.7, Qwen3 MoE 변형 등 새로운 아키텍처를 지원할 수 있습니다.
> Frontier architectures change quickly, with new attention blocks, routing mechanisms, and expert layouts arriving across model families, so Miles handles them through plug-in model specs — small spec files that insert custom PyTorch components (for example, a gated attention-output module, a Gated-Delta-Net block, or a model-specific MoE router) directly into Megatron's model pipeline. This lets Miles support new architectures — for example DeepSeek-V3/V4, GLM-4.7, and Qwen3 MoE variants — without maintaining a long-lived Megatron fork that constantly diverges from upstream.

### 병렬화를 인식하는 체크포인팅 / Parallelism-aware checkpointing

Miles는 Megatron의 병렬화를 인식하는 분산 체크포인트 형식을 사용하므로, 모델을 Hugging Face에서 한 번만 변환하면 가중치를 처음부터 다시 변환하지 않고도 서로 다른 텐서/파이프라인/컨텍스트/전문가 병렬 설정 전반에 걸쳐 불러올 수 있습니다. 대규모 학습 작업을 운영하는 팀에게 이는 모델이나 클러스터 형태가 바뀔 때마다 체크포인트 변환과 병렬화 변경이 별도의 엔지니어링 프로젝트가 되지 않는다는 것을 의미합니다.
> Miles uses Megatron's parallelism-aware distributed checkpoint format, so a model can be converted from Hugging Face once and then loaded across different tensor / pipeline / context / expert parallel configurations without re-converting weights from scratch. For teams operating large training jobs, this means checkpoint conversion and parallelism changes don't become a separate engineering project every time the model or cluster shape changes.

### 백엔드를 패치하지 않고 학습 확장하기 / Extending training without patching the backend

Miles는 학습 루프에서 잘 정의된 지점 — 모델 초기화 이후, 로그 확률(log-probability) 계산 이전, 각 학습 스텝 이전 — 에 훅(hook)을 노출하여, 사용자가 Megatron 내부를 수정하지 않고도 보조 손실, 커스텀 지표, 샘플 단위 진단, 클리핑(clipping) 규칙, 알고리즘별 동작을 추가할 수 있게 합니다. 설계 목표는 단순합니다: 백엔드는 강력하게 유지하되, 사용자 커스터마이징은 그 바깥에 두는 것입니다.
> Miles exposes hooks at well-defined points in the training loop — after model initialization, before log-probability computation, and before each training step — so users can add auxiliary losses, custom metrics, sample-level diagnostics, clipping rules, or algorithm-specific behavior without editing Megatron internals. The design goal is simple: keep the backend powerful, but keep user customization outside it.

## PyTorch: 모델, 수치 연산, 확장성을 위한 공통 계층 / PyTorch: The Common Layer for Models, Numerics, and Extensibility

PyTorch는 Miles 내부의 공통 프로그래밍 모델입니다: 모델 컴포넌트는 일반적인 `torch.nn.Module`이고, 손실은 표준 autograd 그래프이며, 혼합 정밀도(mixed precision), 그래디언트 체크포인팅, 분산 프리미티브, 프로파일링 모두 익숙한 PyTorch 워크플로우 안에 그대로 유지됩니다. 이는 LLM RL 사후 학습이 빠르게 변화하기 때문에 중요합니다 — 팀은 매번 새로운 추상화를 배우지 않고도 새로운 보상, 손실, 라우터, 모델 모듈, 디버깅 도구를 추가해야 합니다.
> PyTorch is the common programming model inside Miles: model components are regular torch.nn.Modules, losses are standard autograd graphs, and mixed precision, gradient checkpointing, distributed primitives, and profiling all stay inside familiar PyTorch workflows. This matters because LLM RL post-training changes fast — teams need to add new rewards, losses, routers, model modules, and debugging tools without learning a new abstraction each time.

### PyTorch 네이티브 모델 확장성 / PyTorch-native model extensibility

Miles의 플러그인 모델 스펙 메커니즘은 `torch.nn.Module`을 중심으로 구축되어 있어, 새로운 아키텍처를 지원한다는 것은 새 컴포넌트를 평범한 PyTorch 코드로 작성하고 이를 Megatron의 모델 파이프라인에 연결하는 것을 의미합니다 — autograd, 혼합 정밀도, 그래디언트 체크포인팅, 모듈 생명주기가 모두 PyTorch 사용자가 기대하는 방식대로 동작합니다. 팀은 Miles에서 모델을 실행하기 위해 별도의 중간 추상화로 모델을 변환할 필요가 없습니다.
> Miles' plug-in model-spec mechanism is built around torch.nn.Modules, so supporting a new architecture means writing the new component as ordinary PyTorch code and connecting it into Megatron's model pipeline — autograd, mixed precision, gradient checkpointing, and module lifecycle all keep working the way PyTorch users expect. Teams don't have to translate the model into a separate intermediate abstraction to get it running on Miles.

### PyTorch 네이티브 RL 커스터마이징 / PyTorch-native RL customization

같은 원칙이 RL 알고리즘에도 적용됩니다: 롤아웃 함수, 보상, 손실 함수, 샘플 필터, 지표, 학습 루프 훅은 모두 실행 시점에 제공되는 Python 모듈을 통해 커스터마이징되며, 학습 그래프의 나머지 부분과 조합되는 표준 PyTorch 연산을 사용합니다. 팀은 트레이너를 다시 작성하지 않고도 기존 레시피에서 시작해 보상을 교체하거나, 보조 손실을 추가하거나, 샘플 필터링을 바꾸거나, 새로운 진단 도구를 계측할 수 있습니다.
> The same principle applies to RL algorithms: rollout functions, rewards, loss functions, sample filters, metrics, and training-loop hooks are all customized through Python modules provided at launch time, using standard PyTorch operations that compose with the rest of the training graph. A team can start from an existing recipe and replace the reward, add an auxiliary loss, change sample filtering, or instrument new diagnostics without rewriting the trainer.

### 파이프라인 전반의 낮은 정밀도 레시피 / Low-precision recipes across the pipeline

Miles는 PyTorch의 dtype 시스템 위에 낮은 정밀도 파이프라인을 구축하며, BF16, FP8, MXFP8, INT4-QAT 레시피가 백엔드에만 국한된 고립된 기능으로 존재하는 대신 학습과 롤아웃 전반에 걸쳐 적용됩니다. 이런 일관성은 RL에서 특히 중요한데, 샘플을 생성하는 데 쓰이는 정책과 학습용 로그 확률을 계산하는 데 쓰이는 정책이 정렬된 상태를 유지해야 하기 때문입니다. Miles는 이런 수치 선택을 명시적이고 재현 가능하게 만들도록 설계되었습니다.
> Miles builds its low-precision pipeline on PyTorch's dtype system, with BF16, FP8, MXFP8, and INT4-QAT recipes that span training and rollout rather than living as isolated backend-only features. This consistency matters for RL because the policy used to generate samples and the policy used to compute training log probabilities must stay aligned, and Miles is designed to make those numerical choices explicit and reproducible.

### 익숙한 도구로 프로파일링하고 디버깅하기 / Profiling and debugging in familiar tools

대규모 RL 성능 문제는 롤아웃 지연 시간, 학습 연산, 집합 통신(collective communication), 데이터 이동, 가중치 동기화, 샘플 필터링, 스케줄링 등 어디에서든 나타날 수 있습니다. 그래서 Miles는 PyTorch 프로파일러를 연결하여 학습 단계의 Chrome 트레이스를 캡처하고, 이를 표준 도구로 살펴볼 수 있게 합니다. Megatron의 PyTorch 기반 백엔드, 그리고 지원되는 경우 사용할 수 있는 그래프 컴파일 경로와 결합되어, 이는 디버깅과 성능 작업을 익숙한 PyTorch 생태계 안에 그대로 유지시켜 줍니다.
> Large-scale RL performance issues can surface anywhere — rollout latency, training compute, collective communication, data movement, weight synchronization, sample filtering, or scheduling — so Miles wires in the PyTorch profiler to capture Chrome traces of training phases for inspection in standard tooling. Combined with Megatron's PyTorch-based backend and graph-compile paths where supported, this keeps debugging and performance work inside the familiar PyTorch ecosystem.

## Miles가 기본으로 제공하는 것 / What Miles Provides Out of the Box

Miles는 대규모 LLM RL 사후 학습에 필요한 핵심 시스템 기능을 제공하도록 설계되었습니다:
> Miles is designed to provide the core systems features needed for large-scale LLM RL post-training:

1. **롤아웃과 학습 통합(Rollout and training integration)** — SGLang 롤아웃과 Megatron-LM 학습을 연결하며, 서로 다른 GPU 예산과 활용률 목표에 맞춰 분리형과 코로케이션형 실행을 모두 지원합니다.
2. **비동기 실행(Asynchronous execution)** — 완전 비동기 모드는 롤아웃을 학습에서 분리합니다: 롤아웃 액터가 큐에 샘플을 계속 스트리밍하면 트레이너가 자신의 속도로 소비하여, 두 단계 사이의 반복(iteration)별 블로킹을 없앱니다.
3. **빠른 가중치 동기화(Fast weight synchronization)** — 매 학습 업데이트 이후, 새로운 가중치는 전용 NCCL/RDMA 채널을 통해 롤아웃 워커로 전달되며, Ray는 제어 경로만 처리하여 대량의 텐서 바이트가 Python 데이터 경로를 거치지 않도록 합니다.
4. **MoE를 인식하는 롤아웃/학습 정렬(MoE-aware rollout/training alignment)** — 롤아웃 라우팅 리플레이(Rollout Routing Replay)가 롤아웃/학습 경계 전반에서 라우팅 결정을 보존하여, 그렇지 않으면 MoE RL을 불안정하게 만들 수 있는 트레이너 대 롤아웃 라우팅 불일치를 줄입니다.
5. **낮은 정밀도 지원(Low-precision support)** — 고립된 학습 전용 레시피가 아니라 엔드투엔드 RL 스택의 일부로 설계된 통합 BF16 / FP8 / MXFP8 / INT4-QAT 파이프라인입니다.
6. **롤아웃과 학습 전반의 LoRA(LoRA across rollout and training)** — LoRA는 롤아웃과 학습 경로 모두에서 지원되어, 대형 베이스 모델에서 비용을 줄이고 반복 속도를 높이는 매개변수 효율적인(parameter-efficient) 사후 학습을 가능하게 합니다.
7. **장애 허용과 관측 가능성(Fault tolerance and observability)** — Ray의 작업 및 액터 모델이 감독, 로그 집계, 대시보드 가시성을 제공하며, 랭크 수준의 장애 허용이 몇 주에 걸친 학습 실행을 계속 진행시킵니다. PyTorch 프로파일러 통합은 학습 수준의 시야를 담당합니다.
8. **폭넓은 모델 및 하드웨어 지원(Broad model and hardware support)** — Miles는 DeepSeek-V4, Kimi K2.5 / K2.6, GLM-5 / 5.1, Qwen3.5 / 3.6를 비롯한 프런티어 및 오픈 소스 모델을 위한 바로 실행 가능한 레시피를 제공하며, NVIDIA의 플래그십 Hopper / Blackwell GPU를 지원합니다.

> 1. **Rollout and training integration** — Connects SGLang rollout with Megatron-LM training, with both disaggregated and colocated execution to fit different GPU budgets and utilization targets.
> 2. **Asynchronous execution** — Fully async mode decouples rollout from training: rollout actors stream samples continuously into a queue that the trainer drains at its own pace, eliminating the per-iteration blocking between the two phases.
> 3. **Fast weight synchronization** — After each training update, fresh weights flow to rollout workers over dedicated NCCL/RDMA channels, with Ray handling only the control path so bulk tensor bytes stay off the Python data path.
> 4. **MoE-aware rollout/training alignment** — Rollout Routing Replay preserves routing decisions across the rollout/training boundary, reducing the trainer-vs-rollout routing mismatch that would otherwise destabilize MoE RL.
> 5. **Low-precision support** — A unified BF16 / FP8 / MXFP8 / INT4-QAT pipeline designed as part of the end-to-end RL stack rather than as isolated training-only recipes.
> 6. **LoRA across rollout and training** — LoRA is supported in both rollout and training paths, enabling parameter-efficient post-training that reduces cost and speeds up iteration on large base models.
> 7. **Fault tolerance and observability** — Ray's job and actor model provide supervision, log aggregation, and dashboard visibility, while rank-level fault tolerance keeps week-long training runs moving; PyTorch profiler integration covers the training-level view.
> 8. **Broad model and hardware support** — Miles ships ready-to-run recipes for frontier and open-source models including DeepSeek-V4, Kimi K2.5 / K2.6, GLM-5 / 5.1, and Qwen3.5 / 3.6, with support for NVIDIA flagship Hopper / Blackwell GPUs.

## 작은 코어와 많은 확장 지점 / A Small Core with Many Extension Points

Miles의 가장 중요한 설계 선택 중 하나는 핵심 트레이너를 작게 유지한다는 것입니다.
> One of Miles' most important design choices is that the core trainer stays small.

새로운 알고리즘이나 모델 계열마다 사용자에게 프레임워크를 포크하도록 강요하는 대신, Miles는 명시적인 확장 지점(extension point)을 노출합니다:
> Instead of forcing users to fork the framework for every new algorithm or model family, Miles exposes explicit extension points:

1. 커스텀 생성 동작을 위한 **롤아웃 함수**.
2. 작업별 지도(supervision)를 위한 **보상 함수**.
3. 새로운 RL 목적함수를 위한 **손실 함수**.
4. 데이터 선택과 거부를 위한 **샘플 필터**.
5. 지표, 진단, 보조 손실, 커스텀 업데이트 로직을 위한 **학습 훅**.
6. 아키텍처별 모듈을 위한 **모델 스펙**.

> 1. **Rollout functions** for custom generation behavior.
> 2. **Reward functions** for task-specific supervision.
> 3. **Loss functions** for new RL objectives.
> 4. **Sample filters** for data selection and rejection.
> 5. **Training hooks** for metrics, diagnostics, auxiliary losses, and custom update logic.
> 6. **Model specs** for architecture-specific modules.

이러한 확장 지점 덕분에 Miles는 고전적인 RLHF 방식 학습, 규칙 기반 보상 학습, 코드 및 에이전틱(agentic) 작업, MoE 사후 학습, 낮은 정밀도 실험, 커스텀 관측 가능성이나 안전성 검사가 필요한 프로덕션 파이프라인 등 다양한 사후 학습 워크플로우 전반에서 유용하게 쓰입니다.
> These extension points make Miles useful across a range of post-training workflows: classic RLHF-style training, rule-based reward training, code and agentic tasks, MoE post-training, low-precision experiments, and production pipelines that need custom observability or safety checks.

요약하면, Miles는 배치, 가중치 동기화, 장애 허용, 낮은 정밀도 레시피와 같은 시스템 수준의 결정을 대신 내려주어, 사용자 코드는 알고리즘과 제품 로직에 집중할 수 있게 합니다.
> In short, Miles makes the systems-level decisions — placement, weight sync, fault tolerance, low-precision recipes — so that user code can focus on algorithm and product logic.

![작은 코어와 많은 확장 지점 / A small core with many extension points](/assets/blog/2026-06-30-miles-a-pytorch-native-stack-for-large-scale-llm-rl-post-training/small-core-many-extension-points.png){:style="width:100%"}

## 앞으로의 전망 / Looking Ahead

LLM 사후 학습은 더 큰 모델, 더 긴 컨텍스트, 더 많은 MoE, 그리고 더 비동기적이고 에이전틱하며 시스템 집약적인 RL 파이프라인을 향해 빠르게 나아가고 있으며, Miles는 바로 이 흐름을 위해 만들어졌습니다: SGLang, Ray, Megatron-LM, PyTorch를 작고 플러그 가능한 트레이너 뒤에 조합함으로써, 연구자와 인프라 팀에게 알고리즘 실험부터 대규모 RL 실행까지 이어지는 PyTorch 네이티브 경로를 제공합니다. 이것이 바로 프런티어 규모의 LLM RL 사후 학습을 더 쉽게 재현하고, 확장하고, 운영할 수 있도록 Miles를 오픈 소스로 공개하는 이유입니다.
> LLM post-training is moving quickly — larger models, longer contexts, more MoE, and more asynchronous, agentic, system-intensive RL pipelines — and Miles is built for that trajectory: by composing SGLang, Ray, Megatron-LM, and PyTorch behind a small pluggable trainer, it gives researchers and infrastructure teams a PyTorch-native path from algorithm experimentation to large-scale RL runs, which is why we are open-sourcing Miles to make frontier-scale LLM RL post-training easier to reproduce, extend, and operate.
