---
layout: blog_detail
title: "torchcomms: 현대적인 PyTorch 통신 API"
author: Team torchcomms at Meta
category: ["pytorch.org", "translation"]
org_title: "torchcomms: a modern PyTorch communications API"
org_link: https://pytorch.org/blog/torchcomms/
---

## 소개 / Introduction

torchcomms는 PyTorch Distributed(PTD)와 함께 사용하기 위한 새로운 실험적 경량 통신 API입니다. 핵심 API와 함께, 100,000개 이상의 GPU로 확장할 수 있도록 개발한 새로운 백엔드인 NCCLX도 오픈소스로 공개합니다.
> Torchcomms is a new experimental, lightweight communication API intended for use with PyTorch Distributed (PTD). In addition to the core API, we are open-sourcing NCCLX, a new backend we developed to scale to over 100,000 GPUs.

torchcomms의 첫 번째 릴리즈에서는 PyTorch에서 대규모 모델 학습에 필요한 기반 API와 백엔드를 제공합니다. 이번 초기 릴리즈는 대규모 분산 학습을 안정적이고 고성능으로 수행할 수 있게 하는 핵심 통신 프리미티브(primitive)에 중점을 두고 있습니다. 향후 1년간 새로운 집합 통신(collective)의 프로토타이핑을 쉽게 하고, 내장된 내결함성(fault tolerance)으로 원활하게 확장하며, 디바이스 중심 통신 패턴을 최적화하는 기능을 도입하여 계속 발전시켜 나갈 것입니다. 로드맵은 연구자와 개발자가 더 빠르게 움직이고, 대규모로 새로운 아이디어를 테스트하며, 차세대 대규모 AI 시스템을 구축할 수 있도록 하는 데 초점을 맞추고 있습니다.
> With our first release of torchcomms, we're delivering the foundational APIs and backends required for large-scale model training in PyTorch. This initial release focuses on core communication primitives that enable reliable and performant distributed training at scale. Over the next year, we'll continue to mature the offering—introducing features that make it easier to prototype new collectives, scale seamlessly with built-in fault tolerance, and optimize device-centric communication patterns. Our roadmap is focused on empowering researchers and developers to move faster, test new ideas at scale, and build the next generation of large-scale AI systems.

torchcomms는 대규모 환경에서 새로운 통신 패러다임을 검증하기 위한 첫 걸음입니다. 혁신을 가속화하기 위해, API를 완전히 공개적으로 개발하며, 진화하는 과정에서 커뮤니티 피드백을 적극적으로 수렴하고 있습니다. 이러한 공개 개발 방식 때문에, API는 아직 초기 단계이며 성숙해지는 과정에서 호환성이 깨지는 변경(breaking change)이 있을 수 있습니다. 시간이 지남에 따라, torchcomms는 차세대 분산 기술의 시험장 역할을 하게 되며, 장기적으로는 모든 PyTorch Distributed 기능을 이 새로운 기반 위로 마이그레이션하는 것이 목표입니다. torchcomms가 안정화되면, PyTorch에서 확장 가능하고 내결함성이 있으며 디바이스 중심적인 분산 학습의 핵심 기반이 될 것입니다.
> Torchcomms is our first step toward proving out new communication paradigms at scale. To accelerate innovation, we're developing the API fully in the open, inviting community feedback as it evolves. Because of this open development process, the API is still early and may undergo breaking changes as it matures. Over time, torchcomms will serve as a proving ground for next-generation distributed technologies, with the long-term goal of migrating all PyTorch Distributed functionality onto this new foundation. As torchcomms stabilizes, it will become the backbone of scalable, fault-tolerant, and device-centric distributed training in PyTorch.

### 프로젝트 목표 / Project Goals

torchcomms를 통해, PyTorch 분산 통신의 차세대 기반을 마련하고 있습니다. 목표는 개발자와 연구자가 더 빠르게 움직이고, 더 넓은 범위로 확장하며, 더 다양한 하드웨어를 대상으로 할 수 있는 유연하고 확장 가능한 기반을 구축하는 것입니다. 구체적으로 다음과 같은 목표를 향해 나아가고 있습니다:
> With torchcomms, we're laying the groundwork for the next generation of distributed communication in PyTorch. Our goal is to build a flexible, extensible foundation that enables developers and researchers to move faster, scale further, and target a wider variety of hardware. Specifically, we're working toward the following objectives:

1. **통신 프리미티브의 빠른 프로토타이핑 / Fast Prototyping of Communication Primitives** – 머신러닝 연구자들은 새로운 통신 패러다임을 빠르게 실험할 수 있어야 합니다. torchcomms는 통신을 PyTorch의 핵심 수치 연산 프리미티브와 분리함으로써, 기존 기능을 깨뜨리지 않고 통신 계층을 독립적으로 반복 개선할 수 있게 합니다—새로운 집합 통신, API, 백엔드를 자유롭게 추가할 수 있습니다. 이 설계는 트리 외부(out-of-tree) 백엔드도 지원하여, 연구자와 하드웨어 벤더가 자신의 디바이스와 기능에 맞춘 특화된 통신 스택을 쉽게 통합할 수 있습니다.
2. **100K+ GPU로의 확장 / Scaling to 100K+ GPUs** – 최신 학습 워크로드를 수십만 개의 GPU로 확장하려면 통신 리소스 관리 방식을 재고해야 합니다. 지연 초기화(lazy initialization)나 점대점(point-to-point) 연산의 제한된 동시성 의미론 같은 현재 접근 방식은 NCCL과 같은 라이브러리 내에서 확장성을 제약합니다. torchcomms는 즉시 초기화(eager initialization, 백엔드 리소스를 사용자가 명시적으로 관리)와 모델별 힌트를 도입하여 커뮤니케이터, NVLink 버퍼, RoCE 리소스의 할당과 공유를 최적화합니다—진정한 대규모 분산 작업을 위한 길을 열어줍니다.
3. **이기종 하드웨어 지원 / Heterogeneous Hardware Support** – 기존 집합 통신 백엔드는 일반적으로 단일 벤더나 하드웨어 제품군에 최적화되어 있습니다. torchcomms는 처음부터 이기종 시스템을 고려하여 설계하고 있으며, 단일 학습 작업 내에서 여러 하드웨어 세대와 벤더에 걸친 혼합 배포를 가능하게 합니다. 이러한 유연성은 생태계가 동종(homogeneous) GPU 클러스터를 넘어 진화함에 따라 매우 중요합니다.
4. **대규모 내결함성 / Fault Tolerance at Scale** – 현재 오픈소스 PyTorch Distributed에는 견고한 내결함성 프로세스 그룹이 부족하여, torchft와 같은 상위 수준 라이브러리의 안정성이 제한됩니다. torchcomms는 내결함성 HSDP 및 내결함성 Streaming DiLoCo와 같은 알고리즘을 대규모로 지원할 수 있는 내결함성 백엔드를 오픈소스로 공개하여 이 격차를 해소하는 것을 목표로 합니다—성능 저하 없이 복원력을 제공합니다.
5. **단방향 통신 / One-Sided Communication** – 단방향 통신(예: RDMA 스타일 시맨틱)은 강화 학습, 체크포인팅, 대규모 언어 모델에서의 비동기 워크플로우에 점점 더 필수적이 되고 있습니다. torchcomms는 단방향 통신에 대한 일급(first-class) 지원을 제공하여, 분산 프로세스 간의 효율적이고 오버헤드가 낮은 메시지 전달 및 데이터 교환을 가능하게 합니다.
6. **디바이스 중심 집합 통신 / Device-Centric Collectives** – 추론과 학습에서 초저지연을 달성하려면, 통신과 연산이 긴밀하게 결합되어야 합니다. torchcomms는 디바이스(예: GPU)에 통신 메타데이터와 로직이 직접 상주할 수 있는 디바이스 중심 집합 통신 API를 개발하고 있습니다. 여기에는 GPU에서의 직접 RDMA 연산(예: IBGDA)과 CPU 프록시 기반 설계가 모두 포함됩니다. 이러한 기능을 통해 개발자는 연산과 통신 작업을 매끄럽게 융합하여, 새로운 수준의 성능을 달성할 수 있습니다.

> 1. **Fast Prototyping of Communication Primitives** – Machine learning researchers need to experiment rapidly with new communication paradigms. By decoupling communications from PyTorch's core numeric primitives, torchcomms makes it possible to iterate on communication layers independently—adding new collectives, APIs, or backends without breaking existing functionality. This design also enables out-of-tree backends, allowing researchers and hardware vendors to easily integrate specialized communication stacks tailored to their devices and features.
> 2. **Scaling to 100K+ GPUs** – Scaling modern training workloads to hundreds of thousands of GPUs requires rethinking how communication resources are managed. Current approaches, such as lazy initialization and limited concurrency semantics for point-to-point operations, constrain scalability within libraries like NCCL. Torchcomms introduces eager initialization (where backend resources are explicitly managed by the user) and model-specific hints to optimize how communicators, NVLink buffers, and RoCE resources are allocated and shared—paving the way for truly massive distributed jobs.
> 3. **Heterogeneous Hardware Support** – Existing collective backends are typically optimized for a single vendor or hardware family. With torchcomms, we're designing for heterogeneous systems from the ground up—enabling mixed deployments that span multiple hardware generations and vendors within a single training job. This flexibility is critical as the ecosystem evolves beyond homogeneous GPU clusters.
> 4. **Fault Tolerance at Scale** – Today's open-source PyTorch Distributed lacks robust fault-tolerant process groups, which limits the reliability of higher-level libraries like torchft. Torchcomms aims to close that gap by open-sourcing a fault-tolerant backend capable of supporting algorithms such as fault-tolerant HSDP and fault-tolerant Streaming DiLoCo at scale—delivering resilience without compromising performance.
> 5. **One-Sided Communication** – One-sided communication (e.g., RDMA-style semantics) is increasingly essential for asynchronous workflows in reinforcement learning, checkpointing, and large language models. Torchcomms will provide first-class support for one-sided communication, enabling efficient, low-overhead message passing and data exchange between distributed processes.
> 6. **Device-Centric Collectives** – To achieve ultra-low latency for inference and training, communication and computation must be tightly coupled. Torchcomms is developing device-centric collective APIs, which enable communication metadata and logic to live directly on the device (e.g. the GPU). This includes both direct RDMA operations from the GPU (e.g., IBGDA) and CPU proxy-based designs. These capabilities allow developers to fuse compute and communication operations seamlessly, unlocking new levels of performance.

### 왜 새로운 API인가? / Why a new API?

자주 듣는 질문이 있습니다: "왜 새로운 API인가요?"
> A common question we hear is: "Why a new API?"

torchcomms를 통해, 현재 다른 어떤 통신 라이브러리에도 존재하지 않는 기능을 도입하는 야심찬 목표를 추구하고 있습니다. 빠르게 나아가기 위해서는, 기존 인터페이스에 제약받지 않고 공개적으로 반복하며 설계를 발전시킬 수 있는 자유가 필요합니다. 이는 초기 단계에서 커뮤니티와 함께 실험하고 개선하는 과정에서 API에 호환성이 깨지는 변경이 있을 수 있음을 의미합니다.
> With torchcomms, we're pursuing a set of ambitious goals—introducing capabilities that don't yet exist in any other communication library today. To move quickly, we need the freedom to iterate in the open and evolve the design without being constrained by existing interfaces. This means that, during its early stages, the API may experience breaking changes as we experiment and refine it in collaboration with the community.

PyTorch Distributed의 기존 c10d API는 상당한 기술 부채(technical debt)를 안고 있어 확장하거나 현대화하기 어렵습니다. torchcomms API가 안정화되면, 기존 c10d::Backend 인터페이스를 지원 중단(deprecate)하고 torchcomms를 PyTorch Distributed의 내부 구현으로 채택할 계획입니다. 이 전환은 점진적으로 최소한의 영향으로 이루어질 것입니다—대부분의 사용자와 모델은 기존과 동일하게 작동하면서, 새로운 백엔드의 성능, 확장성, 유연성의 이점을 자동으로 누리게 됩니다.
> The existing c10d APIs in PyTorch Distributed carry significant technical debt, making them difficult to extend or modernize. As the torchcomms API stabilizes, we plan to deprecate the old c10d::Backend interface and adopt torchcomms as the underlying implementation for PyTorch Distributed. This transition will be done gradually and with minimal disruption—most users and models will continue to work as they do today, while automatically benefiting from the performance, scalability, and flexibility of the new backends.

## 빠른 시작 / Quickstart

먼저, torchcomms 설치 방법은 [설치 안내](https://github.com/meta-pytorch/torchcomms?tab=readme-ov-file#installation)를 참고하세요.
> First, see the [Installation instructions](https://github.com/meta-pytorch/torchcomms?tab=readme-ov-file#installation) for how to install torchcomms.

더 자세한 문서는 [https://meta-pytorch.org/torchcomms/](https://meta-pytorch.org/torchcomms/)에서 확인할 수 있습니다.
> For more documentation, check out: [https://meta-pytorch.org/torchcomms/](https://meta-pytorch.org/torchcomms/)

### 기본 사용법 / Basic Usage

torchcomms는 기본 백엔드와 커뮤니케이터를 감싸는 경량 래퍼입니다. 핵심 API는 백엔드 메서드에 직접 매핑되며, 완전한 객체 지향 API로 설계되어 있습니다.
> Torchcomms is a lightweight wrapper around the underlying backends and communicators. The core APIs map directly to the backend methods and are designed as a fully object-oriented API.

```python
import torchcomms

# torchrun이 제공하는 MASTER_PORT/MASTER_ADDR/RANK/WORLD_SIZE 환경 변수를 사용하여
# 커뮤니케이터를 즉시 초기화합니다.
# 이 커뮤니케이터는 단일 디바이스에 바인딩됩니다.
comm = torchcomms.new_comm("ncclx", torch.device("cuda"), name="my_comm")
print(f"I am rank {comm.get_rank()} of {comm.get_size()}!")

t = torch.full((10, 20), value=comm.rank, dtype=torch.float)

# 현재 스트림에서 all_reduce 실행
comm.allreduce(t, torchcomms.ReduceOp.SUM, async_op=False)

# 백그라운드 스트림에서 all_reduce 실행
work = comm.allreduce(t, torchcomms.ReduceOp.SUM, async_op=True)
work.wait()

# 커뮤니케이터를 8개씩 그룹으로 분할
split_groups = torch.arange(comm.get_size()).view(-1, 8).tolist()
tp_comm = comm.split(split_groups)
```

## DeviceMesh

torchcomms는 FSDP2와 텐서 병렬화(tensor parallelism) 등 PyTorch 병렬화 라이브러리와의 호환성을 위해 DeviceMesh도 지원합니다.
> Torchcomms also supports compatibility with DeviceMesh for compatibility with PyTorch parallelism libraries such as FSDP2.

```python
import torchcomms
from torchcomms.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

comm = torchcomms.new_comm("ncclx", torch.device("cuda:0"), name="global")

mesh = init_device_mesh(
    mesh_dim_comms=(comm,),
    mesh_dim_names=("global",),
)
fully_shard(model, device_mesh=mesh)
```

## 초기 백엔드 / Initial Backends

새로운 torchcomms API와 함께, 다양한 하드웨어 플랫폼을 위한 여러 백엔드를 릴리즈했습니다.
> Along with the new torchcomms APIs, we have released several backends for a variety of hardware platforms.

![torchcomms 백엔드 아키텍처 / torchcomms backend architecture](/assets/blog/2025-10-22-torchcomms/1-4.png){:style="width:100%"}

### NCCLX

NCCLX는 널리 사용되는 [NCCL](https://github.com/NVIDIA/nccl) 라이브러리의 Meta 확장을 포함합니다. NCCLX는 프로덕션에서 검증되었으며, Llama3과 Llama4와 같은 대규모 언어 모델(LLM)의 대규모 학습과 추론에 사용되고 있습니다. 현재 Meta의 모든 생성형 AI 서비스는 NCCLX로 구동됩니다. NCCLX의 주요 기능은 다음과 같습니다:
> NCCLX contains the Meta extension to the popular [NCCL](https://github.com/NVIDIA/nccl) library. NCCLX is production-tested – it is used for large scale training and inference for large language models (LLMs) such as Llama3 and Llama4. Today, all of Meta's generative AI services are backed by NCCLX. Some key features of NCCLX include:

* 확장 가능한 초기화(scalable initialization)
* 제로 카피(zero-copy) 및 SM 비사용 통신
* 커스텀 집합 통신 알고리즘
* 네트워크 트래픽 부하 분산
* 단방향 통신(one-sided communication)
* GPU 상주(GPU-resident) 및 저지연 집합 통신
* 장애 분석기 및 장애 위치 파악

> * Scalable initialization
> * Zero-copy and SM-free communication
> * Custom collective algorithms
> * Network traffic load balancing
> * One-sided communication
> * GPU-resident and low latency collectives
> * Fault analyzer and localization

업스트림 NCCL과 병행하여, 이러한 Meta 자체 최적화와 커스텀 기능을 호스팅하기 위한 별도의 Custom Transport(CTran) 스택을 개발했습니다. CTran은 NVLink, IB/RoCE, TCP 전송을 포함하여, 다양한 하드웨어 루틴을 통해 하위 수준 통신 프리미티브를 지원하고, 다양한 통신 시맨틱(예: 집합 통신, 점대점, RMA)을 위한 통신 알고리즘을 구축합니다.
> In parallel with the upstream NCCL, we have developed a separate Custom Transport (CTran) stack to host these Meta in-house optimizations and custom features. CTran contains NVLink, IB/RoCE and TCP transports to support lower-level communication primitives via different hardware routines and build communication algorithms for various communication semantics (e.g., collectives, point-to-point, RMA) over the transports.

NCCLX와 CTran 모두 torchcomms와 함께 오늘 오픈소스로 공개됩니다. NCCLX/CTran에 대한 자세한 내용은 이번 주 후반에 백서(white paper)에서 다룰 예정입니다.
> Both NCCLX and CTran are open sourced today, along with torchcomms. We will discuss more details of NCCLX/CTran in a white paper later this week.

### NCCL과 RCCL / NCCL and RCCL

NCCLX 외에도, torchcomms는 업스트림 NCCL도 지원합니다. 현재 PyTorch Distributed NCCL 사용자는 기존 통신 라이브러리 설정을 변경하지 않고도 torchcomms를 쉽게 사용해볼 수 있습니다.
> In addition to NCCLX, torchcomms also supports upstream NCCL. Current PyTorch Distributed NCCL users can try out torchcomms easily without changing the underlying communication library setup.

현재 PyTorch Distributed에서 AMD RCCL 지원은 NCCL 프로세스 그룹을 통해 이루어지고 있습니다. torchcomms 릴리즈의 일환으로, 네이티브 RCCL 백엔드도 포함했습니다. 이를 통해 torchcomms는 첫날부터 네이티브 멀티 벤더 GPU를 지원합니다. 또한 서로 다른 라이브러리가 더 독립적으로 발전할 수 있게 합니다.
> The AMD RCCL support in the current PyTorch Distributed is through the NCCL process group. As part of torchcomms release, we have also included a native RCCL backend. This allows torchcomms to provide native multi-vendor GPU support from Day 1. It allows different libraries to evolve more independently.

### Gloo

Gloo는 머신 간에 CPU 메타데이터를 전송하거나 테스트에 사용하는 백엔드로 알려져 있을 것입니다. 이것이 주된 용도이지만, InfiniBand와 단방향 연산 같은 새로운 고급 기능도 갖추고 있습니다. 최근에는 Gloo가 100,000개 이상의 워커로 확장할 수 있는 새로운 "lazy init" 모드도 추가했습니다.
> You may know of Gloo as the backend you use when you need to transfer CPU metadata between machines or for tests. That is the main use case but it also has some new advanced features such as infiniBand and one sided operations. We recently also added a new "lazy init" mode that allows Gloo to scale to 100k or more workers.

## 구성 가능성: torchtitan / Composability: torchtitan

torchtitan에 통합하여 새로운 torchcomms API의 호환성과 정확성을 검증했습니다. 이 통합은 DeviceMesh 연동을 사용하여 FSDP2 및 텐서 병렬화와 같은 기존 PyTorch 기술과의 호환성을 제공합니다.
> We've demonstrated compatibility and correctness of the new torchcomms API by integrating it in with torchtitan. This uses the device mesh integration to provide compatibility with the existing PyTorch technologies, such as FSDP2 and tensor parallelism.

torchtitan 통합 [링크](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/torchcomms), 통합 코드는 `torchtitan/experiments/torchcomms/` 경로에 있습니다.
> [Link](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/torchcomms) to torchtitan integration, the integration code will be under path: torchtitan/experiments/torchcomms/.

torchtitan 손실/성능 곡선 [링크](https://www.internalfb.com/mlhub/pipelines/runs/mast/llama3_8b_comms_dp_memory_test_full_print-64-yifanmao-vmnmdg2?job_attempt=1&version=0&tab=tb_legacy&env=PRODUCTION) (FSDP2 사용):
> [Link](https://www.internalfb.com/mlhub/pipelines/runs/mast/llama3_8b_comms_dp_memory_test_full_print-64-yifanmao-vmnmdg2?job_attempt=1&version=0&tab=tb_legacy&env=PRODUCTION) to torchtitan loss/performance curves: (with FSDP2)

![torchtitan 손실 곡선 / torchtitan loss curves](/assets/blog/2025-10-22-torchcomms/2-6.png){:style="width:100%"}

![torchtitan 성능 곡선 / torchtitan performance curves](/assets/blog/2025-10-22-torchcomms/3-4.png){:style="width:100%"}

## 새로운 API / New APIs

### 집합 통신 시맨틱 변경 사항 / Collective Semantic Changes

기존 PyTorch Distributed API에서 상속받은 집합 통신에 여러 변경 사항을 적용했습니다. 이러한 변경은 고수준 시맨틱을 기저 디바이스 시맨틱에 더 잘 맞추고, 유연성을 향상시키기 위한 것입니다.
> We've made a number of changes to the existing collectives that were inherited from the existing PyTorch Distributed APIs. These are intended to make the high level semantics better match the underlying device semantics and to improve flexibility.

1. 모든 연산은 전역 `dist.*` API 대신 객체 지향 API를 통해 수행됩니다.
2. 각 `torchcomm.TorchComm` 객체는 단일 디바이스와 커뮤니케이터에 매핑됩니다.
3. 백엔드는 즉시 초기화되며, 생성 시 디바이스를 전달해야 합니다.
4. 모든 연산은 "전역(global)" 랭크가 아닌 커뮤니케이터 랭크를 사용합니다.
5. 모든 연산은 발행된 순서대로 실행되며, 동시 연산은 배치(batch) API를 사용해야 합니다.
6. send/recv는 배치 API를 통해 발행하지 않는 한 동시에 실행되지 않습니다.

> 1. All operations are done through object oriented APIs rather than using the global dist.* APIs.
> 2. Each torchcomm.TorchComm object maps to a single device and communicator.
> 3. Backends are eagerly initiated and require a device to be passed in at creation time.
> 4. All operations use the communicator ranks rather than "global" ranks.
> 5. All operations execute in order they were issued and concurrent operations must be run using the batch API.
> 6. send/recvs do not execute concurrently unless issued via the batch API.

### 윈도우 API / Window APIs

원격 메모리에 대한 동적 put/get 연산을 허용하는 윈도우 API 지원을 추가하고 있습니다. 체크포인팅이나 비동기 연산을 포함한 특정 사용 사례에서 이 방식은 전통적인 집합 통신과 달리 한쪽만 관여하면 되므로 훨씬 더 높은 성능을 보이며 표현하기도 쉽습니다.
> We're adding support for window APIs to allow for dynamic put/get operations on remote memory. For certain use cases including checkpointing, async operations this can be significantly more performant and easier to express since only one side needs to be involved unlike traditional collectives.

윈도우 API를 사용하면 다른 랭크에 걸쳐 GPU 또는 CPU 메모리에 메모리 버퍼를 생성할 수 있습니다. 생성된 버퍼는 자동으로 등록되며, 기저의 RDMA 또는 NVL 전송을 활용하는 Put 및 Get API를 통해 제로 카피 단방향 통신으로 접근할 수 있습니다. 또한, 윈도우 API는 원자적(atomic) 시그널링 메커니즘을 제공하여 비동기 통신 기능을 더욱 강화합니다.
> The window APIs enable users to create a memory buffer—either in GPU or CPU memory—across different ranks. Once created, the buffer is automatically registered and can be accessed via the provided Put and Get APIs, leveraging the underlying RDMA or NVL transport for zero-copy, one-sided communication. Additionally, the window APIs offer an atomic signaling mechanism, further enhancing asynchronous communication capabilities.

윈도우 API는 현재 활발히 개발 중이며 아직 실험적입니다.
> The window APIs are under active development and still experimental.

### 전송 API / Transport APIs

기저 전송을 직접 사용하여 점대점 연산을 수행할 수 있는 전송 API를 추가하고 있습니다. 이 API는 윈도우 API와 유사하지만 집합 통신 라이브러리에 종속되지 않습니다. 초기에는 RPC와 같은 연산을 위한 전용 네트워크를 통한 RDMA만 지원합니다. 이는 내부적으로 CTran의 IB 백엔드에 의해 지원됩니다.
> We're adding transport APIs that allow for doing point to point operations using the underlying transport directly. This provides a similar API to window APIs but not tied to a collective library. Initially we're providing support just for RDMA over a dedicated Network which is intended for use in RPC like operations. This is internally supported by the IB backend in CTran.

RdmaTransport는 원격 메모리에 직접 쓸 수 있는 write API를 제공합니다. 사용자는 메모리를 등록하고 프로세스 간에 핸들을 교환하여 쓰기를 수행해야 합니다. 이는 사실상 제로 카피 데이터 전송이며, CPU 또는 GPU 메모리 모두에서 제로 카피 방식으로 수행할 수 있습니다. 이 API는 순수 전송 API이므로 연산(reduce 등)은 수행하지 않습니다.
> The RdmaTransport provides a write API that allows users to directly write into the remote memory. Users would need to register the memory and exchange its handle between processes to facilitate the write. This is effectively a zero copy data transfer, and can be done for a CPU or GPU memory in a zero copy fashion. These APIs are only transport APIs and do no compute (no reduce, etc).

전송 API는 현재 활발히 개발 중이며 아직 실험적입니다.
> The transport APIs are under active development and are still experimental.

### 내결함성 API / Fault Tolerance APIs

내결함성 집합 통신을 제공하는 새로운 백엔드를 만들고 있습니다. 이 새로운 백엔드는 전적으로 CTran 전송 위에 구축되었으며, 장애 감지, 타임아웃, 오류 복구, 오류 후 안전한 재구성을 제공합니다.
> We're working on creating a new backend that provides fault tolerant collectives. This new backend is built entirely on the CTran transport and provides failure detection, timeouts, error recovery and safe reconfiguration after errors.

## 확장성 / Extensibility

### 새로운 집합 통신으로 백엔드 확장하기 / Extending Backends with New Collectives

torchcomms는 기저 백엔드에 대한 직접 접근을 지원하도록 설계되어 있습니다. 이를 통해 공유 백엔드 인터페이스에 추가하기 전에 새로운 API를 빠르게 프로토타이핑할 수 있습니다.
> Torchcomms is designed to support direct access to the underlying backends. This allows for fast prototyping of new APIs before we standardize them and add them to the shared backend interface.

다음은 새로운 커스텀 연산을 추가하는 예시입니다:
> Here's an example of adding a new custom operation:

```cpp
class TorchCommMyBackend : public TorchCommBackend {
 public:
  std::shared_ptr<TorchWork> quantized_all_reduce(
    at::Tensor& tensor,
    ReduceOp op,
    bool async_op) {
    // 구현
  }
};

PYBIND11_MODULE(_comms_my_backend, m) {
  py::class_<TorchCommMyBackend, std::shared_ptr<TorchCommMyBackend>>(
      m, "TorchCommMyBackend")
    .def(
        "quantized_all_reduce",
        &TorchCommMyBackend::quantized_all_reduce,
        py::call_guard<py::gil_scoped_release>());
}
```

모델에서 사용하려면, `unsafe_get_backend()`을 호출하고 새로운 메서드를 호출하면 됩니다.
> To use in your model, it's as easy as calling `unsafe_get_backend()` and calling the new method.

```python
import torchcomms

comm = torchcomms.new_comm("my_backend")
backend = comm.unsafe_get_backend()

backend.quantized_all_reduce(t, ReduceOp.SUM, async_op=False)
```

프로토타이핑이 완료되면, 새로운 연산을 표준 torchcomms API에 업스트림하는 것을 환영합니다.
> Once prototyping is done, we're happy to upstream new operations into the standard torchcomms API.

### 새로운 torchcomm 백엔드 작성하기 / Writing a new torchcomm Backend

torchcomms의 핵심 기능 중 하나는 서드파티 백엔드를 훨씬 쉽게 작성할 수 있다는 것입니다. 이러한 백엔드는 더 이상 PyTorch의 일부로 빌드할 필요가 없으며, pip을 사용하여 다른 Python 확장처럼 간단히 설치할 수 있습니다.
> One of the key features of torchcomms is that it makes it much easier to write third-party backends. These backends no longer need to be built as part of PyTorch and can be simply installed like any other Python extension using pip.

![torchcomms 백엔드 구조 / torchcomms backend structure](/assets/blog/2025-10-22-torchcomms/4-2.png){:style="width:100%"}

새로운 백엔드를 작성하려면 TorchCommBackend 인터페이스를 구현해야 합니다: [https://github.com/meta-pytorch/torchcomms/blob/main/comms/torchcomms/TorchCommBackend.hpp](https://github.com/meta-pytorch/torchcomms/blob/main/comms/torchcomms/TorchCommBackend.hpp)
> To write a new backend you need to implement the TorchCommBackend interface: [https://github.com/meta-pytorch/torchcomms/blob/main/comms/torchcomms/TorchCommBackend.hpp](https://github.com/meta-pytorch/torchcomms/blob/main/comms/torchcomms/TorchCommBackend.hpp)

```cpp
// MyBackend.hpp

class MyBackend : public TorchCommBackend {
 public:
  ...
};

// MyBackend.cpp

namespace {
class MyBackendRegistration {
 public:
  MyBackendRegistration() {
    TorchCommFactory::get().register_backend(
        "my_backend", []() { return std::make_shared<MyBackend>(); });
  }
};

static MyBackendRegistration registration{};
}

// MyBackendPy.cpp

PYBIND11_MODULE(_comms_my_backend, m) {
  py::class_<MyBackend, std::shared_ptr<MyBackend>>(m, "MyBackend");
}
```

Python C 확장을 빌드한 후에는, torchcomms가 찾을 수 있도록 setup.py에 메타데이터를 추가하기만 하면 됩니다.
> Once you have your Python C extension building you then just need to add some metadata to the setup.py so torchcomms can find it.

```python
setup(
    name="my_backend",
    entry_points={
        "torchcomms.backends": [
            "my_backend = my_backend._comms_my_backend",
        ]
    },
)
```

그런 다음 `pip install` 후 다른 백엔드와 동일하게 사용할 수 있습니다.
> Then you can use it like any other backend after you `pip install` it.

```python
import torchcomms

comm = torchcomms.new_comm("my_backend", ...)
```

## 다음 단계 / Next Steps

torchcomms는 완전히 새로운 API이며 현재 활발히 개발 중입니다. 여러분의 참여를 환영하니, 사용에 관심이 있거나 개선에 기여하고 싶다면 연락해 주세요.
> Torchcomms is a brand new API and is very much under active development. We'd love for you to get involved so please reach out if you're interested in using it or want to help improve it.

이 글에서 설명한 기능들을 활발히 개발 중이며, 가까운 시일 내에 안정화하고 더 많은 디바이스에 대한 하드웨어 지원을 개선할 계획입니다.
> We're actively working on the features described in this blog post and hope to have them stabilized in the near future as well as improving hardware support for more devices.

더 자세한 문서는 [https://meta-pytorch.org/torchcomms/](https://meta-pytorch.org/torchcomms/)에서 확인할 수 있습니다.
> For more documentation check out: [https://meta-pytorch.org/torchcomms/](https://meta-pytorch.org/torchcomms/)

## 감사의 말 / Acknowledgements

torchcomms와 torchcomms-backends의 대규모 학습 및 추론 프로덕션 개발에 핵심적인 역할을 해준 많은 현직 및 전직 Meta 직원들의 기여에 감사드립니다. 특히 Tristan Rice, Pavan Balaji, Subodh Iyengar, Qiye Tan, Rodrigo De Castro, Sudharssun Subramanian, Junjie Wang, Feng Tian, Saif Hasan, Min Si, Yifan Mao, Dingming Wu, Zhaoyang Han, Blake Matheny, Art Zhu, Denis Boyda, Regina Ren, Jingyi Yang, Bingzhe Liu, Shuqiang Zhang, Mingran Yang, Cen Zhao, Adi Gangidi, Ashmitha Jeevaraj Shetty, Bruce Wu, Ching-Hsiang Chu, Yulun Wang, Srinivas Vaidyanathan, Chris Gottbrath, Davide Italiano, Shashi Gandham, Omar Baldonado, James Hongyi Zeng에게 특별히 감사드립니다.
> We would like to acknowledge the contributions of many current and former Meta employees who have played a crucial role in developing torchcomms and torchcomms-backends for large-scale training and inference in production. In particular, we would like to extend special thanks to Tristan Rice, Pavan Balaji, Subodh Iyengar, Qiye Tan, Rodrigo De Castro, Sudharssun Subramanian, Junjie Wang, Feng Tian, Saif Hasan, Min Si, Yifan Mao, Dingming Wu, Zhaoyang Han, Blake Matheny, Art Zhu, Denis Boyda, Regina Ren, Jingyi Yang, Bingzhe Liu, Shuqiang Zhang, Mingran Yang, Cen Zhao, Adi Gangidi, Ashmitha Jeevaraj Shetty, Bruce Wu, Ching-Hsiang Chu, Yulun Wang, Srinivas Vaidyanathan, Chris Gottbrath, Davide Italiano, Shashi Gandham, Omar Baldonado, James Hongyi Zeng
