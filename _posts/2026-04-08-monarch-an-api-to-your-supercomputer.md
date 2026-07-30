---
layout: blog_detail
title: "Monarch: 슈퍼컴퓨터를 다루는 API"
author: The PyTorch Team at Meta
ext_author: Junghwan Park (박정환)
image: /assets/blog/2026-04-08-monarch-an-api-to-your-supercomputer/hero.png
category: ["pytorch.org", "translation"]
date: 2026-04-08 12:00:00
org_title: "Monarch: an API to your supercomputer"
org_link: https://pytorch.org/blog/monarch-an-api-to-your-supercomputer/
---

![Monarch: 슈퍼컴퓨터를 다루는 API / Monarch: an API to your supercomputer](/assets/blog/2026-04-08-monarch-an-api-to-your-supercomputer/hero.png){:style="width:100%"}

거대한 클러스터에서 분산 학습 작업을 돌리는 일은 어렵습니다! 분산 강화 학습처럼 더 복잡한 구성을 다루기 시작하면 특히 그렇습니다. 이런 작업을 디버깅하는 일은 답답하고, 변경한 내용을 반영해 결과를 확인하기까지 걸리는 시간도 매우 긴 편입니다.
> Getting distributed training jobs to run on huge clusters is hard!  This is especially true when you start looking at more complex setups like distributed reinforcement learning. Debugging these kinds of jobs is frustrating, and the turnaround time for changes tends to be very slow.

Monarch는 간단한 Python API로 클러스터를 프로그래밍할 수 있게 해 주는 PyTorch용 분산 프로그래밍 프레임워크입니다. 슈퍼컴퓨터를 일관성 있고 직접 제어할 수 있는 하나의 시스템으로 드러내어, 마치 노트북에 GPU 수천 개가 붙어 있는 것처럼 로컬 개발의 경험을 대규모 학습으로 가져옵니다. 완전한 학습 시스템 하나를 단일 Python 프로그램으로 정의할 수 있습니다. 핵심 기본 요소(primitive)는 명시적이고 최소한으로 유지되므로, 내결함성(fault tolerance)·오케스트레이션·도구 통합 같은 상위 수준 기능은 재사용 가능한 라이브러리로 만들 수 있습니다.
> Monarch is a distributed programming framework for PyTorch that makes the cluster programmable through a simple Python API. It exposes the supercomputer as a coherent, directly controllable system—bringing the experience of local development to large-scale training, as if your laptop had 1000s of GPUs attached.  A complete training system can be defined in a single Python program. Core primitives are explicit and minimal, enabling higher-level capabilities—fault tolerance, orchestration, tooling integration—to be built as reusable libraries.

Monarch는 에이전트(agent) 활용에 맞춰 최적화되어 있습니다. 일관된 인프라 추상화를 제공하고, 에이전트가 이미 능숙하게 다루는 표준 SQL 기반 API로 텔레메트리(telemetry)를 노출합니다. 에이전트는 여러분의 개발 머신에서 돌기만 해도 많은 개발 작업을 처리할 수 있는데, Monarch는 그 개발 머신을 슈퍼컴퓨터로 바꿔 놓는 데 아주 능해서 에이전트의 역량을 한 단계 끌어올립니다.
> Monarch is optimized for agentic usage, providing consistent infrastructure abstractions and exposing telemetry via standard SQL-based APIs that agents already excel at using. Agents can do a lot of development tasks by just running on your dev machine, and Monarch is really good at turning your devmachine into a supercomputer, leveling-up those agents.

이 프로젝트는 2025년 10월 PyTorch 컨퍼런스에서 공개되었으며, 관련 내용은 [PyTorch Monarch 소개](https://pytorch.org/blog/introducing-pytorch-monarch/)에서 읽어볼 수 있습니다. 이번 글에서는 Monarch가 에이전트 주도 학습 개발을 위한 효과적인 프레임워크로 어떻게 발전해 왔는지 다룹니다. 또한 네이티브 Kubernetes 지원, RDMA 개선, 분산 텔레메트리 등 10월 이후의 주요 개선 사항도 함께 살펴봅니다.
> The project launched at the PyTorch conference in October 2025; you can read about it here: [Introducing PyTorch Monarch](https://pytorch.org/blog/introducing-pytorch-monarch/). This blog covers how Monarch has evolved into an effective framework for agent-driven training development.  It will also cover Monarch’s major improvements since October, including native Kubernetes support, RDMA improvements, distributed telemetry, and more.

## Monarch에서의 에이전트 주도 개발 / Agentic Development in Monarch

Monarch는 슈퍼컴퓨팅 클러스터를 호스트(host)·프로세스(proc)·액터(actor)라는 일관된 모델로 표현하고, 여기에 "필요한 것이 다 들어 있는(batteries included)" 인프라를 함께 제공해 여러분의 에이전트에게 초능력을 부여합니다! 에이전트는 실행 중인 코드를 직접 관리하고 디버깅하며, 의존성과 데이터를 빠르게 동기화하고, 새 코드를 실행하고, 어디에 배포되었는지와 무관하게 효율적이고 일관된 방식으로 호스트·프로세스·액터를 추가로 프로비저닝할 수 있습니다.
> By representing your supercomputing cluster through a coherent model of hosts, procs, and actors, and pairing it with “batteries included” infrastructure, Monarch gives your agent superpowers! It can directly manage and debug running code, rapidly sync dependencies and data, run new code, and provision additional hosts, procs, and actors in an efficient and consistent way regardless of where it is deployed.

에이전트 주도 개발을 뒷받침하기 위해 Monarch가 사용하는 주요 기능을 빠르게 살펴봅시다:

- RDMA 기반 원격 파일 시스템 – 클라이언트의 파일을 읽기 전용으로 마운트한 파일 시스템 형태로 RDMA를 통해 작업(job) 내 모든 호스트에 배포합니다. 이를 통해 머신러닝 아이디어를 반복 실험하는 동안 코드·의존성·컨테이너를 매우 빠르게 동기화할 수 있습니다. Monarch의 RDMA 파일 시스템은 다시 Monarch RDMA 버퍼와 PyFuse 위에 만들어졌습니다.
- 분산 SQL 텔레메트리 – Monarch에 통합된 경량 분산 SQL 엔진으로 모든 분산 프로세스·액터 등에서 실시간 상태 정보, pyspy 트레이스, 로그를 수집합니다. Monarch를 사용해 DataFusion 분산 SQL 쿼리 엔진을 *현장에서(in situ)* 직접 실행했습니다. 각 노드는 실시간 상태 정보를 여러 테이블에 기록하고, 에이전트는 그 테이블을 직접 효율적으로 조회할 수 있습니다. 덕분에 디버깅할 때 시스템 상태를 살펴보기가 매우 쉬워집니다.
- Jobs API – 리소스(호스트)를 한 번만 프로비저닝해 두고, 할당을 반복하는 비용을 치르지 않고 필요한 만큼 작업을 실행합니다. Monarch는 Kubernetes와 SLURM을 기본으로 지원하며, 다른 스케줄러는 Monarch Job을 구현해 통합할 수 있습니다.

> Let’s quickly review some key features Monarch uses to empower agentic development:
> - RDMA-Powered Remote File System – Distribute files from the client on a read-only mounted filesystem to every host in the job via RDMA.  This lets you very rapidly sync code, dependencies, and containers while iterating on the machine learning ideas. Monarch’s RDMA filesystem in turn is built on Monarch RDMA buffers and PyFuse.
> - Distributed SQL Telemetry – Use Monarch’s integrated lightweight distributed SQL engine to collect live state information, pyspy traces, and logs from all distributed processes/actors/etc. We used Monarch to directly run a DataFusion distributed SQL query engine \*in situ\*; each node in turn writes live state information into a set of tables that can then be queried directly and efficiently by an agent.  This makes it very easy to explore the state of the system when debugging.
> - Jobs API – Provision resources (hosts) once and run as many jobs as needed on them without paying the repeated allocation penalty. Monarch comes with support for Kubernetes and SLURM; other schedulers can be integrated by implementing a Monarch Job.

이 기능들이 모이면 에이전트는 개발의 주요 국면 전반에서 효율적으로 움직일 수 있습니다. 한 곳에서 작업을 빠르게 재시작하고, 새 코드·의존성·데이터를 빠르게 동기화하고, 빠르게 디버깅할 수 있습니다. 요컨대 Monarch는 분산 시스템을 로컬처럼 느껴지게 만들고, 문제를 풀 때 반복 주기를 줄여 주는 도구 상자를 제공합니다.
> Collectively, these features enable agents to be efficient across some key phases of development; they can restart jobs fast, sync new code, dependencies, and data fast, and debug fast, all from a central point.  In short, Monarch makes the distributed system feel local and provides a toolbox to reduce the iteration time when tackling problems.

## Monarch에 새로 생긴 것 / What’s new in Monarch?

2025년 10월 PyTorch 컨퍼런스에서 공개된 이후(약 6개월 전) Monarch에 무엇이 새로 생겼는지 살펴봅시다.
> Let’s review what is new in Monarch since its launch at the PyTorch Conference in October 2025 (~6 months ago).

### Kubernetes

이제 Monarch는 Kubernetes를 일급(first-class)으로 지원합니다.
> Monarch now has first-class Kubernetes support.

- Monarch-kubernetes 오픈소스 저장소 – MonarchMesh 커스텀 리소스 정의(Custom Resource Definition), 참조용 KubeBuilder 오퍼레이터, hello-world 데모를 담은 전용 저장소([github.com/meta-pytorch/monarch-kubernetes](https://github.com/meta-pytorch/monarch-kubernetes))입니다. MonarchMesh 라벨 전파 덕분에 Kueue를 통한 스케줄링도 가능합니다.
- 적시(just-in-time) 파드 프로비저닝 – 파드를 미리 예약해 두는 대신 필요할 때 할당해 클러스터 활용률을 높입니다.
- 외부 게이트웨이 – 클러스터 외부의 클라이언트도 Kubernetes 안에서 실행 중인 Monarch 메시(mesh)에 연결할 수 있습니다(0.5에 포함됩니다).
- 버전이 매겨진 Docker 컨테이너와 nightly 컨테이너 – 재현 가능한 배포를 위해 [GHCR](https://github.com/meta-pytorch/monarch/pkgs/container/monarch)에 게시됩니다.

> - Monarch-kubernetes OSS repository – A dedicated repo ([github.com/meta-pytorch/monarch-kubernetes](https://github.com/meta-pytorch/monarch-kubernetes)) with a MonarchMesh Custom Resource Definition, a reference KubeBuilder operator, and a hello-world demo.  The MonarchMesh label propagation also enables scheduling via Kueue.
> - Just-in-time pod provisioning – Pods are allocated on demand rather than reserved upfront, improving cluster utilization.
> - External gateway – Out-of-cluster clients can now connect to Monarch meshes running inside Kubernetes (landing in 0.5).
> - Versioned and nightly Docker containers – Published to [GHCR](https://github.com/meta-pytorch/monarch/pkgs/container/monarch) for reproducible deployments.

### RDMA와 네트워킹 / RDMA & Networking

Monarch는 RDMA에 대한 투자를 이어 가며 여러 새로운 백엔드를 지원하고, 이들을 지원하고 사용하기 더 쉽게 만드는 상위 수준 API를 제공하고 있습니다.
> Monarch has continued its investment in RDMA, adding support for multiple new backends and providing a higher-level API to make supporting and using them easier.

- AWS EFA RDMA 지원 – Monarch의 RDMABuffer가 이제 AWS의 Elastic Fabric Adapter(EFA)를 지원하면서, 고성능 네트워킹이 InfiniBand를 넘어 확장됩니다. 16Gbps에서 검증되었으며 TCP보다 10배 빠릅니다(14.5GB를 7.6초에 전송). PyPI nightly 빌드에서 사용할 수 있습니다.
- AMD ROCm GPU 지원 – 이제 GPU-direct RDMA와 RCCL 집합 통신(collective communication)이 Mellanox 인터페이스를 갖춘 AMD GPU에서 ROCm을 통해 동작합니다.
- 통합 RDMA API – InfiniBand(mlx5), AWS EFA, ROCm에서 모두 동작하는 하드웨어 이식 가능한 RDMA 인터페이스입니다. 한 번 작성해 어떤 패브릭(fabric)에서든 실행할 수 있고, 사용할 수 없는 환경에서는 Monarch 액터 메시징으로 대체됩니다.

> - AWS EFA RDMA support – Monarch’s RDMABuffer now supports Elastic Fabric Adapter (EFA) on AWS, extending high-performance networking beyond InfiniBand. Validated at 16 Gbps – 10x faster than TCP (14.5 GB in 7.6 seconds). Available in PyPI nightlies.
> - AMD ROCm GPU support – GPU-direct RDMA and RCCL collective communication now work on AMD GPUs via ROCm with Mellanox interfaces.
> - Unified RDMA API – A hardware-portable RDMA interface that works across InfiniBand (mlx5), AWS EFA, and ROCm, letting users write once and run on any fabric, or fall back to Monarch actor messaging when not available.

### 관측 가능성과 텔레메트리 / Observability & Telemetry

Monarch는 관측 가능성(observability)과 텔레메트리에 크게 힘을 실어, 에이전트 주도 개발을 뒷받침하는 프로그램적 메커니즘을 추가했습니다. 새로운 네이티브 대시보드와 터미널 UI(TUI), 그리고 DevOps 팀이 흔히 쓰는 오픈소스 표준 지원도 함께 들어왔습니다.
> Monarch has leaned heavily into observability & telemetry, adding programmatic mechanisms to empower agentic development.  There are also new native dashboards, Terminal UI (TUIs), and support for OSS standards commonly used by DevOps teams.

- 분산 SQL 텔레메트리 – 클라이언트에서 접근할 수 있는 SQL 엔드포인트로, 서드파티 의존성 없이 분산 시스템을 손쉽게 분석할 수 있습니다.
- Admin API와 터미널 UI – 실행 중인 Monarch 작업을 살펴보고 관리하는 터미널 기반 인터페이스로, 내부 상태에 접근하는 강력한 API가 뒷받침합니다.
- **OpenTelemetry 통합** – 지표·로그·시각화에 대한 네이티브 OTel 지원을 Kubernetes에서 제공해, 어떤 클러스터에서든 완전한 관측 가능성을 확보합니다. Prometheus, Loki, Grafana 등 흔히 쓰이는 오픈소스 도구와 쉽게 연동됩니다.
- **작업별 오픈소스 대시보드(베타)** – 외부 도구 없이 분산 작업을 시각화하고 디버깅할 수 있는 내장 웹 대시보드입니다.

> - Distributed SQL Telemetry – A client-accessible SQL endpoint, enabling easy analysis of the distributed system without 3rd party dependencies.
> - Admin API & Terminal UI – A terminal-based interface for inspecting and managing live Monarch jobs, backed by a powerful API for accessing internals.
> - **OpenTelemetry integration** – Native OTel support for metrics, logs, and visualization on Kubernetes, giving users full observability on any cluster.  This is easily integrated with Prometheus, Loki, Grafana, and other common OSS tooling.
> - **Per-job OSS dashboard (Beta)** – A built-in web dashboard for visualizing and debugging distributed jobs without external tooling.

![프로세스 토폴로지와 py-spy 트레이스를 보여주는 Monarch 터미널 UI / Monarch terminal UI showing process topology and py-spy traces](/assets/blog/2026-04-08-monarch-an-api-to-your-supercomputer/pytblog_040626a.png){:style="width:100%"}

![호스트·프로세스·액터 계층을 보여주는 Monarch 대시보드의 DAG 화면 / Monarch dashboard DAG view of hosts, procs, and actors](/assets/blog/2026-04-08-monarch-an-api-to-your-supercomputer/pytblog_040626b.png){:style="width:100%"}

### 이식성과 설치 / Portability & Installation

이제 Monarch는 훨씬 더 작아지고 시작도 훨씬 빨라져서, 어느 때보다 쓰기 쉬워졌습니다.
> Monarch is now significantly more compact and much faster to start, making it easier to use than ever.

- 설치 용량 100배 감소, 시작 속도 8배 향상 – pip wheel 용량이 100분의 1 수준으로 줄었고 콜드 스타트 시간도 극적으로 빨라졌습니다. libpython 링크 요구 사항은 완전히 제거되었습니다.
- torch 의존성 제거 – v0.2부터 torchmonarch는 더 이상 torch를 pip 의존성으로 끌어오지 않아 설치가 단순해지고 버전 충돌도 피할 수 있습니다.
- 네이티브 uv 지원 – Monarch는 빠른 Python 패키지 관리자인 [uv](https://github.com/astral-sh/uv)와 별도 설정 없이 바로 동작합니다. `git clone`, `cd`, `uv run example.py` 세 개의 명령으로 시작할 수 있습니다. [예제 저장소](https://github.com/allenwang28/monarch-uv)를 참고하세요.
- PyPI 패키징 통합 – 모든 패키지가 torchmonarch라는 하나의 이름으로 통합되었고, nightly 빌드는 PEP 440 사전 배포 버전 규칙을 따릅니다: `pip install --pre torchmonarch`. v0.4에는 ARM64 Linux 빌드도 추가되었습니다.

> - 100x smaller install, 8x faster startup – The pip wheel footprint was reduced by two orders of magnitude with dramatically faster cold-start times. libpython linking requirements were removed entirely.
> - Torch dependency removed – As of v0.2, torchmonarch no longer pulls in torch as a pip dependency, simplifying installation and avoiding version conflicts.
> - Native uv support – Monarch works out of the box with [uv](https://github.com/astral-sh/uv), the fast Python package manager. Three commands to get started: git clone, cd, uv run example.py. See the [example repo](https://github.com/allenwang28/monarch-uv).
> - Consolidated PyPI packaging – All packages unified under a single torchmonarch name with PEP 440 pre-release versioning for nightlies: pip install --pre torchmonarch. ARM64 Linux builds are added as well to v0.4

### 개발자 경험 / Developer Experience

- 대화형 SPMD – SPMD(Single Program, Multiple Data) 작업으로 하는 대화형·노트북 스타일 개발 지원이 개선되었습니다.
- RDMA 파일 시스템 – 호스트 간 파일 동기화가 빠르고 편리합니다.

> - Interactive SPMD – Improved support for interactive, notebook-style development with SPMD (Single Program, Multiple Data) jobs.
> - RDMA File System – Fast, convenient file-sync across hosts.

## 협업 / Collaborations

출시 이후 Monarch를 더 좋게 만드는 데 힘을 보태 준 협업자들에게도 잠시 감사를 전하고 싶습니다.
> We’d also like to take a moment to acknowledge some collaborators that have helped make Monarch better since its release.

- [**SkyPilot**](https://docs.skypilot.co/en/stable/docs/index.html)
  - 명령 하나로 어떤 Kubernetes 클러스터에서든 Monarch 실행 – SkyPilot 통합을 사용하면 Monarch 코드를 바꾸지 않고도 어떤 K8s 클러스터나 클라우드에서든 `sky launch`로 Monarch 워크로드를 띄울 수 있습니다. GPU가 있는 곳이면 어디서든 써야 하는 팀에 좋습니다.
  - 인프라 설정 없이 다중 노드 분산 학습 – SkyPilot이 노드 프로비저닝, 네트워킹, 갱 스케줄링(gang scheduling)을 처리하므로 사용자는 Monarch 학습 로직에만 집중할 수 있습니다. 이 통합은 Monarch의 JobTraits API를 사용해 SkyPilot을 작업 백엔드로 연결합니다. k8s 클러스터에 별도의 오퍼레이터를 설치할 필요가 없습니다.
  - 자세한 내용은 [https://github.com/meta-pytorch/monarch/tree/main/examples/skypilot](https://github.com/meta-pytorch/monarch/tree/main/examples/skypilot)을 참고하세요.

> - [**SkyPilot**](https://docs.skypilot.co/en/stable/docs/index.html)
>   - Run Monarch on any Kubernetes cluster with a single command – the SkyPilot integration lets users sky launch Monarch workloads on any K8s cluster or cloud without changing their Monarch code. Great for teams that need GPUs wherever they’re available.
>   - Multi-node distributed training with zero infra setup – SkyPilot handles node provisioning, networking, and gang scheduling so users can focus on their Monarch training logic. The integration uses Monarch’s JobTraits API to plug into SkyPilot as the job backend. No need to install separate operators on your k8s clusters.
>   - See  [https://github.com/meta-pytorch/monarch/tree/main/examples/skypilot](https://github.com/meta-pytorch/monarch/tree/main/examples/skypilot) for more.

- [**VERL**](https://github.com/volcengine/verl)
  - VERL은 분산 RLHF 사후 학습(post-training)에 널리 쓰이는 오픈소스 프레임워크입니다. ByteDance의 VeRL 팀과 협업해 VeRL의 단일 컨트롤러(single-controller) 아키텍처를 위한 Monarch 백엔드를 개발했습니다. 여기에는 Monarch의 Job API 위에 만든 새로운 리소스 풀 추상화, 여러 역할을 한 곳에 배치(colocated)하는 워커 지원, VeRL의 DataProto 교환 패턴에서 텐서를 대역 외(out-of-band)로 옮기는 RDMA 기반 전송 계층, 그리고 전역 액터 레지스트리에 의존하지 않고 액터 핸들 탐색 문제를 해결하는 vLLM 서버 통합이 포함됩니다. VeRL의 하이브리드 엔진(hybrid-engine) 학습 모드를 사용해, 이 백엔드를 통해 VeRL의 PPO·GRPO 학습 루프가 Monarch에서 실행될 수 있음을 검증했고, 성능 저하 없이 수치적으로 동일한 결과를 얻었습니다. 이 작업에서 얻은 발견 하나는, VeRL의 단일 컨트롤러 인터페이스는 깔끔하게 추상화되어 있지만 Ray API 사용이 코드베이스 전반에 드러나 있다는 점입니다 — 그래서 침습적이지 않은 백엔드 교체는 인터페이스만 보고 짐작하는 것보다 손이 더 많이 갑니다. 이는 Ray 위에 만들어진 프레임워크에서 흔히 나타나는 양상이며, Monarch와 VeRL 커뮤니티가 시간을 두고 함께 풀어갈 수 있는 문제입니다.

> - **[VERL](https://github.com/volcengine/verl)**
>   - VERL is a popular open-source framework for distributed RLHF post-training. In collaboration with ByteDance’s VeRL team, we developed a Monarch backend for VeRL’s single-controller architecture, implementing new resource pool abstractions built on Monarch’s Job API, colocated multi-role worker support, an RDMA-based transport layer that moves tensors out-of-band for VeRL’s DataProto exchange pattern, and a vLLM server integration that solves actor handle discovery without relying on a global actor registry. We validated that VeRL’s PPO and GRPO training loops can run on Monarch through this backend using VeRL’s hybrid-engine training mode, producing numerically identical results with no performance regression. One finding from this work: while VeRL’s single-controller interface is cleanly abstracted, Ray API usage surfaces throughout the broader codebase — making a non-invasive backend swap more involved than the interface alone suggests. This is a common pattern in frameworks built on Ray, and something the Monarch and VeRL communities can collaborate on over time.

- **AMD**
  - Monarch는 AMD를 지원 플랫폼으로 추가하며 주요 하드웨어 인프라 전반으로 호환성과 성능을 확장했습니다. AMD의 파트너들이 자사 ROCm 플랫폼에서 Monarch를 검증해, MI300/325/355 클러스터에서 SLURM 기반 오케스트레이션이 매끄럽게 동작하도록 했습니다. 이 통합으로 HPC와 AI 연구에서 널리 쓰이는 익숙한 SLURM 생태계를 활용해 AMD GPU 전반에서 AI 워크로드를 효율적으로 스케줄링하고 관리하며 확장할 수 있습니다.
  - 이들의 노력 덕분에 이제 Monarch는 Mellanox 네트워크 인터페이스를 갖춘 AMD 클러스터에서 GPU 간 빠른 통신을 위한 RDMA(Remote Direct Memory Access)를 지원합니다. 이 하드웨어 조합은 Azure나 Oracle 같은 주요 클라우드 제공업체에서 사용할 수 있으며, 분산 강화 학습과 대규모 AI 워크로드에 필수적인 고처리량·저지연 데이터 전송을 가능하게 합니다.

> - **AMD**
>   - Monarch expanded its compatibility and performance across leading hardware infrastructure adding AMD as a supported platform. Our partners at AMD validated Monarch on their ROCm platform, enabling seamless SLURM-based orchestration for MI300/325/355 clusters. This integration allows users to efficiently schedule, manage, and scale AI workloads across AMD GPUs, leveraging the familiar SLURM ecosystem widely used in HPC and AI research.
>   - Thanks to their effort, Monarch now supports RDMA (Remote Direct Memory Access) for fast GPU-to-GPU communication on AMD clusters equipped with Mellanox network interfaces. This hardware combination is available on major cloud providers like Azure and Oracle, enabling high-throughput, low-latency data transfers essential for distributed RL training and large-scale AI workloads.

## 마치며 / Conclusion

Monarch는 여러분의 슈퍼컴퓨터를 다루는 API로, 분산 AI 개발을 로컬 앱을 만드는 일처럼 느껴지게 합니다. AI 학습의 미래는 속도와 단순함을 요구하며, Monarch는 사람과 에이전트 모두에게 그 둘을 제공합니다. 최신 기능을 살펴보고, 성장하고 있는 오픈소스 우선 개발 커뮤니티에 참여해 분산 컴퓨팅의 다음 장을 함께 만들어 가기를 권합니다.
> Monarch is the API for your supercomputer; making distributed AI development feels like building a local app. The future of AI training demands speed and simplicity Monarch provides for both humans and agents. We encourage you to explore the latest features, join our growing OSS-first development community, and help shape the next chapter of distributed computing.

## 감사의 말 / Acknowledgments

이 작업을 가능하게 해 준 Monarch 팀 전체에 감사드립니다. 또한 GitHub의 [주요 기여자들](https://github.com/meta-pytorch/monarch/graphs/contributors)에게 특별히 감사드립니다!
> Thank you to the whole Monarch team for making this work possible.  Also, a special thanks to our [Top Contributors](https://github.com/meta-pytorch/monarch/graphs/contributors) on GitHub!

- 🙏 Monarch를 Kubernetes와 통합하는 데 도움을 준 Google Cloud와 Runhouse의 파트너들, 그리고 기여해 준 SkyPilot과 AMD의 파트너들에게 특별히 감사드립니다!

> - 🙏 Special thanks to our partners at Google Cloud and Runhouse for helping integrate monarch with kubernetes, and to our partners at SkyPilot and AMD for their contributions!

Ahmad Sharif, Allen Wang, Ali Sol, Amir Afzali, Carole-Jean Wu, Chris Gottbrath, Christian Puhrsch, Colin Taylor, Do Hyung (Dave) Kwon, Gayathri Aiyer, Hamid Shojanazeri, Jiyue Wang, Joe Spisak, John William Humphreys, Jun Li, Lucas Pasqualin, Marius Eriksen, Matthew Zhang, Matthias Reso, Peng Zhang, Riley Dulin, Rithesh Baradi, Robert Rusch, Sam Lurye, Samuel Hsia, Shayne Fletcher, Tao Lin, Thomas Wang, Victoria Dudin, Zachary DeVito
