---
layout: blog_detail
title: "PyTorch 2.12 출시 공지"
author: PyTorch Team
category: ["pytorch.org", "translation"]
org_title: "PyTorch 2.12 Release Blog"
org_link: https://pytorch.org/blog/pytorch-2-12-release-blog/
---

PyTorch 2.12([릴리즈 노트](https://github.com/pytorch/pytorch/releases/tag/v2.12.0))의 출시를 발표하게 되어 기쁩니다!
> We are excited to announce the release of PyTorch® 2.12 ([release notes](https://github.com/pytorch/pytorch/releases/tag/v2.12.0))!

PyTorch 2.12 릴리즈에는 다음과 같은 변경 사항이 포함되어 있습니다:
> The PyTorch 2.12 release features the following changes:

- CUDA에서의 배치 `linalg.eigh`가 cuSolver 백엔드 선택 방식 개선으로 최대 100배 빨라졌습니다
- 새로운 `torch.accelerator.Graph` API가 CUDA, XPU 및 트리 외(out-of-tree) 백엔드 전반에 걸쳐 그래프 캡처와 재생을 통합합니다
- `torch.export.save`가 마이크로스케일링(Microscaling, MX) 양자화 형식을 지원하여, 공격적으로 압축된 모델의 완전한 내보내기(export)가 가능합니다
- Adagrad가 `fused=True`를 지원하게 되어 단일 커널 옵티마이저 구현을 제공하는 Adam, AdamW, SGD에 합류합니다
- `torch.cond` 제어 흐름을 이제 CUDA 그래프 내부에서 캡처하고 재생할 수 있습니다
- ROCm 사용자에게 확장 가능한 메모리 세그먼트(expandable memory segments), rocSHMEM 대칭 메모리 집합 통신(symmetric memory collectives), FlexAttention 파이프라이닝이 제공됩니다

> - Batched `linalg.eigh` on CUDA is up to 100x faster due to updated cuSolver backend selection
> - New `torch.accelerator.Graph` API unifies graph capture and replay across CUDA, XPU, and out-of-tree backends
> - `torch.export.save` now supports Microscaling (MX) quantization formats, enabling full export of aggressively compressed models
> - Adagrad now supports `fused=True`, joining Adam, AdamW, and SGD with a single-kernel optimizer implementation
> - `torch.cond` control flow can now be captured and replayed inside CUDA Graphs
> - ROCm users gain expandable memory segments, rocSHMEM symmetric memory collectives, and FlexAttention pipelining

이번 릴리즈는 PyTorch 2.11 이후 457명의 기여자로부터 2,926회의 커밋으로 구성되었습니다. 헌신적인 커뮤니티의 기여에 진심으로 감사드립니다. 언제나 그렇듯, 새로운 버전을 사용하시며 생기는 문제를 보고해주시면 PyTorch 2.12를 개선하는 데 도움이 됩니다. PyTorch 2 시리즈를 시작하는 방법에 대한 자세한 정보는 [시작하기](https://pytorch.org/get-started/pytorch-2-x/) 페이지에서 확인할 수 있습니다.
> This release is composed of 2,926 commits from 457 contributors since PyTorch 2.11. We want to sincerely thank our dedicated community for your contributions. As always, we encourage you to try these out and report any issues as we improve 2.12. More information about how to get started with the PyTorch 2-series can be found at our [Getting Started](https://pytorch.org/get-started/pytorch-2-x/) page.

궁금한 점이 있으신가요? 5월 20일 수요일 오전 10시(PST)에 진행되는 라이브 Q&A에 참여해보세요. Joe Spisak, Andrey Talman, Alban Desmaison이 패널로, Chris Gottbrath가 모더레이터로 참여합니다. 이번 릴리즈에 대한 간략한 소개와 함께 여러분의 질문에 실시간으로 답변해 드립니다. [지금 등록하기](https://pytorch.org/event/pytorch-2-12-release-live-qa/)
> Have questions? Join us on Wednesday, May 20 at 10 am PST for a live Q&A with panelists Joe Spisak, Andrey Talman, and Alban Desmaison, and moderator Chris Gottbrath. We will provide a brief overview of the release and answer your questions live. [Register now.](https://pytorch.org/event/pytorch-2-12-release-live-qa/)

2.x 시리즈 전반에 걸쳐 PyTorch는 연구 중심의 프레임워크에서, 대규모 프로덕션 학습 및 추론을 위한 통합된 하드웨어에 구애받지 않는(hardware-agnostic) 플랫폼으로 진화해 왔습니다. [PyTorch 2.10](https://pytorch.kr/blog/2026/pytorch-2-10-release-blog/)은 백엔드 간 성능 프리미티브와 TorchScript의 공식 지원 중단(deprecation)으로 그 토대를 마련했습니다. [PyTorch 2.11](https://pytorch.kr/blog/2026/pytorch-2-11-release-blog/)은 분산 학습을 위한 미분 가능한 집합 통신(differentiable collectives), 차세대 GPU에서의 FlashAttention-4, 더 광범위한 내보내기(export) 지원으로 그 기반을 확장했습니다.
> Throughout the 2.x series, PyTorch has been evolving from a research-first framework into a unified, hardware-agnostic platform for production training and inference at scale. [PyTorch 2.10](https://pytorch.org/blog/pytorch-2-10-release-blog/) laid the groundwork with cross-backend performance primitives and the formal deprecation of TorchScript. [PyTorch 2.11](https://pytorch.org/blog/pytorch-2-11-release-blog/) expanded that foundation with differentiable collectives for distributed training, FlashAttention-4 on next-generation GPUs, and broader export coverage.

PyTorch 2.12는 이러한 방향성을 이어갑니다: 새로운 장치에 구애받지 않는(device-agnostic) `torch.accelerator.Graph` API가 CUDA, XPU 및 트리 외 백엔드 전반에 걸쳐 그래프 캡처와 재생을 통합하고, 배치 고윳값 분해(eigenvalue decomposition)는 최대 100배 빨라졌으며, `torch.export`가 마이크로스케일링 양자화 형식을 지원하여 공격적으로 압축된 모델 배포가 가능해졌습니다. 이러한 일련의 릴리즈를 통해 PyTorch는 여러 백엔드에서 더 빨라지고 더 다양한 플랫폼에서 사용 가능해지면서, AI 혁신을 지속적으로 뒷받침하고 있습니다.
> PyTorch 2.12 continues this direction: a new device-agnostic torch.accelerator.Graph API unifies graph capture and replay across CUDA, XPU, and out-of-tree backends; batched eigenvalue decomposition is up to 100x faster; and torch.export now supports Microscaling quantization formats for deploying aggressively compressed models. Across these releases, PyTorch is becoming faster across backends and usable in a wider variety of platforms as it continues to enable AI innovation.

## 성능 관련 기능 / Performance Features

**CUDA에서 배치 고윳값 분해(`linalg.eigh`) 최대 100배 향상 / Up to 100x faster batched eigendecomposition on CUDA (`linalg.eigh`)**

CUDA에서의 `linalg.eigh` 백엔드 선택 방식이 전면 개편되었습니다. 레거시 MAGMA 백엔드는 cuSolver로 대체되었고(Grayson Derossi의 PR #174619), cuSolver 디스패치 휴리스틱은 `syevj_batched`를 무조건적으로 사용하도록 업데이트되었습니다(Johannes Z의 PR #175403). 배치 대칭/에르미트(Hermitian) 고윳값 문제에 대해 이전 릴리즈 대비 최대 100배의 속도 향상을 제공하며, CuPy와의 오랜 성능 격차가 해소되었습니다.
> The backend selection for linalg.eigh on CUDA has been overhauled. The legacy MAGMA backend was deprecated in favor of cuSolver (PR #174619 by Grayson Derossi), and the cuSolver dispatch heuristics were updated to use syevj\_batched unconditionally (PR #175403 by Johannes Z). For batched symmetric/Hermitian eigenvalue problems, this yields up to 100x speedups over the previous release, resolving longstanding performance gaps with CuPy.

이전에 수 분이 걸리던 워크로드(PyTorch가 각 행렬 풀이를 개별적으로 비효율적으로 디스패치했기 때문)가 이제는 cuSolver의 `syevj_batched` 커널을 사용하여 수 초 만에 실행됩니다. 이 커널은 많은 수의 소/중형 행렬을 단일 GPU 연산으로 처리하도록 설계되었습니다. 이러한 성능 향상은 배치 행렬의 고윳값 분해에 의존하는 과학 계산 및 머신러닝 워크로드에 특히 의미가 큽니다. ([문서의 사용 예시](https://docs.pytorch.org/docs/2.11/generated/torch.linalg.eigh.html))
> Workloads which previously took minutes (because PyTorch was inefficiently dispatching each matrix solve individually) now run in seconds by using cuSolver's syevj\_batched kernel, which is designed to process many small/medium matrices as a single GPU operation. These gains are especially relevant for scientific computing and machine learning workloads that rely on eigendecompositions of batched matrices. ([example usage in the doc](https://docs.pytorch.org/docs/2.11/generated/torch.linalg.eigh.html))

**Fused Adagrad 옵티마이저 / Fused Adagrad optimizer**

Adagrad 옵티마이저가 이제 `fused=True`를 지원하여, 각 연산마다 별도의 커널을 실행하는 대신 옵티마이저의 전체 단계(step)를 단일 CUDA 커널로 수행합니다. 이는 커널 실행 오버헤드와 메모리 트래픽을 줄여줍니다. Adagrad는 Adam, AdamW, SGD와 함께 fused 변형을 제공하는 옵티마이저 대열에 합류합니다. 기반이 되는 CUDA 커널은 2.11 사이클에서 @MeetThePatel이 기여했으며(PR #159008), 이를 사용자에게 노출하는 Python 프론트엔드는 2.12에서 Jane Xu가 마무리했습니다(PR #177672).
> The Adagrad optimizer now supports fused=True, performing the entire optimizer step in a single CUDA kernel rather than launching separate kernels for each operation. This reduces kernel launch overhead and memory traffic. Adagrad joins Adam, AdamW, and SGD in offering a fused variant. The underlying CUDA kernel was contributed by @MeetThePatel in the 2.11 cycle (PR #159008), with the Python frontend exposing it to users finalized by Jane Xu in 2.12 (PR #177672).

## 하드웨어 전반의 컴파일 및 내보내기 / Compilation and export across hardware

**`torch.accelerator.Graph`: 장치에 구애받지 않는 가속기 그래프 캡처 및 스트림 API / `torch.accelerator.Graph`: Device Agnostic Accelerator Graph Capture and Stream API**

`torch.accelerator.Graph`는 그래프 캡처와 재생을 위한 새로운 장치에 구애받지 않는(device-agnostic) API로, `torch.xpu.XPUGraph`와 같은 백엔드별 구현체 위에 통합된 추상화 계층을 제공합니다. 각 백엔드는 경량 `GraphImplInterface`를 통해 자체 구현체를 등록할 수 있어, 백엔드의 자율성을 유지하면서도 일관된 사용자 대상 API를 제공할 수 있습니다.
> `torch.accelerator.Graph` is a new device-agnostic API for graph capture and replay, providing a unified abstraction over backend-specific implementations such as `torch.xpu.XPUGraph`. Each backend can register its own implementation through a lightweight GraphImplInterface, preserving backend autonomy while enabling a consistent user-facing API.

이와 함께, `c10::Stream`과 `torch.Stream`은 이제 `is_capturing()` 메서드를 노출하여, 디바이스별 `is_current_stream_capturing`을 백엔드에 구애받지 않는 대안으로 대체합니다. 스트림 컨텍스트 매니저의 재진입(reentrance) 문제도 수정되었습니다. 이러한 변경 사항을 통해 백엔드 간 스트림 및 그래프 관리에 균등한 지원이 이루어지며, XPU 백엔드에 대한 초기 지원과 함께 `PrivateUse1`을 통해 트리 외 백엔드로도 확장이 가능합니다.
Guangye Yu(Intel)가 6개의 PR을 통해 기여했으며, 그중 C++ 인터페이스(PR #171269)와 Python 프론트엔드(PR #171285)가 핵심입니다. ([docstring의 사용 예시](https://github.com/pytorch/pytorch/blob/1d803512199040e98738e95d0dc074acbde9fb5c/torch/accelerator/graphs.py#L11-L48))
> Alongside this, `c10::Stream` and `torch. Stream` now exposes an `is_capturing()` method, replacing the device-specific `is_current_stream_capturing` with a backend-agnostic alternative. Stream context manager reentrance was also fixed. Together, these changes bring cross-backend parity to stream and graph management, with initial support for the XPU backend and extensibility to out-of-tree backends via `PrivateUse1`.
> Contributed by Guangye Yu (Intel) across six PRs, anchored by the C++ interface (PR #171269) and Python frontend (PR #171285). ([usage example in docstring](https://github.com/pytorch/pytorch/blob/1d803512199040e98738e95d0dc074acbde9fb5c/torch/accelerator/graphs.py#L11-L48))

**`torch.export`가 이제 마이크로스케일링(MX) 양자화 형식 지원 / `torch.export` now supports Microscaling (MX) quantization formats**

모델이 연구에서 프로덕션으로 이동함에 따라 `torch.export`는 배포를 위해 PyTorch 모델을 직렬화하는 표준 경로가 되었습니다. 그러나 마이크로스케일링(Microscaling, MX) 양자화 — 모델 크기와 추론 비용을 줄이기 위해 점점 인기를 얻고 있는 기법 — 를 사용하는 모델은 이전에는 내보낼 수 없었습니다. `torch.export.save`가 MX 형식(MXFP4, MXFP6, MXFP8)에서 공유 블록 스케일 지수(shared block-scale exponent)로 사용되는 `float8_e8m0fnu` dtype을 처리하지 못했기 때문입니다.
> As models move from research to production, `torch.export` is the standard path for serializing PyTorch models for deployment. However, models using Microscaling (MX) quantization — an increasingly popular technique for reducing model size and inference cost — could not previously be exported because `torch.export.save` did not handle the `float8_e8m0fnu` dtype used as the shared block-scale exponent in MX formats (MXFP4, MXFP6, MXFP8).

PyTorch 2.12에서는 `torch.export.save`와 `torch.export.load`가 이 dtype을 가진 Tensor를 올바르게 직렬화 및 역직렬화하여, 마이크로스케일링 양자화를 활용하는 모델에 대한 전체 내보내기-배포(export-to-deployment) 워크플로우의 막힘을 해소했습니다. 이는 비용 제약이 있거나 엣지 환경에 대규모 언어 모델을 배포하는 팀에게 특히 유용합니다. 이러한 환경에서는 공격적인 양자화가 필수적입니다. Chizkiyahu Raful(ARM)이 기여했습니다(PR #176270).
> In PyTorch 2.12, `torch.export.save` and `torch.export.load` now correctly serialize and deserialize tensors with this dtype, unblocking the full export-to-deployment workflow for models leveraging Microscaling quantization. This is particularly relevant for teams deploying large language models to cost-constrained or edge environments where aggressive quantization is essential. Contributed by Chizkiyahu Raful (ARM) (PR #176270).

**CUDA 그래프 내에서 torch.cond로 제어 흐름 캡처 / Capture Control flow with torch.cond within CUDA Graph**

`torch.cond`를 사용하는 제어 흐름(control-flow) 영역을 이제 CUDA 그래프의 일부로 캡처하고 재생할 수 있습니다. 이전에는 데이터에 의존적인 제어 흐름은 분기 평가가 CPU에서 이루어졌기 때문에 CUDA 그래프 트리(graph trees)로 대체(fallback)되어야 했습니다. CUDA 12.4의 조건부 IF 노드를 활용함으로써, `torch.cond` 분기는 이제 단일 그래프 캡처 내에서 전적으로 GPU에서 평가됩니다.
> Control-flow regions using torch.cond can now be captured and replayed as part of CUDA Graphs. Previously, data-dependent control flow forced fallback to CUDA graph trees because branching was evaluated on the CPU. By leveraging CUDA 12.4's conditional IF nodes, torch.cond branches are now evaluated entirely on the GPU within a single graph capture.

이는 Daniel Galvez와 Ting-Yang Kuei(NVIDIA)가 기여했으며(PR #168912), Paul Zhang(Meta)이 Inductor 순서 지정(ordering) 지원을 추가했습니다(PR #179457). 현재는 eager 및 cudagraphs 백엔드에서 동작하며, Inductor 지원은 향후 릴리즈에 추가될 예정입니다.
> This was contributed by Daniel Galvez and Ting-Yang Kuei (NVIDIA)  (PR #168912), with Inductor ordering support added by Paul Zhang (Meta) (PR #179457). This currently works with the eager and cudagraphs backends; Inductor support is planned for a future release.

**XPU를 위한 FMA 기반 addcdiv 저수준화 / FMA-based addcdiv lowering for XPU**

Inductor가 이제 `addcdiv` 연산에 대해 fused multiply-add(FMA) 명령을 사용하여, Triton 커널 퓨전(fusion) 이점을 유지하면서 eager CUDA 실행과 비트 단위로 동일한 수치 결과를 달성합니다.
> Inductor now uses fused multiply-add (FMA) instructions for addcdiv operations, achieving bitwise numerical parity with eager CUDA execution while preserving Triton kernel fusion benefits.

`addcdiv`는 Adam, AdamW, RMSprop을 비롯한 많은 옵티마이저 업데이트 규칙의 핵심에 있는 fused 산술 연산(`result = input + value × (tensor1 / tensor2)`)입니다. 이전에는 Inductor의 저수준화(lowering)가 별도의 곱셈과 나눗셈 명령을 사용하여, eager 모드와 비교했을 때 작은 부동소수점 반올림 차이가 발생했습니다. 이러한 차이는 수천 번의 학습 단계에 걸쳐 누적되어, 컴파일된 모델이 수치적으로 동일한 결과를 생성하는지 검증하기 어렵게 만들었습니다.
> `addcdiv` is a fused arithmetic operation (`result = input + value × (tensor1 / tensor2)`) that sits at the heart of many optimizer update rules, including Adam, AdamW, and RMSprop. Previously, Inductor's lowering used separate multiply and divide instructions, introducing small floating-point rounding differences compared to eager mode. These differences accumulate over thousands of training steps, making it difficult to validate that compiled models produce numerically identical results.

이는 처음에 Michael Lazos(Meta)가 CUDA에 대해 구현했고(PR #174912), 이후 Guangye Yu(Intel)가 XPU로 확장하여(PR #176163) Intel GPU에서의 여러 수치 정확성 문제를 수정했습니다. 이제 옵티마이저 위주의 학습 루프에서 `torch.compile`을 사용하는 누구나 수치 재현성을 희생하지 않고도 컴파일된 성능을 얻을 수 있으며, 이는 NVIDIA와 Intel 하드웨어 모두에서 적용됩니다.
> This was first implemented for CUDA by Michael Lazos (Meta) (PR #174912), then extended to XPU by Guangye Yu (Intel) (PR #176163), fixing several numerical correctness issues on Intel GPUs. Anyone using `torch.compile` with optimizer-heavy training loops now gets compiled performance without sacrificing numerical reproducibility — on both NVIDIA and Intel hardware.

## 분산 학습 / Distributed Training

**커스텀 연산에서 ProcessGroup 지원 / ProcessGroup support in custom ops**

이제 커스텀 연산자가 ProcessGroup 객체를 직접 인자로 받을 수 있게 되어, 호출자가 이를 문자열 그룹 이름으로 변환하여 전역 레지스트리에서 조회할 필요가 없어졌습니다. 모든 c10d 함수형 집합 통신 연산(`all_reduce`, `reduce_scatter` 등)이 ProcessGroup 객체와 문자열 이름 모두를 받을 수 있도록 업데이트되었습니다. Aaron Orenstein(Meta)이 기여했습니다(PR #172795).
> Custom operators can now accept ProcessGroup objects directly as arguments rather than requiring callers to convert them to string group names and looking them up in a global registry. All c10d functional collective ops (all\_reduce, reduce\_scatter, etc) have been updated to accept both ProcessGroup objects directly and the string names. Contributed by Aaron Orenstein (Meta) (PR #172795).

**다중 GPU/다중 노드 프로파일링 개선 / Multi-GPU/multi-node profiling improvements**

PyTorch Profiler Events API가 이제 flow ID, flow type, activity type, 미완료 이벤트(unfinished events), Python 함수 이벤트를 노출하여, `events()`를 Chrome trace JSON 출력과 동등한 수준으로 만들고 더 풍부한 프로그래밍 방식의 사후 분석을 가능하게 합니다. 또한 새로운 `seq_num` 필드를 사용하여 NCCL 집합 통신 트레이스를 랭크 간에 상관 분석할 수 있게 되었습니다 — 동일한 집합 통신에 참여하는 모든 랭크는 프로세스 그룹 내에서 동일한 시퀀스 번호를 공유합니다. 이러한 변경 사항들은 다중 GPU 및 노드 전반에서 분산 학습 성능을 디버깅하기 위한 도구를 크게 개선합니다. API 확장은 Ryan Zhang(Meta)이 기여했고(PR #177888), NCCL `seq_num`은 Marvin Dsouza(Meta)가 추가했습니다(PR #177148).
> PyTorch Profiler Events API now exposes flow IDs, flow types, activity types, unfinished events, and Python function events — bringing events() to parity with the Chrome trace JSON output and enabling richer programmatic post-hoc analysis. In addition it is now possible to correlate NCCL collective traces across ranks using a new seq\_num field – all ranks participating in the same collective share the same sequence number within a process group. Together these changes significantly improve the tooling for debugging distributed training performance across multiple GPUs and nodes.  API enrichment by Ryan Zhang (Meta) (PR #177888) and NCCL seq\_num added by Marvin Dsouza (Meta) (PR #177148).

**FlightRecorder: ncclx + gloo 백엔드 / FlightRecorder: ncclx + gloo Backends**

FlightRecorder의 트레이스 분석기가 이제 기존 nccl 및 xccl 백엔드와 함께 ncclx 및 gloo 백엔드를 지원하여, 더 광범위한 집합 통신 백엔드에 걸친 분산 통신 추적이 가능해졌습니다. 또한 FlightRecorder는 이제 이전에 추적되지 않았던 torchcomms 연산(예: `all_gather_single`, `reduce_scatter_v`, `barrier`)을 인식합니다. 이번 사이클에서는 여러 프로세스 그룹이 FlightRecorder 싱글톤에 동시 접근할 때 무한 루프를 일으킬 수 있던 경쟁 조건(race condition)도 수정되었습니다. 백엔드 허용 목록(allowlist)은 Lily Janjigian(Meta)이 추가했고(PR #180268), torchcomms 연산 지원은 Tushar Jain이 기여했습니다(PR #178359).
> FlightRecorder's trace analyzer now supports ncclx and gloo backends alongside the existing nccl and xccl backends, enabling distributed communication tracing across a broader set of collective backends. Additionally, FlightRecorder now recognizes torchcomms operations (e.g., all\_gather\_single, reduce\_scatter\_v, barrier) that were previously untracked. A race condition that could cause an infinite loop when multiple process groups concurrently accessed the FlightRecorder singleton was also fixed in this cycle.  Backend allowlist added by Lily Janjigian (Meta) (PR #180268), with  torchcomms operation support by Tushar Jain (PR #178359).

## 플랫폼 관련 업데이트 / Platform Related Updates

### CUDA

**CUDA 그래프 커널 어노테이션 / CUDA Graph kernel annotations**

`torch.cuda.graph`가 이제 `enable_annotations` 키워드 인자를 받아, 캡처된 CUDA 그래프 내의 개별 커널에 어노테이션 메타데이터(예: 집합 통신 연산 이름, 프로세스 그룹, 메시지 크기)를 주입합니다. 동반 후처리 스크립트(`python -m torch.cuda._annotate_cuda_graph_trace`)로 트레이서를 후처리하면, 어노테이션이 트레이스에 병합됩니다. 이러한 어노테이션은 Perfetto/Chrome 프로파일러 트레이스에 표시되어, 재생되는 그래프 내 각 커널이 무엇을 하는지 훨씬 쉽게 파악할 수 있게 해줍니다. Shangdi Yu(Meta)가 기여했습니다(PR #179768).
> `torch.cuda.graph` now accepts an enable\_annotations kwarg that injects annotation metadata (e.g., collective op names, process groups, message sizes) into individual kernels within captured CUDA graphs. After post-processing tracer with a companion post-processing script (python -m torch.cuda.\_annotate\_cuda\_graph\_trace) annotations are merged into traces. These annotations appear in Perfetto/Chrome profiler traces, making it significantly easier to understand what each kernel in a replayed graph is doing. Contributed by Shangdi Yu (Meta) (PR #179768).

**CUDA Green Context 작업 큐 제한 / CUDA Green Context workqueue limit**

CUDA Green Context가 이제 작업 큐(workqueue) 제한 지정을 지원하여, GPU 리소스 분할에 대한 더 세밀한 제어가 가능해졌습니다. 이 실험적(experimental) 기능을 통해 사용자는 green context 내에서 동시 작업 제출의 수를 제한할 수 있어, 동시 워크로드 간에 더 예측 가능한 리소스 공유가 가능합니다. Matthias Jouanneaux(NVIDIA)가 기여했습니다(PR #177242).
> CUDA Green Contexts now support specifying a workqueue limit, giving finer-grained control over GPU resource partitioning. This experimental feature allows users to constrain the number of concurrent work submissions within a green context, enabling more predictable resource sharing across concurrent workloads. Contributed by Matthias Jouanneaux (NVIDIA) (PR #177242).

### ROCm

**ROCm: 확장 가능한 세그먼트 / ROCm: Expandable segments**

AMD GPU(ROCm >= 7.02)가 이제 PyTorch의 캐싱 할당자에서 확장 가능한 메모리 세그먼트(expandable memory segments)를 지원하여, 가상 메모리 API를 통해 동적으로 할당을 늘려 메모리 단편화(fragmentation)를 줄이는 CUDA 기능과 동등한 기능을 제공합니다. Prachi Gupta(AMD)가 추가했습니다(PR #173330).
> AMD GPUs (ROCm >= 7.02) now support expandable memory segments in PyTorch's caching allocator, matching the CUDA feature that reduces memory fragmentation by dynamically growing allocations via virtual memory APIs. Added by Prachi Gupta (AMD) (PR #173330)

**ROCm: rocSHMEM 지원 / ROCm: rocSHMEM support**

rocSHMEM 지원은 AMD GPU에서 대칭 메모리 집합 통신 연산(`torch.ops.symm_mem.*`)을 가능하게 하며, NVSHMEM 기반의 GPU 통신 프리미티브 — 점대점(point-to-point), 브로드캐스트, all-to-all, MoE 지향 2D AllToAllv 포함 — 를 ROCm으로 포팅합니다. rocSHMEM 구현은 NVSHMEM과 rocSHMEM 간의 API 및 워프 크기 차이를 처리하기 위해 전용 컴파일 단위를 사용합니다. Prachi Gupta가 기여했습니다(PR #173518).
> rocSHMEM support enables symmetric memory collective operations (torch.ops.symm\_mem.\*) on AMD GPUs, porting the NVSHMEM-based on-GPU communication primitives — including point-to-point, broadcast, all-to-all, and MoE-oriented 2D AllToAllv — to ROCm. The rocSHMEM implementation uses a dedicated compilation unit to handle API and warp-size differences between NVSHMEM and rocSHMEM. Contributed by Prachi Gupta (PR #173518).

**ROCm: hipSPARSELt 및 FP8 반구조화 희소성 / ROCm: hipSPARSELt and FP8 semi-structured sparsity**

ROCm >= 7.12의 PyTorch 빌드에서 hipSPARSELt가 기본적으로 활성화되어, AMD GPU에 반구조화(2:4) 희소성(semi-structured sparsity) 지원이 추가되었습니다. MI350X(gfx950)에서는 hipSPARSELt를 통해 FP8(`float8_e4m3fn`) 입력도 FP32 출력과 함께 지원됩니다. 이는 이전에 CUDA 전용이었던 `torch._cslt_sparse_mm` 희소성 가속 경로를 동일하게 제공합니다. hipSPARSELt는 rraminen(AMD)이 활성화했고(PR #170852), FP8 반구조화 희소성은 Benji Beck(Meta)이 추가했습니다(PR #179310).
> hipSPARSELt is now enabled by default in PyTorch builds on ROCm >= 7.12, bringing semi-structured (2:4) sparsity support to AMD GPUs. FP8 (float8\_e4m3fn) inputs are also now supported through hipSPARSELt on MI350X (gfx950), with FP32 output. This enables the same torch.\_cslt\_sparse\_mm sparsity acceleration path that was previously CUDA-only. hipSPARSELt enabled by rraminen (AMD) (PR #170852), with FP8 semi-structured sparsity added by Benji Beck (Meta) (PR #179310).

**ROCm: Inductor FlexAttention 파이프라이닝 / ROCm: Inductor FlexAttention pipelining**

AMD GPU의 FlexAttention이 이제 Triton 백엔드에서 2단계 파이프라이닝을 사용하여, MI350X에서 다양한 어텐션 패턴(causal, alibi, sliding window)과 shape에 걸쳐 5~26%의 속도 향상을 제공합니다. 이는 한 줄의 구성 변경(`num_stages=1`에서 `2`로)으로, 더 효율적인 메모리-연산 중첩(overlap)을 가능하게 합니다. nithinsubbiah가 기여했습니다(PR #176676).
> FlexAttention on AMD GPUs now uses two-stage pipelining in the Triton backend, delivering 5-26% speedups across a range of attention patterns (causal, alibi, sliding window) and shapes on MI350X. This was a one-line configuration change (num\_stages=1 to 2) that unlocks more efficient memory-compute overlap. Contributed by nithinsubbiah (PR #176676).

### Apple MPS

**MPS: Metal-4 오프라인 셰이더 컴파일 / MPS: Metal-4 offline shader compilation**

Apple Silicon 바이너리 휠이 이제 macOS 26에서 metal-4 표준으로 사전 컴파일된(ahead-of-time-compiled) Metal-4 셰이더와 함께 제공됩니다. 이는 첫 실행 시 런타임 셰이더 컴파일 오버헤드를 제거하여, MPS 워크로드의 시작 지연 시간을 줄입니다. Isalia20(Irakli Salia)가 기여했습니다(PR #179378).
> Apple Silicon binary wheels now ship with ahead-of-time-compiled Metal-4 shaders, built on macOS 26 with the metal-4 standard. This eliminates the runtime shader compilation overhead on first run, reducing startup latency for MPS workloads. Contributed by Isalia20 (Irakli Salia) (PR #179378).

## 지원 중단 및 호환성 변경 사항 / Deprecations and Breaking Changes

#### 분산: torchcomms를 위한 예정된 호환성 변경 / Distributed: Planned Breaking Changes for torchcomms

PyTorch Distributed에 torchcomms를 직접 통합하여 모두가 그 이점을 기본적으로 누릴 수 있도록 열심히 작업해 왔습니다. 향후 릴리즈(2.13+)에서는 torchcomms를 기본값으로 사용할 계획이며, 여기에는 ProcessGroup 동작 방식에 대한 일부 호환성 변경이 포함됩니다. 이러한 변경 사항이 대부분의 모델에서 자동으로 작동하도록 하고, 생태계의 비호환성을 수정하는 것을 목표로 하고 있지만, 그럼에도 일부 모델은 영향을 받을 수 있습니다.
> We've been working hard on integrating torchcomms directly into PyTorch Distributed so everyone can get the benefits out of the box. In an upcoming  release (2.13+) we're planning on using torchcomms by default, which includes some breaking changes to how ProcessGroups operate. We aim to make these changes work automatically for most models and fix any incompatibilities in the ecosystem, but nevertheless, some models will be impacted.

torchcomms는 아직 다듬는 중이지만, 지금 바로 사용하여 새로운 API, 내결함성(fault tolerance), 윈도우, 확장성, 디버깅 기능에 접근할 수 있습니다. 시작하려면 `pip install torchcomms`를 실행하고 `TORCH_DISTRIBUTED_USE_TORCHCOMMS=1`을 설정하세요.
> We're still polishing torchcomms but you can use it right now and get access to the new APIs, fault tolerance, window, scalability, and debuggability features. To get started, `pip install torchcomms` and set `TORCH_DISTRIBUTED_USE_TORCHCOMMS=1`.

자세한 내용은 [https://github.com/meta-pytorch/torchcomms](https://github.com/meta-pytorch/torchcomms)을 참고하세요.
> See [https://github.com/meta-pytorch/torchcomms](https://github.com/meta-pytorch/torchcomms) for more details.

주요 변경 사항:
> Key changes:

- **Eager 초기화**: 모든 ProcessGroup/communicator가 `dist.init_process_group` 동안 즉시(eagerly) 초기화되어야 하며, 단일 백엔드 디바이스만 지원합니다. 즉, 초기화 시 디바이스를 명시해야 합니다.
- **P2P 연산**: 각 ProcessGroup/communicator가 기반 communicator와 1:1로 일치하도록 만드는 것이 목표입니다. 즉, 동일한 그룹/스트림에서 발행된 P2P 연산은 동시 실행이 보장되지 않습니다. 동시 P2P 연산을 사용하려면 배치 API 또는 별도의 그룹/communicator를 사용해야 합니다.
- **torchcomms 의존성**: torchcomms를 PyTorch Distributed의 필수 패키지로 만들고, 기존 c10d::Backends를 단일하고 더 현대적인 통신 정의(definition)로 대체하기 위해 지원 중단(deprecate)할 계획입니다.

> - Eager Initialization: We will require all ProcessGroup/communicators to be eagerly initialized during dist.init\_process\_group and only support a single backend device. This means that the device will have to be specified during initialization.
> - P2P operations: We aim to make each ProcessGroup/communicator match 1:1 with the underlying communicator. This means that P2P operations issued on the same group/stream will not be guaranteed to run concurrently. Concurrent P2P operations will be required to use the batch APIs or a separate group/communicator.
> - torchcomms dependency: We plan to make torchcomms a required package for PyTorch Distributed and deprecate the existing c10d::Backends in favor of a single, more modern communication definition.

torchcomms 통합은 PyTorch Distributed 팀이 주도하고 있으며, 2.12에서는 Yifan Mao의 백엔드 래퍼 리팩토링(PR #177157)과 Tushar Jain의 FlightRecorder 통합(PR #175270)이 그 기반 작업으로 포함되어 있습니다.
> The torchcomms integration is being led by the PyTorch Distributed team, with groundwork in 2.12, including backend wrapper refactoring by Yifan Mao (PR #177157) and FlightRecorder integration by Tushar Jain (PR #175270).

**TorchScript 지원 중단(Deprecated) / Torchscript is now Deprecated**

TorchScript는 2.10에서 지원 중단(deprecated)되었으며, jit trace 및 script API를 대체하려면 [torch.export](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html)를, 임베디드 런타임을 대체하려면 [ExecuTorch](https://docs.pytorch.org/executorch/stable/index.html)를 사용해야 합니다. 자세한 내용은 PTC에서 진행한 [이 발표](https://youtu.be/X2YbbDmCsOI?si=8s6Ue3BKIa_FYUne&t=903)를 참고하세요.
> Torchscript was deprecated in 2.10 and [torch.export](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html) should be used to replace the jit trace and script APIs, and [Executorch](https://docs.pytorch.org/executorch/stable/index.html) should be used to replace the embedded runtime. For more details, see [this talk](https://youtu.be/X2YbbDmCsOI?si=8s6Ue3BKIa_FYUne&t=903) from PTC.

**CUDA 12.8 휠 지원 중단 / Deprecation of the CUDA 12.8 Wheel**

PyTorch 2.12부터 CUDA 12.8 바이너리 휠은 지원이 중단되며, 표준 릴리즈 매트릭스의 일부로 더 이상 게시되지 않습니다. 기본 휠은 CUDA 13.0으로 유지되며(PyPI에서 `pip install torch`로 설치), CUDA 13.2가 실험적 빌드로 추가되었습니다.
> Starting with PyTorch 2.12, the CUDA 12.8 binary wheel is deprecated and will no longer be published as part of the standard release matrix. The default wheel remains CUDA 13.0 (via `pip install torch` from PyPI), and CUDA 13.2 has been added as an experimental build.

이전 아키텍처(예: Pascal, Volta)에서 실행하는 사용자는 이번 릴리즈에서도 계속 지원되는 CUDA 12.6 휠로 전환해야 합니다. 최신 GPU(예: Blackwell)에서 실행하는 사용자는 CUDA 13.0+ 휠을 사용해야 합니다. 이 경우 NVIDIA 드라이버를 580.65.06(Linux) 또는 580.88(Windows)로 업그레이드해야 합니다.
> Users running on older architectures (e.g., Pascal, Volta) should switch to the CUDA 12.6 wheel, which remains supported in this release. Users running on newer GPUs (e.g., Blackwell) should use the CUDA 13.0+ wheels; note that this requires an NVIDIA driver upgrade to 580.65.06 (Linux) or 580.88 (Windows).

—

***업데이트(2026-05-19)**: [pytorch/pytorch#177276](https://github.com/pytorch/pytorch/pull/177276)이 2.12 릴리즈에 포함되지 않아, 다음 문장이 제거되었습니다:* *동반 API(torch.\_C.\_mps\_loadMetallib)도 추가되어, Triton Apple MPS 백엔드의 컴파일 타임 metallib 워크플로우를 지원하면서, 사전 컴파일된 .metallib 블롭(blob)을 직접 로드할 수 있게 되었습니다.*
> ***Updated (2026-05-19)**: Removed the following sentence, as [pytorch/pytorch#177276](https://github.com/pytorch/pytorch/pull/177276) did not land in the 2.12 release:* *A companion API (torch.\_C.\_mps\_loadMetallib) was also added for loading pre-compiled .metallib blobs directly, supporting the Triton Apple MPS backend's compile-time metallib workflow.*
