---
layout: blog_detail
title: "PyTorch 2.15 출시 공지"
author: PyTorch Foundation
author_type: Organization
ext_author: Junghwan Park (박정환)
image: /assets/blog/2026-10-28-pytorch-2-15-release-blog/hero.png
category: ["pytorch.org", "translation"]
date: 2026-10-28 12:00:00
org_title: "PyTorch 2.15 Release Blog"
org_link: https://pytorch.org/blog/pytorch-2-15-release-blog/
---

![PyTorch 2.15 출시 공지 대표 이미지 / PyTorch 2.15 Release Blog](/assets/blog/2026-10-28-pytorch-2-15-release-blog/hero.png){:style="width:100%"}

PyTorch® 2.15([릴리즈 노트](https://github.com/pytorch/pytorch/releases/tag/v2.15.0))의 출시를 발표하게 되어 기쁩니다!
> We are excited to announce the release of PyTorch® 2.15 ([release notes](https://github.com/pytorch/pytorch/releases/tag/v2.15.0))!

PyTorch 2.15 릴리즈에는 다음과 같은 변경 사항이 포함되어 있습니다:

- **새로 작성한(rewritten) NCCL 백엔드가 기본값이 되며,** 멈춤(hang)이 발생하면 기록을 덤프하는 Flight Recorder를 갖추고, `TORCH_DIST_USE_NCCL2=0` 또는 `nccl-legacy`로 되돌릴 수 있습니다
- **torch.distributed에 단방향(one-sided) 전송(transport)이 추가되어,** NIXL 백엔드와 CUDA 그래프로 캡처할 수 있는 전송을 제공합니다
- **FSDP2와 파이프라인 처리(pipelining)에서 메모리와 통신을 더 세밀하게 제어할 수 있게 되어,** 정밀도에 민감한 매개변수를 fp32로 유지하고, 변화도 누적(gradient accumulation)을 CUDA 그래프로 캡처하며, 변화도 리덕션(gradient reduction)을 계산과 중첩(overlap)할 수 있습니다
- **NVIDIA Blackwell GPU에서 4비트(NVFP4) LLM 디코딩이 더 빨라지며,** 이는 NVGEMM에서 커널 실행(kernel launch) 횟수를 줄이고 실행 사이의 중첩을 개선한 덕분입니다
- **Apple Silicon이 FP8을 지원하고 디코드가 더 빨라지며,** `float8_e4m3fn`과 `_scaled_mm`, 디코드 빠른 경로를 갖춘 새로운 grouped\_mm(MoE 디코드에서 초당 토큰 수 1.32~1.34배), 공개 `torch.mps.Stream` API를 제공합니다
- **더 넓은 플랫폼 지원**: AMD gfx950의 FlyDSL 커널, gfx1250(MI450)의 Tensor Data Mover 지원, ROCm 10.0 휠, Intel GPU에서 더 간단해진 SYCL 커널 작성과 다중 GPU 프로파일링, CUDA 13.4와 함께 제공되는 기본 휠 CUDA 13.2, 그리고 Python 3.15에서의 torch.compile

> The PyTorch 2.15 release features the following changes:
> - **Rewritten NCCL backend becomes the default,** with a flight recorder that dumps on hangs and `TORCH_DIST_USE_NCCL2=0` or `nccl-legacy` as rollback paths
> - **One-sided transports for torch.distributed,** with a NIXL backend and CUDA-graph-capturable transfers
> - **FSDP2 and pipelining gain finer memory and communication control,** with fp32 for precision-sensitive parameters, CUDA-graph-captured gradient accumulation, and gradient reduction overlapped with compute
> - **Faster 4-bit (NVFP4) LLM decoding on NVIDIA Blackwell GPUs** through fewer, better-overlapped kernel launches in NVGEMM
> - **Apple Silicon gains FP8 and faster decode,** with `float8_e4m3fn` and `_scaled_mm`, a new grouped\_mm with a decode fast path (1.32–1.34x tokens/s on MoE decode), and public `torch.mps.Stream` APIs
> - **Broader platform support** with FlyDSL kernels on AMD gfx950, Tensor Data Mover support for gfx1250 (MI450), ROCm 10.0 wheels, simpler SYCL kernel authoring and multi-GPU profiling on Intel GPUs, CUDA 13.2 as the default wheel with CUDA 13.4 alongside it, and torch.compile on Python 3.15

이번 릴리즈는 PyTorch 2.14 이후 583명의 기여자로부터 3,117회의 커밋으로 구성되었습니다. 헌신적인 커뮤니티의 기여에 진심으로 감사드립니다. 언제나 그렇듯, 새로운 버전을 사용해 보시고 문제가 있으면 보고해 주시면 2.15를 개선하는 데 도움이 됩니다. PyTorch 2 시리즈를 시작하는 방법에 대한 자세한 정보는 [시작하기](https://pytorch.org/get-started/locally/) 페이지에서 확인할 수 있습니다.
> This release is composed of 3,117 commits from 583 contributors since PyTorch 2.14. We want to sincerely thank our dedicated community for your contributions. As always, we encourage you to try these out and report any issues as we improve 2.15. More information about how to get started with the PyTorch 2-series can be found at our [Getting Started](https://pytorch.org/get-started/locally/) page.

PyTorch 2.15에 대해 궁금한 점이 있다면 2026년 11월 5일 목요일 오전 11시(태평양 시간)에 열리는 Q&A 웨비나에서 물어봐 주세요. Tristan Rice, Natalia Gimelshein, Huy Do, Jane Xu(모두 Meta)가 2.15 릴리즈를 개괄하고, PyTorch와 이번 릴리즈의 새로운 기능에 대한 커뮤니티의 질문에 답변합니다. 진행은 Chris Gottbrath(Gottbrath Tech)가 맡습니다. [지금 등록하세요.](https://pytorch.org/event/pytorch-2-15-release-live-qa/)
> Bring any questions you may have about PyTorch 2.15 to our Q&A webinar on Thursday, November 5, 2026, at 11 AM PT. Tristan Rice, Natalia Gimelshein, Huy Do, and Jane Xu (all Meta) will share an overview of the 2.15 release and answer community questions about PyTorch and the new capabilities in this release. Chris Gottbrath (Gottbrath Tech) will moderate. [Register today.](https://pytorch.org/event/pytorch-2-15-release-live-qa/)

2.x 시리즈 전반에 걸쳐 PyTorch는 연구 중심의 프레임워크에서, 대규모 프로덕션 학습 및 추론을 위한 통합된 하드웨어에 구애받지 않는(hardware-agnostic) 플랫폼으로 진화해 왔습니다. [PyTorch 2.13](https://pytorch.kr/blog/2026/pytorch-2-13-release-blog/)은 Apple Silicon에 FlexAttention을 도입했고, Inductor에 CuTeDSL 코드 경로를 제공했으며, 대규모 클러스터 학습을 위한 torchcomms를 선보였습니다. [PyTorch 2.14](https://pytorch.kr/blog/2026/pytorch-2-14-release-blog/)는 그 경로를 NVGEMM으로 성숙시켰고, 새로 작성한 NCCL 백엔드를 미리 선보였으며, Apple Silicon에 네이티브 선형대수를 도입했습니다.
> Throughout the 2.x series, PyTorch has been evolving from a research-first framework into a unified, hardware-agnostic platform for production training and inference at scale. [PyTorch 2.13](https://pytorch.org/blog/pytorch-2-13-release-blog/) landed FlexAttention on Apple Silicon, gave Inductor a CuTeDSL code path, and introduced torchcomms for large-cluster training. [PyTorch 2.14](https://pytorch.org/blog/pytorch-2-14-release-blog/) matured that path into NVGEMM, previewed the rewritten NCCL backend, and brought native linear algebra to Apple Silicon.

PyTorch 2.15는 그 작업 위에 쌓아 올린 릴리즈입니다. 이제 새로 작성한 NCCL 백엔드가 모든 `nccl` 프로세스 그룹을 뒷받침하고, NVGEMM은 낮은 정밀도(low-precision) 디코드에 맞춰 튜닝됐으며, Apple Silicon은 FP8과 디코드 전용 커널을 추가하고, AMD 지원은 새로운 컴파일러 DSL과 차세대 Instinct(프리뷰 지원)로 확장되며, Intel GPU는 더 간단한 커널 프로그래밍 모델과 더 깊은 프로파일링 기능을 갖추게 됩니다.
> PyTorch 2.15 builds on that work. The rewritten NCCL backend now backs every `nccl` process group, NVGEMM is tuned for low-precision decode, Apple Silicon adds FP8 and decode-specific kernels, AMD support extends to new compiler DSLs and (preview support) for next Instinct generation, and Intel GPUs gain a simpler kernel programming model and deeper profiling.

## 분산 학습 / Distributed Training

### 새로 작성한 NCCL 백엔드가 기본값으로 / The Rewritten NCCL Backend Becomes the Default

모델을 여러 GPU에 걸쳐 학습할 때 GPU들은 통신 백엔드를 통해 끊임없이 데이터를 주고받으며, NVIDIA 하드웨어에서는 그 백엔드가 NCCL입니다. PyTorch 2.15에서는 `nccl`을 사용하는 모든 작업이 기본적으로 새로 작성한 백엔드에서 실행됩니다.
> When a model trains across many GPUs, those GPUs constantly exchange data through a communication backend, and on NVIDIA hardware that backend is NCCL. In PyTorch 2.15, every job that uses `nccl` runs on our rewritten backend by default.

PyTorch 2.14는 torchcomms에서 이식한, 새로 작성한 NCCL 백엔드를 선택적으로 활성화하는(opt-in) `nccl2` 백엔드로 미리 선보였습니다. 2.15에서는 이 백엔드가 모든 nccl 프로세스 그룹의 기본값이 되며, `TORCH_DIST_USE_NCCL2=0`과 `nccl-legacy` 백엔드 이름으로 이전 백엔드로 되돌릴 수 있습니다. 이제 커뮤니케이터(communicator)는 프로세스 그룹과 함께 즉시(eagerly) 생성되며, 재구성(reconfiguration) 시에는 축소/확장(shrink/grow) 빠른 경로(NCCL 2.29 이상)를 선택적으로 사용해, 살아남은 랭크(rank)가 전체 부트스트랩을 건너뛰게 할 수 있습니다. 내결함성(fault tolerance)을 위한 재구성은 RCCL 2.30.7을 사용하는 ROCm에서도 활성화됩니다.
> PyTorch 2.14 previewed our rewritten NCCL backend, ported from torchcomms, as the opt-in `nccl2` backend. In 2.15 it is the default for every nccl process group, with `TORCH_DIST_USE_NCCL2=0` and the `nccl-legacy` backend name as rollback paths. Communicators are now created eagerly with the process group, and reconfiguration can opt into a shrink/grow fast path (NCCL 2.29 or newer) so surviving ranks skip a full bootstrap. Reconfigure for fault tolerance is also enabled on ROCm with RCCL 2.30.7.

프로덕션 환경에서 이 백엔드를 디버깅할 수 있도록, 백엔드에 구애받지 않는 Flight Recorder가 집합 통신(collective)이 멈추면 사후 분석(post-mortem) 기록을 디스크에 쓰고, 와치독(watchdog) 상태를 디버그 서버에 게시하여 어느 랭크에서든 타임아웃이 발생하면 모든 랭크에서 Flight Recorder 덤프가 실행되도록 합니다. 클러스터 규모에서는 GPU 하나가 멈춰도 전체 학습 실행이 중단될 수 있으며, 그 원인을 찾으려면 대개 장애를 손으로 재현해야 했습니다. Flight Recorder는 멈춘 순간 각 랭크가 무엇을 하고 있었는지를 기록해 둡니다.
> To make the backend debuggable in production, a backend-agnostic flight recorder writes a post-mortem to disk when a collective hangs, and watchdog health is published to the debug server so a timeout on any rank triggers flight-recorder dumps from every rank. At cluster scale, a single stalled GPU can halt an entire training run, and finding the cause has often meant reproducing the failure by hand. The flight recorder captures what each rank was doing at the moment of the hang.

API 안정성: Unstable
> API Unstable

(PR [#192281](https://github.com/pytorch/pytorch/pull/192281), [#192895](https://github.com/pytorch/pytorch/pull/192895), [#199240](https://github.com/pytorch/pytorch/pull/199240), [#198865](https://github.com/pytorch/pytorch/pull/198865), [#198339](https://github.com/pytorch/pytorch/pull/198339) 작성: Tristan Rice, Meta; [#194774](https://github.com/pytorch/pytorch/pull/194774) 작성: Chinmay Kuchinad, AMD; [#192233](https://github.com/pytorch/pytorch/pull/192233) 및 [#199092](https://github.com/pytorch/pytorch/pull/199092) 작성: Tushar Jain, Meta; [#192075](https://github.com/pytorch/pytorch/pull/192075) 작성: Atharva Kshirsagar, Red Hat)
> (PRs [#192281](https://github.com/pytorch/pytorch/pull/192281), [#192895](https://github.com/pytorch/pytorch/pull/192895), [#199240](https://github.com/pytorch/pytorch/pull/199240), [#198865](https://github.com/pytorch/pytorch/pull/198865) and [#198339](https://github.com/pytorch/pytorch/pull/198339) by Tristan Rice, Meta; [#194774](https://github.com/pytorch/pytorch/pull/194774) by Chinmay Kuchinad, AMD; [#192233](https://github.com/pytorch/pytorch/pull/192233) and [#199092](https://github.com/pytorch/pytorch/pull/199092) by Tushar Jain, Meta; [#192075](https://github.com/pytorch/pytorch/pull/192075) by Atharva Kshirsagar, Red Hat)

### NIXL 백엔드를 갖춘 단방향 전송 / One-Sided Transports with a NIXL Backend

torch.distributed에 단방향(one-sided) 데이터 이동을 위한 Python 전송(transport) API가 추가되며, 첫 번째 백엔드는 NIXL입니다. 전송은 프로세스 그룹이나 집합 통신 없이 스토어(store)를 통해 피어(peer)에 연결되고, `Transport.read_stream/write_stream`은 전송을 CUDA 스트림에 넣으므로(enqueue) `torch.cuda.graph`로 캡처할 수 있습니다.
> torch.distributed gains a Python transport API for one-sided data movement, with NIXL as the first backend. Transports connect to a peer through a store, without needing a process group or collectives, and `Transport.read_stream/write_stream` enqueue transfers on a CUDA stream so they can be captured with `torch.cuda.graph`.

API 안정성: Unstable
> API Unstable

(PR [#195643](https://github.com/pytorch/pytorch/pull/195643), [#197032](https://github.com/pytorch/pytorch/pull/197032), [#198889](https://github.com/pytorch/pytorch/pull/198889), [#198918](https://github.com/pytorch/pytorch/pull/198918) 작성: Tristan Rice, Meta)
> (PRs [#195643](https://github.com/pytorch/pytorch/pull/195643), [#197032](https://github.com/pytorch/pytorch/pull/197032), [#198889](https://github.com/pytorch/pytorch/pull/198889) and [#198918](https://github.com/pytorch/pytorch/pull/198918) by Tristan Rice, Meta)

### FSDP2와 파이프라인 병렬화: 메모리 및 통신 제어 / FSDP2 and Pipeline Parallelism: Memory and Communication Control

이제 FSDP2에서 정밀도와 캡처를 더 세밀하게 제어할 수 있습니다. MoE 라우터 게이트와 정규화(norm) 가중치처럼 정밀도에 민감한 매개변수는 fp32로 유지하고 나머지 모델은 bf16으로 학습하면서도, `fully_shard`마다 all-gather는 여전히 한 번만 수행합니다(`MixedPrecisionPolicy(param_dtype_override_fn=...)`). 이제 변화도 누적(gradient accumulation) 스텝 전체를 하나의 CUDA 그래프로 캡처할 수 있습니다. 변화도 정밀도도 매개변수별로 설정할 수 있습니다. 변화도는 `.grad`에 직접 누적되고 각 매개변수의 `grad_dtype`을 따르며, 이로 인해 `reduce_dtype=None`일 때의 기본 동작이 바뀝니다(지원 중단 및 하위 호환성 변경 사항 참고).
> FSDP2 now gives finer control over precision and capture. Precision-sensitive parameters, such as MoE router gates and norm weights, can stay in fp32 while the rest of the model trains in bf16, still with one all-gather per `fully_shard` (`MixedPrecisionPolicy(param_dtype_override_fn=...)`). A full gradient-accumulation step can now be captured in a single CUDA graph. Gradient precision can also be set per parameter. Gradients accumulate directly in `.grad` and follow each parameter's `grad_dtype`, which changes the default behavior when `reduce_dtype=None` (see Deprecations and Backwards-Incompatible Changes).

파이프라인 병렬화(pipeline parallelism)는 이제 모든 마이크로배치(microbatch)의 수신 버퍼를 미리 할당하지 않고 각 수신 직전에 할당하며, 순방향(forward) 출력은 송신이 끝나는 대로 해제합니다. 스케줄은 `max_outstanding_sends` 예산을 받으며, FSDP 언샤드(unshard)는 스텝 시작 시점에 한꺼번에 발행되지 않고 각 랭크의 시작 버블(startup bubble)에 걸쳐 시차를 두고 발행됩니다. 스케줄이 결정하는 파이프라인 병렬 랭크 사이의 고유한 연결(edge)마다 이제 전용 P2P 커뮤니케이터가 할당됩니다. FSDP 변화도 리덕션(reduce grads)도 이제 뒤따르는 계산과 중첩할 수 있습니다. 이러한 개선으로 PyTorch 파이프라인 처리는 CUDA 그래프로 캡처할 수 있고(cudagraphable), 결정적(deterministic)이며, 최적화된 메모리 프로파일을 갖추게 되고, 드러나는 통신(exposed communication)을 최소화하기 위한 성능 튜닝 옵션(knob)을 제공합니다.
> Pipeline parallelism now allocates receive buffers just before each receive instead of for every microbatch up front, and releases forward outputs once their sends complete. Schedules accept a `max_outstanding_sends` budget, and FSDP unshards are staggered across each rank's startup bubble instead of all being issued at the start of a step. Each unique edge between pipeline-parallel ranks, as determined by the schedule, now gets a dedicated P2P communicator. FSDP reduce grads can now be overlapped with subsequent compute. These improvements make PyTorch pipelining cudagraphable, deterministic, with an optimized memory profile, and expose performance tuning knobs for minimizing exposed communication.

API 안정성: Unstable
> API Unstable

(PR [#196640](https://github.com/pytorch/pytorch/pull/196640), [#196724](https://github.com/pytorch/pytorch/pull/196724), [#196727](https://github.com/pytorch/pytorch/pull/196727), [#196641](https://github.com/pytorch/pytorch/pull/196641), [#196963](https://github.com/pytorch/pytorch/pull/196963) 작성: Chien-Chin Huang, Meta; [#196626](https://github.com/pytorch/pytorch/pull/196626) 작성: Anshul Sinha, Meta; [#198417](https://github.com/pytorch/pytorch/pull/198417) 작성: Andrew Or, Apache Spark PMC; [#194114](https://github.com/pytorch/pytorch/pull/194114) 작성: Animesh Jain, Meta; [#196463](https://github.com/pytorch/pytorch/pull/196463), [#196464](https://github.com/pytorch/pytorch/pull/196464), [#196706](https://github.com/pytorch/pytorch/pull/196706) 작성: Sanket Purandare, Meta)
> (PRs [#196640](https://github.com/pytorch/pytorch/pull/196640), [#196724](https://github.com/pytorch/pytorch/pull/196724), [#196727](https://github.com/pytorch/pytorch/pull/196727), [#196641](https://github.com/pytorch/pytorch/pull/196641) and [#196963](https://github.com/pytorch/pytorch/pull/196963) by Chien-Chin Huang, Meta; [#196626](https://github.com/pytorch/pytorch/pull/196626) by Anshul Sinha, Meta; [#198417](https://github.com/pytorch/pytorch/pull/198417) by Andrew Or, Apache Spark PMC); [#194114](https://github.com/pytorch/pytorch/pull/194114) by Animesh Jain, Meta; [#196463](https://github.com/pytorch/pytorch/pull/196463), [#196464](https://github.com/pytorch/pytorch/pull/196464) and [#196706](https://github.com/pytorch/pytorch/pull/196706) by Sanket Purandare, Meta)

## 성능 개선 / Performance Improvements

### 디코드와 선형대수를 위한 더 빠른 MPS 커널 / Faster MPS Kernels for Decode and Linear Algebra

2.14에서 진행한 이전(migration)에 이어, 더 많은 MPS 연산자가 MPSGraph에서 네이티브 Metal 커널로 옮겨 갑니다. 128,256개 항목으로 이루어진 어휘(vocabulary)에 대한 단일 행 softmax의 소요 시간이 8배 줄어들고, upsample 역전파(backward)는 최대 21.7배 빨라지며, 리덕션이 이제 하나의 커널 집합을 공유하게 되어 `norm` 전체 리덕션이 100배 넘게 빨라졌습니다.
> Building on the 2.14 migration, more MPS operators move from MPSGraph to native Metal kernels. A single-row softmax over a 128,256-entry vocabulary drops by 8x, upsample backward is up to 21.7x faster, and reductions now share one set of kernels, making `norm` full reductions more than 100x faster.

디코드의 경우 `F.linear`가 이제 기존 gemv 커널을 사용하며(기하 평균 1.23배 속도 향상, 작은 프로젝션에서는 최대 5배), 새로운 MPS `grouped_mm`에는 gemv 경로가 있어 M3 Ultra에서 MoE 디코드 처리량이 1.32~1.34배 높아집니다. 몇몇 선형대수 솔버도 다시 작성했습니다. 512 × 512 행렬의 QR 분해는 약 7초에서 1.5ms로 줄었고, 삼각 행렬 풀이(triangular solve)는 이제 블록 단위로 처리됩니다.
> For decode, `F.linear` now uses the existing gemv kernels (1.23x geometric-mean speedup, up to 5x on small projections), and the new MPS `grouped_mm` has a gemv path that gives 1.32–1.34x higher MoE decode throughput on an M3 Ultra. Several linear algebra solvers were also rewritten: QR of a 512 x 512 matrix falls from about 7 seconds to 1.5 ms, and triangular solves are now blocked.

(PR [#196071](https://github.com/pytorch/pytorch/pull/196071), [#199687](https://github.com/pytorch/pytorch/pull/199687), [#198611](https://github.com/pytorch/pytorch/pull/198611), [#198495](https://github.com/pytorch/pytorch/pull/198495), [#194933](https://github.com/pytorch/pytorch/pull/194933), [#198721](https://github.com/pytorch/pytorch/pull/198721), [#198725](https://github.com/pytorch/pytorch/pull/198725), [#198732](https://github.com/pytorch/pytorch/pull/198732), [#195944](https://github.com/pytorch/pytorch/pull/195944), [#196900](https://github.com/pytorch/pytorch/pull/196900) 작성: Irakli Salia, Hugging Face; [#196922](https://github.com/pytorch/pytorch/pull/196922), [#196241](https://github.com/pytorch/pytorch/pull/196241), [#195369](https://github.com/pytorch/pytorch/pull/195369) 작성: Nikita Shulga, Thinking Machines Lab; [#193153](https://github.com/pytorch/pytorch/pull/193153) 작성: Sunghyun Choi, 한국공학대학교(Tech University of Korea); [#188473](https://github.com/pytorch/pytorch/pull/188473) 작성: Shrivas Shankar, University of Wisconsin-Madison)
> (PRs [#196071](https://github.com/pytorch/pytorch/pull/196071), [#199687](https://github.com/pytorch/pytorch/pull/199687), [#198611](https://github.com/pytorch/pytorch/pull/198611), [#198495](https://github.com/pytorch/pytorch/pull/198495), [#194933](https://github.com/pytorch/pytorch/pull/194933), [#198721](https://github.com/pytorch/pytorch/pull/198721), [#198725](https://github.com/pytorch/pytorch/pull/198725), [#198732](https://github.com/pytorch/pytorch/pull/198732), [#195944](https://github.com/pytorch/pytorch/pull/195944) and [#196900](https://github.com/pytorch/pytorch/pull/196900) by Irakli Salia, Hugging Face; [#196922](https://github.com/pytorch/pytorch/pull/196922), [#196241](https://github.com/pytorch/pytorch/pull/196241) and [#195369](https://github.com/pytorch/pytorch/pull/195369) by Nikita Shulga, Thinking Machines Lab; [#193153](https://github.com/pytorch/pytorch/pull/193153) by Sunghyun Choi, Tech University of Korea; [#188473](https://github.com/pytorch/pytorch/pull/188473) by Shrivas Shankar, University of Wisconsin-Madison)

## 핵심 기능 / Core Features

### 기본 CUDA 13.2와 CUDA 13.4 휠 / CUDA 13.2 by Default and CUDA 13.4 Wheels

기본 PyTorch 휠이 CUDA 13.2(13.2.2 툴킷)로 바뀌며, CUDA 13.4 GA 휠이 Linux x86\_64, Linux aarch64, Windows용으로 함께 제공됩니다. CUDA 13.4 휠은 NVIDIA Rubin GPU에 대한 실험적 지원을 추가합니다.
> The default PyTorch wheel moves to CUDA 13.2 (13.2.2 toolkit), with CUDA 13.4 GA wheels shipping alongside it for Linux x86\_64, Linux aarch64 and Windows. The CUDA 13.4 wheels add experimental support for NVIDIA Rubin GPUs.

(PR [#198913](https://github.com/pytorch/pytorch/pull/198913) 및 [#196014](https://github.com/pytorch/pytorch/pull/196014) 작성: Andrey Talman, Meta; [#194996](https://github.com/pytorch/pytorch/pull/194996) 및 [#198843](https://github.com/pytorch/pytorch/pull/198843) 작성: Ashish Karale, NVIDIA; [#196560](https://github.com/pytorch/pytorch/pull/196560), [#199269](https://github.com/pytorch/pytorch/pull/199269), [#196351](https://github.com/pytorch/pytorch/pull/196351) 작성: Ting Lu, NVIDIA; [#192653](https://github.com/pytorch/pytorch/pull/192653) 작성: Wei Wang, NVIDIA; [#186245](https://github.com/pytorch/pytorch/pull/186245) 작성: Lakshay Garg, NVIDIA)
> (PRs [#198913](https://github.com/pytorch/pytorch/pull/198913) and [#196014](https://github.com/pytorch/pytorch/pull/196014) by Andrey Talman, Meta; [#194996](https://github.com/pytorch/pytorch/pull/194996) and [#198843](https://github.com/pytorch/pytorch/pull/198843) by Ashish Karale, NVIDIA; [#196560](https://github.com/pytorch/pytorch/pull/196560), [#199269](https://github.com/pytorch/pytorch/pull/199269) and [#196351](https://github.com/pytorch/pytorch/pull/196351) by Ting Lu, NVIDIA; [#192653](https://github.com/pytorch/pytorch/pull/192653) by Wei Wang, NVIDIA; [#186245](https://github.com/pytorch/pytorch/pull/186245) by Lakshay Garg, NVIDIA)

### ROCm 10.0과 TheRock 휠 / ROCm 10.0 and TheRock Wheels

PyTorch 2.15는 2.14에서 도입한 ROCm 7.14 휠에 더해 ROCm 10.0 휠을 추가합니다. 이제 모든 ROCm 휠은 AMD의 TheRock 패키지를 기반으로 빌드되며, ROCm 라이브러리를 번들로 포함하는 대신 ROCm을 pip 의존성(대응하는 rocm 패키지)으로 설치합니다.
> PyTorch 2.15 adds ROCm 10.0 wheels alongside the ROCm 7.14 wheels introduced in 2.14. All ROCm wheels are now built on AMD's TheRock packages, and they install ROCm as a pip dependency (the matching rocm package) instead of bundling the ROCm libraries.

(PR [#194919](https://github.com/pytorch/pytorch/pull/194919), [#198895](https://github.com/pytorch/pytorch/pull/198895), [#196610](https://github.com/pytorch/pytorch/pull/196610) 작성: Ethan Wee, AMD; [#195672](https://github.com/pytorch/pytorch/pull/195672) 작성: Andrey Talman, Meta)
> (PRs [#194919](https://github.com/pytorch/pytorch/pull/194919), [#198895](https://github.com/pytorch/pytorch/pull/198895) and [#196610](https://github.com/pytorch/pytorch/pull/196610) by Ethan Wee, AMD; [#195672](https://github.com/pytorch/pytorch/pull/195672) by Andrey Talman, Meta)

### Python 3.11~3.15 지원과 Python 3.15에서의 torch.compile / Python 3.11 to 3.15, and torch.compile on Python 3.15

PyTorch 2.15는 프리스레드(free-threaded) 3.14t 및 3.15t 빌드를 포함해 Python 3.11부터 3.15까지 지원합니다. 2.15에서 Dynamo가 Python 3.15 지원을 추가합니다.
> PyTorch 2.15 supports Python 3.11 through 3.15, including the free-threaded 3.14t and 3.15t builds. In 2.15, Dynamo adds Python 3.15 support.

(PR [#178393](https://github.com/pytorch/pytorch/pull/178393), [#194117](https://github.com/pytorch/pytorch/pull/194117), [#198018](https://github.com/pytorch/pytorch/pull/198018) 작성: Rob Timpe, OpenTeams; [#198233](https://github.com/pytorch/pytorch/pull/198233), [#199398](https://github.com/pytorch/pytorch/pull/199398), [#199356](https://github.com/pytorch/pytorch/pull/199356) 작성: Andrey Talman, Meta)
> (PRs [#178393](https://github.com/pytorch/pytorch/pull/178393), [#194117](https://github.com/pytorch/pytorch/pull/194117) and [#198018](https://github.com/pytorch/pytorch/pull/198018) by Rob Timpe, OpenTeams; [#198233](https://github.com/pytorch/pytorch/pull/198233), [#199398](https://github.com/pytorch/pytorch/pull/199398) and [#199356](https://github.com/pytorch/pytorch/pull/199356) by Andrey Talman, Meta)

## 플랫폼 기능 및 업데이트 / Platform Features and Updates

### CUDA

#### NVFP4 디코드를 위한 NVGEMM / NVGEMM for NVFP4 Decode

2.14에서 추가된 CuTeDSL GEMM 백엔드인 NVGEMM은 이제 ModelOpt NVFP4 선형 계층의 스칼라 역양자화(dequantization) 계수를 GEMM 안으로 접어 넣어, 디코드 중 계층마다 따로 실행되던 점별(pointwise) 커널 실행을 없앱니다. nvMatmulHeuristics가 놓치는 Blackwell NVFP4 디코드 shape은 범위가 제한된(bounded), shape을 고려하는(shape-aware) 후보 정책이 다루며, NVGEMM 커널은 프로그래밍 방식 종속 실행(programmatic dependent launch)을 사용할 수 있어 뒤따르는 Triton 소비자(consumer) 커널이 생산자(producer) GEMM의 끝부분과 중첩됩니다. NVGEMM은 또한 CUTLASS Operators 위로 옮겨 갑니다.
> NVGEMM, the CuTeDSL GEMM backend added in 2.14, now folds the scalar dequantization factor of ModelOpt NVFP4 linear layers into the GEMM, removing a separate pointwise launch per layer during decode. A bounded, shape-aware candidate policy covers Blackwell NVFP4 decode shapes that nvMatmulHeuristics misses, and NVGEMM kernels can use programmatic dependent launch so a Triton consumer overlaps with the tail of the producer GEMM. NVGEMM also moves onto CUTLASS Operators.

API 안정성: Unstable
> API Unstable

(PR [#194651](https://github.com/pytorch/pytorch/pull/194651), [#194652](https://github.com/pytorch/pytorch/pull/194652), [#197485](https://github.com/pytorch/pytorch/pull/197485), [#194792](https://github.com/pytorch/pytorch/pull/194792), [#194793](https://github.com/pytorch/pytorch/pull/194793) 작성: Michael Lazos, Meta; [#194364](https://github.com/pytorch/pytorch/pull/194364) 작성: Masaki Kozuki, NVIDIA)
> (PRs [#194651](https://github.com/pytorch/pytorch/pull/194651), [#194652](https://github.com/pytorch/pytorch/pull/194652), [#197485](https://github.com/pytorch/pytorch/pull/197485), [#194792](https://github.com/pytorch/pytorch/pull/194792) and [#194793](https://github.com/pytorch/pytorch/pull/194793) by Michael Lazos, Meta; [#194364](https://github.com/pytorch/pytorch/pull/194364) by Masaki Kozuki, NVIDIA)

### ROCm

#### gfx950의 FlyDSL 커널과 gfx1250의 Tensor Data Mover / FlyDSL Kernels on gfx950 and Tensor Data Mover on gfx1250

FlyDSL이 CuTeDSL, Triton, Helion에 이어 PyTorch가 사용할 수 있는 커널 DSL로 합류합니다. gfx950에서 Inductor는 GEMM, 비정형(ragged) `F.grouped_mm`, MXFP4/MXFP8 `scaled_mm`에 대해 FlyDSL 템플릿을 Triton, ATen과 나란히 오토튜닝하며, 즉시 실행(eager) 모드에서는 FlyDSL이 RMSNorm과 topk 구현을 대체(override)할 수 있습니다. FlyDSL은 선택 사항이며 FlyDSL 0.3.x가 필요합니다.
> FlyDSL joins CuTeDSL, Triton and Helion as a kernel DSL that PyTorch can use. On gfx950, Inductor autotunes FlyDSL templates for GEMM, ragged `F.grouped_mm` and MXFP4/MXFP8 `scaled_mm` alongside Triton and ATen, and in eager mode FlyDSL can override RMSNorm and topk. FlyDSL is optional and requires FlyDSL 0.3.x.

gfx1250(MI450)에서 Inductor는 밀집(dense) mm/addmm, 점별 연산과 리덕션의 코드 생성, 그리고 FlexAttention 순방향과 디코드에 Tensor Data Mover를 사용합니다. gfx1250은 나이틀리(nightly) 휠 아키텍처 목록에 추가됩니다.
> On gfx1250 (MI450), Inductor uses the Tensor Data Mover for dense mm/addmm, pointwise and reduction codegen, and FlexAttention forward and decode. gfx1250 is added to the nightly wheel architecture list.

API 안정성: Unstable
> API Unstable

(PR [#192877](https://github.com/pytorch/pytorch/pull/192877), [#190903](https://github.com/pytorch/pytorch/pull/190903), [#194032](https://github.com/pytorch/pytorch/pull/194032), [#191446](https://github.com/pytorch/pytorch/pull/191446), [#191447](https://github.com/pytorch/pytorch/pull/191447) 작성: Xiaobing Zhang, AMD; [#194981](https://github.com/pytorch/pytorch/pull/194981) 작성: Yutao Xu, AMD; [#196719](https://github.com/pytorch/pytorch/pull/196719) 작성: Heng Yi, AMD; [#191166](https://github.com/pytorch/pytorch/pull/191166), [#196525](https://github.com/pytorch/pytorch/pull/196525), [#197329](https://github.com/pytorch/pytorch/pull/197329) 작성: Glen Cao, AMD; [#196610](https://github.com/pytorch/pytorch/pull/196610) 작성: Ethan Wee, AMD)
> (PRs [#192877](https://github.com/pytorch/pytorch/pull/192877), [#190903](https://github.com/pytorch/pytorch/pull/190903), [#194032](https://github.com/pytorch/pytorch/pull/194032), [#191446](https://github.com/pytorch/pytorch/pull/191446) and [#191447](https://github.com/pytorch/pytorch/pull/191447) by Xiaobing Zhang, AMD; [#194981](https://github.com/pytorch/pytorch/pull/194981) by Yutao Xu, AMD; [#196719](https://github.com/pytorch/pytorch/pull/196719) by Heng Yi, AMD; [#191166](https://github.com/pytorch/pytorch/pull/191166), [#196525](https://github.com/pytorch/pytorch/pull/196525) and [#197329](https://github.com/pytorch/pytorch/pull/197329) by Glen Cao, AMD; [#196610](https://github.com/pytorch/pytorch/pull/196610) by Ethan Wee, AMD)

### MPS (Apple Silicon)

#### Apple Silicon의 FP8, 스트림, 더 빠른 MoE 추론 / FP8, Streams and faster MoE inference on Apple Silicon

이제 MPS에서 `torch.float8_e4m3fn`을 지원하며, 텐서 단위(tensorwise) 또는 행 단위(rowwise) 스케일링을 사용하는 `_scaled_mm`과 디코드 크기의 shape을 위한 gemv 빠른 경로도 함께 지원합니다. MPS에는 CUDA의 스트림 API를 본뜬 공개 스트림 API도 추가됩니다: `torch.mps.Stream`, `current_stream`, `set_stream`, `torch.mps.stream` 컨텍스트 매니저, `Tensor.record_stream`.
> `torch.float8_e4m3fn` is now supported on MPS, along with `_scaled_mm` with tensorwise or rowwise scaling and a gemv fast path for decode-sized shapes. MPS also gains a public stream API modeled on CUDA's: `torch.mps.Stream`, `current_stream`, `set_stream`, the `torch.mps.stream` context manager, and `Tensor.record_stream`.

API 안정성: Unstable
> API Unstable

(PR [#194074](https://github.com/pytorch/pytorch/pull/194074), [#196916](https://github.com/pytorch/pytorch/pull/196916), [#197862](https://github.com/pytorch/pytorch/pull/197862), [#196945](https://github.com/pytorch/pytorch/pull/196945), [#196918](https://github.com/pytorch/pytorch/pull/196918), [#196955](https://github.com/pytorch/pytorch/pull/196955) 작성: Irakli Salia, Hugging Face; [#191415](https://github.com/pytorch/pytorch/pull/191415), [#193464](https://github.com/pytorch/pytorch/pull/193464), [#195867](https://github.com/pytorch/pytorch/pull/195867) 작성: Kurt Mohler, OpenTeams; [#196395](https://github.com/pytorch/pytorch/pull/196395) 작성: Aditya Venkataraman, Meta)
> (PRs [#194074](https://github.com/pytorch/pytorch/pull/194074), [#196916](https://github.com/pytorch/pytorch/pull/196916), [#197862](https://github.com/pytorch/pytorch/pull/197862), [#196945](https://github.com/pytorch/pytorch/pull/196945), [#196918](https://github.com/pytorch/pytorch/pull/196918) and [#196955](https://github.com/pytorch/pytorch/pull/196955) by Irakli Salia, Hugging Face; [#191415](https://github.com/pytorch/pytorch/pull/191415), [#193464](https://github.com/pytorch/pytorch/pull/193464) and [#195867](https://github.com/pytorch/pytorch/pull/195867) by Kurt Mohler, OpenTeams; [#196395](https://github.com/pytorch/pytorch/pull/196395) by Aditya Venkataraman, Meta)

### XPU (Intel GPU) / XPU (Intel GPUs)

#### 더 간단해진 SYCL 커널 프로그래밍 / SYCL Kernel Programming Simplified

이번 릴리즈는 XPU 백엔드에 SYCL 자유 함수(free-function) 커널 지원을 도입하며, 기존 SYCL 커널 대부분이 이미 새 프로그래밍 모델로 이전됐습니다. 장황한 펑터(functor) 기반 구현을 명시적인 자유 함수 커널로 대체함으로써, 개발자는 더 직관적인 방식으로 디바이스 코드를 작성할 수 있고, Intel XPU용 커스텀 GPU 커널을 개발하고 이식하는 진입 장벽도 낮아집니다.
> This release introduces SYCL free-function kernel support for the XPU backend, with most existing SYCL kernels already migrated to the new programming model. By replacing verbose functor-based implementations with explicit free-function kernels, developers can write device code in a more intuitive way, lowering the barrier to developing and porting custom GPU kernels for Intel XPUs.

(PR [#196387](https://github.com/pytorch/pytorch/pull/196387), [#196932](https://github.com/pytorch/pytorch/pull/196932), [#198200](https://github.com/pytorch/pytorch/pull/198200), [#198720](https://github.com/pytorch/pytorch/pull/198720), [#200223](https://github.com/pytorch/pytorch/pull/200223) 작성: Yifeng Cui, Intel)
> (PRs [#196387](https://github.com/pytorch/pytorch/pull/196387), [#196932](https://github.com/pytorch/pytorch/pull/196932), [#198200](https://github.com/pytorch/pytorch/pull/198200), [#198720](https://github.com/pytorch/pytorch/pull/198720), and [#200223](https://github.com/pytorch/pytorch/pull/200223) by Yifeng Cui, Intel)

#### 향상된 XPU 런타임 기능 / Enhanced XPU Runtime Capabilities

이번 릴리즈는 torch.xpu.Event에 IPC 지원을 추가해 XPU 런타임 기능을 확장하며, 이를 통해 프로세스 간에 이벤트를 공유하고 torch.multiprocessing과 통합할 수 있습니다. 또한 torch.xpu.MemPool이 이제 use\_on\_oom과 non\_split 할당 정책을 지원하여, 메모리 부족(out-of-memory) 상황을 처리하거나 분산 대칭 메모리(symmetric-memory) 작업 흐름을 활용해야 하는 애플리케이션에 더 큰 유연성을 제공합니다.
> This release expands XPU runtime capabilities with IPC support for torch.xpu.Event, enabling event sharing across processes and integration with torch.multiprocessing. In addition, torch.xpu.MemPool now supports use\_on\_oom and non\_split allocation policies, providing greater flexibility for applications that need to handle out-of-memory conditions or leverage distributed symmetric-memory workflows.

API 안정성: Unstable
> API Unstable

(PR [#191182](https://github.com/pytorch/pytorch/pull/191182), [#192495](https://github.com/pytorch/pytorch/pull/192495), [#192514](https://github.com/pytorch/pytorch/pull/192514), [#192038](https://github.com/pytorch/pytorch/pull/192038), [#192033](https://github.com/pytorch/pytorch/pull/192033) 작성: Guangye Yu, Intel)
> (PRs [#191182](https://github.com/pytorch/pytorch/pull/191182), [#192495](https://github.com/pytorch/pytorch/pull/192495), [#192514](https://github.com/pytorch/pytorch/pull/192514), [#192038](https://github.com/pytorch/pytorch/pull/192038) and [#192033](https://github.com/pytorch/pytorch/pull/192033) by Guangye Yu, Intel)

#### 개선된 XPU 프로파일링 경험 / Improved XPU Profiling Experience

이번 릴리즈는 XPU Graph 재생(replay)에 대한 커널 수준 가시성, XCCL 집합 통신에 대한 랭크 간 상관관계(cross-rank correlation) 분석, 다중 GPU 워크로드에서 모든 Intel GPU에 걸친 커널별 하드웨어 카운터 수집을 추가합니다. 이러한 개선으로 그래프 기반, 다중 GPU, 분산 학습 및 추론에 대해 더 깊은 성능 분석을 할 수 있습니다.
> This release adds kernel-level visibility into XPU Graph replay, cross-rank correlation for XCCL collectives, and per-kernel hardware counter collection across all Intel GPUs in multi-GPU workloads. These enhancements enable deeper performance analysis for graph-based, multi-GPU, and distributed training and inference.

(PR [#195128](https://github.com/pytorch/pytorch/pull/195128/) 작성: Chuanqi Wang, Intel; PR [#5486](https://github.com/intel/torch-xpu-ops/pull/5486) 및 [#1479](https://github.com/pytorch/kineto/pull/1479) 작성: Adrian Ostrowski, Intel)
> (PR [#195128](https://github.com/pytorch/pytorch/pull/195128/) by Chuanqi Wang, Intel; PR [#5486](https://github.com/intel/torch-xpu-ops/pull/5486) and [#1479](https://github.com/pytorch/kineto/pull/1479) by Adrian Ostrowski, Intel)

### C++ ABI

#### 확장된 안정 ABI와 헤더 전용 영역 / Expanded Stable ABI and Header-Only Surface

`torch::stable::sort`가 안정(stable) ABI에 합류하고, Python 상호 운용(interop) 심(shim)에는 기존 텐서 변환에 더해 dtype/device 변환과 `is_tensor` 검사 함수(probe)가 추가됩니다. `c10/util`의 더 많은 부분(irange, load, 수학 상수, 타입 캐스팅, 오버플로 검사)이 `torch/headeronly`로 옮겨 가며, 기존 경로는 새 위치를 가리키는 전달용 include(forwarding include)로 유지됩니다. 이제 확장(extension)은 `TORCH_TARGET_VERSION`이 자신의 libtorch 헤더보다 새로우면 컴파일 시점 오류를 받으며, `load_inline`은 프리스레드 Python에서 안전하게 import할 수 있는 pybind11 모듈을 생성할 수 있습니다.
> `torch::stable::sort` joins the stable ABI, the Python interop shims gain dtype/device conversions and an `is_tensor` probe alongside the existing tensor conversions. More of `c10/util` (irange, load, math constants, type casting and overflow checks) moves into `torch/headeronly`, with the old paths kept as forwarding includes. Extensions now get a compile-time error if `TORCH_TARGET_VERSION` is newer than their libtorch headers, and `load_inline` can generate pybind11 modules that are safe to import on free-threaded Python.

(PR [#198038](https://github.com/pytorch/pytorch/pull/198038) 작성: Adrian Abeyta, Red Hat; [#194673](https://github.com/pytorch/pytorch/pull/194673), [#194781](https://github.com/pytorch/pytorch/pull/194781), [#193962](https://github.com/pytorch/pytorch/pull/193962) 작성: Chris Leonard, Red Hat; [#192774](https://github.com/pytorch/pytorch/pull/192774) 작성: Paweł Gadziński, NVIDIA; [#196802](https://github.com/pytorch/pytorch/pull/196802) 작성: Alex Malyshev, Meta)
> (PRs [#198038](https://github.com/pytorch/pytorch/pull/198038) by Adrian Abeyta, Red Hat; [#194673](https://github.com/pytorch/pytorch/pull/194673), [#194781](https://github.com/pytorch/pytorch/pull/194781) and [#193962](https://github.com/pytorch/pytorch/pull/193962) by Chris Leonard, Red Hat; [#192774](https://github.com/pytorch/pytorch/pull/192774) by Paweł Gadziński, NVIDIA; [#196802](https://github.com/pytorch/pytorch/pull/196802) by Alex Malyshev, Meta)

## 프로파일링 및 디버깅 / Profiling and Debugging

### Cuspy를 이용한 CUDA 그래프 커널 귀속 / CUDA Graph Kernel Attribution with Cuspy

이제 CUDA 그래프에서 재생된 커널을, 그 커널을 캡처한 코드로 거슬러 연결(attribute)할 수 있습니다. 커널 주석(annotation)은 CUPTI 노드 생성 콜백에서 기록되어 `export_chrome_trace`가 트레이스에 인라인으로 써 넣으며, `record_py_stacks`를 사용하면 CUDA 그래프가 각 노드의 Python 실행(launch) 스택을 캡처합니다. CUPTI 모니터의 이름은 Cuspy로 바뀌었으며 `CuspyConfig`로 활성화합니다.
> Kernels replayed from a CUDA graph can now be attributed back to the code that captured them. Kernel annotations are recorded from CUPTI node-creation callbacks and written inline by `export_chrome_trace`, and with `record_py_stacks` CUDA graphs capture the Python launch stack for each node. The CUPTI monitor is renamed Cuspy and is enabled with a `CuspyConfig`.

API 안정성: Unstable
> API Unstable

(PR [#191999](https://github.com/pytorch/pytorch/pull/191999), [#194854](https://github.com/pytorch/pytorch/pull/194854), [#196528](https://github.com/pytorch/pytorch/pull/196528), [#198419](https://github.com/pytorch/pytorch/pull/198419), [#196529](https://github.com/pytorch/pytorch/pull/196529) 작성: Natalia Gimelshein, Meta; [#196066](https://github.com/pytorch/pytorch/pull/196066) 작성: Ryan Zhang, Meta)
> (PRs [#191999](https://github.com/pytorch/pytorch/pull/191999), [#194854](https://github.com/pytorch/pytorch/pull/194854), [#196528](https://github.com/pytorch/pytorch/pull/196528), [#198419](https://github.com/pytorch/pytorch/pull/198419) and [#196529](https://github.com/pytorch/pytorch/pull/196529) by Natalia Gimelshein, Meta; [#196066](https://github.com/pytorch/pytorch/pull/196066) by Ryan Zhang, Meta)

## 지원 중단 및 하위 호환성 변경 사항 / Deprecations and Backwards-Incompatible Changes

- **Python 3.10은 더 이상 지원하지 않습니다.** PyTorch 2.15는 Python 3.11 이상이 필요합니다. 3.10 휠은 더 이상 빌드하지 않으며, CI에서도 3.10을 더 이상 테스트하지 않습니다. 3.10에서 소스로 빌드하는 것도 지원하지 않으며, 다음 릴리즈들에서 torch.compile(Dynamo)을 포함해 남아 있는 3.10 코드 경로를 제거할 예정입니다. [#198233](https://github.com/pytorch/pytorch/pull/198233)과 [#199417](https://github.com/pytorch/pytorch/pull/199417)을 참고하세요.
- **이제 nccl은 새로 작성한 NCCL 백엔드를 선택합니다.** 레거시 `ProcessGroupNCCL`로 되돌리려면 `TORCH_DIST_USE_NCCL2=0`을 설정하거나 `nccl-legacy` 백엔드를 사용하세요. [#192281](https://github.com/pytorch/pytorch/pull/192281)을 참고하세요.
- **torch.ops 및 커스텀 연산(custom op)에서 발생한 C++ 경고가 이제 Python으로 전파됩니다.** [#193451](https://github.com/pytorch/pytorch/pull/193451)을 참고하세요.
- **macOS 14는 더 이상 지원하지 않습니다.** PyTorch 2.15에서 MPS(Apple Silicon GPU) 가속을 사용하려면 macOS 15 이상이 필요합니다. macOS 14는 지원 종료(end of life)되었으며, 이를 위한 MPS 우회 코드(workaround)도 제거됐습니다. PyTorch가 macOS 14에서 여전히 실행될 수는 있지만, 그 환경에서는 테스트하지 않으며 해당 환경의 문제도 수정하지 않습니다. [#199261](https://github.com/pytorch/pytorch/pull/199261)을 참고하세요.
- **FSDP2가 샤딩된(sharded) 변화도와 샤딩되지 않은(unsharded) 변화도의 dtype을 분리합니다.** [#194434](https://github.com/pytorch/pytorch/pull/194434)를 참고하세요.

> - **Python 3.10 is no longer supported.** PyTorch 2.15 requires Python 3.11 or newer: 3.10 wheels are no longer built, and 3.10 is no longer tested in CI. Building from source on 3.10 is not supported, and upcoming releases will remove the remaining 3.10 code paths, including in torch.compile (Dynamo). See [#198233](https://github.com/pytorch/pytorch/pull/198233) and [#199417](https://github.com/pytorch/pytorch/pull/199417).
> - **nccl now selects the rewritten NCCL backend.** Set `TORCH_DIST_USE_NCCL2=0` or use the `nccl-legacy` backend to restore the legacy `ProcessGroupNCCL`. See [#192281](https://github.com/pytorch/pytorch/pull/192281).
> - **C++ warnings from torch.ops and custom ops now propagate to Python.** See [#193451](https://github.com/pytorch/pytorch/pull/193451).
> - **macOS 14 is no longer supported.** PyTorch 2.15 requires macOS 15 or newer for MPS (Apple Silicon GPU) acceleration. macOS 14 has reached end of life, and the MPS workarounds for it have been removed. PyTorch may still run on macOS 14, but it isn't tested there and issues on it won't be fixed. See [#199261](https://github.com/pytorch/pytorch/pull/199261).
> - **FSDP2 separates sharded and unsharded gradient dtypes.** See [#194434](https://github.com/pytorch/pytorch/pull/194434).
