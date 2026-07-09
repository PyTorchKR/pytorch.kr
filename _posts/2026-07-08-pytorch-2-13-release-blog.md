---
layout: blog_detail
title: "PyTorch 2.13 출시 공지"
author: PyTorch Foundation
category: ["pytorch.org", "translation"]
org_title: "PyTorch 2.13 Release Blog"
org_link: https://pytorch.org/blog/pytorch-2-13-release-blog/
---

PyTorch® 2.13([릴리즈 노트](https://github.com/pytorch/pytorch/releases/tag/v2.13.0))의 출시를 발표하게 되어 기쁩니다!
> We are excited to announce the release of PyTorch® 2.13 ([release notes](https://github.com/pytorch/pytorch/releases/tag/v2.13.0))!

PyTorch 2.13 릴리즈에는 다음과 같은 변경 사항이 포함되어 있습니다:
> The PyTorch 2.13 release features the following changes:

- **FlexAttention이 Apple Silicon(MPS)에 도입되어**, 희소(sparse) 패턴에서 SDPA 대비 최대 약 12배의 속도 향상을 제공하며, CUDA에서는 재현 가능한 변화도(gradient) 계산을 위한 결정론적(deterministic) 역방향(backward) 경로를 추가했습니다
- **CuTeDSL "Native DSL" 백엔드**가 Inductor에 주요 GPU 연산을 위한 두 번째 고성능 코드 경로(Triton과 함께)를 제공하며, 컴파일 속도가 더 빠릅니다
- **`nn.LinearCrossEntropyLoss`**가 최종 예측과 손실 계산 연산을 결합하여, 대규모 어휘(large-vocabulary) 언어 모델 학습에서 GPU 최대 메모리 사용량을 최대 4배까지 줄입니다
- **torchcomms**, PyTorch Distributed를 위한 새로운 통신 백엔드로, 대규모 클러스터 학습의 내결함성(fault tolerance), 확장성, 디버깅 용이성을 개선합니다
- **FSDP2가 이제 전용 프로세스 그룹을 통해 reduce-scatter와 all-gather 통신을 중첩(overlap)**(선택적 활성화)하여, 분산 학습 처리량을 높입니다
- **Linux에서 Torch에 대한 Python 3.15 휠 지원**을 pytorch 저장소 인덱스를 통해 제공합니다. 이 지원에는 자유 스레드(free-threaded) 3.15t와 호환되는 빌드가 포함됩니다.
- **더 넓은 플랫폼 지원**: ROCm은 네이티브 HIP CMake와 함께 AOTriton 0.12b를 추가하고, Arm은 Armv9-A `torch.compile` 타겟팅을 추가하며, Intel XPU는 새로운 디바이스 텔레메트리(telemetry) API를 노출합니다

> - **FlexAttention lands on Apple Silicon (MPS),** with up to ~12x speedup over SDPA on sparse patterns, and gains a deterministic backward path on CUDA for reproducible gradient computation
> - **CuTeDSL "Native DSL" backend** gives Inductor a second high-performance code path (alongside Triton) for key GPU operations, with faster compilation
> - **nn.LinearCrossEntropyLoss** combines final prediction and loss computation operations to cut peak GPU memory by up to 4x for large-vocabulary language model training
> - **torchcomms,** a new communications backend for PyTorch Distributed, improves fault tolerance, scalability, and debuggability for large-cluster training
> - **FSDP2 now overlaps reduce-scatter and all-gather communications** via a dedicated process group (opt-in), increasing distributed training throughput
> - **Python 3.15 wheel support for Torch on Linux** via the pytorch repository index. This support includes builds compatible with free-threaded 3.15t.
> - **Broader platform support**: ROCm gains AOTriton 0.12b with native HIP CMake, Arm adds Armv9-A torch.compile targeting, and Intel XPU exposes new device telemetry APIs

이번 릴리즈는 PyTorch 2.12 이후 526명의 기여자로부터 3,328회의 커밋으로 구성되었습니다. 헌신적인 커뮤니티의 기여에 진심으로 감사드립니다. 언제나 그렇듯, 새로운 버전을 사용하시며 생기는 문제를 보고해주시면 PyTorch 2.13을 개선하는 데 도움이 됩니다. PyTorch 2 시리즈를 시작하는 방법에 대한 자세한 정보는 [시작하기](https://pytorch.org/get-started/locally/) 페이지에서 확인할 수 있습니다.
> This release is composed of 3,328 commits from 526 contributors since PyTorch 2.12. We want to sincerely thank our dedicated community for your contributions. As always, we encourage you to try these out and report any issues as we improve 2.13. More information about how to get started with the PyTorch 2-series can be found at our [Getting Started](https://pytorch.org/get-started/locally/) page.

궁금한 점이 있으신가요? 2026년 7월 22일 오전 11시(PT)에 진행되는 라이브 Q&A에 참여해보세요. Alban Desmaison, Andrey Talman, Piotr Bialecki가 패널로, Chris Gottbrath가 모더레이터로 참여합니다. 이번 릴리즈에 대한 간략한 소개와 함께 여러분의 질문에 실시간으로 답변해 드립니다. [지금 등록하기](https://pytorch.org/event/pytorch-2-13-release-live-qa/)
> Have questions? Join us on July 22, 2026, at 11 a.m. PT for a live Q&A with panelists Alban Desmaison, Andrey Talman, Piotr Bialecki and moderator Chris Gottbrath. We will provide a brief overview of the release and answer your questions live. [Register today.](https://pytorch.org/event/pytorch-2-13-release-live-qa/)

2.x 시리즈 전반에 걸쳐 PyTorch는 연구 중심의 프레임워크에서, 대규모 프로덕션 학습 및 추론을 위한 통합된 하드웨어에 구애받지 않는(hardware-agnostic) 플랫폼으로 진화해 왔습니다. [PyTorch 2.11](https://pytorch.kr/blog/2026/pytorch-2-11-release-blog/)은 분산 학습을 위한 미분 가능한 집합 통신(differentiable collectives)과 차세대 GPU에서의 FlashAttention-4를 도입했습니다. [PyTorch 2.12](https://pytorch.kr/blog/2026/pytorch-2-12-release-blog/)는 장치에 구애받지 않는(device-agnostic) `torch.accelerator.Graph` API, 최대 100배 빨라진 배치 고윳값 분해(eigendecomposition), 마이크로스케일링(Microscaling) 양자화 내보내기(export) 지원을 추가했습니다.
> Throughout the 2.x series, PyTorch has been evolving from a research-first framework into a unified, hardware-agnostic platform for production training and inference at scale. [PyTorch 2.11](https://pytorch.org/blog/pytorch-2-11-release-blog/) introduced differentiable collectives for distributed training and FlashAttention-4 on next-generation GPUs. [PyTorch 2.12](https://pytorch.org/blog/pytorch-2-12-release-blog/) added a device-agnostic torch.accelerator.Graph API, up to 100× faster batched eigendecomposition, and Microscaling quantization export support.

PyTorch 2.13은 플랫폼과 규모 전반에 걸쳐 성능을 한층 더 끌어올립니다: FlexAttention이 최대 12배의 속도 향상과 함께 Apple Silicon에 도입되고, CuTeDSL이 CUTLASS 수준의 GEMM 커널을 Inductor에 제공하며, 융합된(fused) `nn.LinearCrossEntropyLoss`가 대규모 어휘 모델의 최대 메모리 사용량을 최대 4배까지 줄입니다. 분산 측면에서는 새로운 torchcomms 백엔드가 클러스터 규모에서 내결함성과 디버깅 용이성을 개선하고, FSDP2가 all-gather와 reduce-scatter 간 통신 중첩을 가능하게 하여 학습 처리량을 높입니다. 또한 이번 릴리즈는 [ExecuTorch의 통합](https://pytorch.org/blog/executorch-becomes-part-of-pytorch-core/)을 PyTorch Core로 편입하여, 온디바이스(on-device) 추론을 프레임워크의 일급(first-class) 기능으로 만들었습니다.
> PyTorch 2.13 pushes performance further across platforms and scales: FlexAttention lands on Apple Silicon with up to 12× speedups, CuTeDSL brings CUTLASS-grade GEMM kernels to Inductor, and fused nn.LinearCrossEntropyLoss cuts peak memory up to 4x for large-vocabulary models. On the distributed side, a new torchcomms backend improves fault tolerance and debuggability at cluster scale, and FSDP2 unlocks communication overlap between all-gather and reduce-scatter for higher training throughput. This release also marks [ExecuTorch's integration](https://pytorch.org/blog/executorch-becomes-part-of-pytorch-core/) into PyTorch Core, making on-device inference a first-class capability of the framework.

## 성능 개선 / Performance Improvements

### FlexAttention Flash 백엔드의 결정론적 역방향 / Deterministic Backward for FlexAttention Flash Backend

이 개선은 변화도 계산을 재현 가능하게 만들어, 기존 CUDA 구현의 FlexAttention에 대한 정확성과 디버깅에 초점을 맞춥니다. 기본적으로 FlexAttention flash 백엔드는 역방향 패스에서 dQ 누적을 위해 원자적(atomic) 연산을 사용하는데, 이는 비결정성(non-determinism)을 유발합니다 — 동일한 입력에 대해 반복 실행하면 변화도가 조금씩 달라질 수 있습니다. 이는 디버깅, 회귀 테스트, 재현 가능한 연구를 어렵게 만듭니다.
> This improvement focuses on correctness and debugging for the existing CUDA implementation of FlexAttention by making gradient computation reproducible. By default, the FlexAttention flash backend uses atomic operations in the backward pass for dQ accumulation, which introduces non-determinism — repeated runs on the same input can produce slightly different gradients. This makes debugging, regression testing, and reproducible research difficult.

새로운 결정론적 역방향 경로(`compute_dq_write_order`)는 원자적 연산을 사전 계산된 쓰기 순서(write ordering)로 대체하여, 의미 있는 성능 저하 없이 비트 단위로 재현 가능한 변화도를 보장합니다. `create_block_mask`에서 측정된 종단 간(end-to-end) 오버헤드는 긴 시퀀스 길이에서도 1% 미만이며(예: S=32768에서 +0.2%), 대부분의 프로덕션 워크로드에서 결정성을 사실상 공짜로 제공합니다. 사용자는 추가 코드 변경 없이 기존 `torch.use_deterministic_algorithms(True)` 설정을 통해 선택적으로 활성화할 수 있습니다.
> The new deterministic backward path (compute\_dq\_write\_order) replaces atomics with a pre-computed write ordering that guarantees bit-for-bit reproducible gradients without meaningful performance penalty. The measured end-to-end overhead on create\_block\_mask is well under 1% at longer sequence lengths (e.g., +0.2% at S=32768), making determinism effectively free for most production workloads. Users can opt in via the existing torch.use\_deterministic\_algorithms(True) setting with no additional code changes.

API 안정성: Unstable
> API Unstable

(PR [#174813](https://github.com/pytorch/pytorch/pull/174813) 작성: Driss Guessous, Meta)
> (PR [#174813](https://github.com/pytorch/pytorch/pull/174813) by Driss Guessous, Meta)

### 대규모 MPS 연산의 네이티브 Metal 이전 / Large MPS Op Migration to Native Metal

PyTorch의 MPS 백엔드는 이전에 대부분의 연산을 Apple의 MPSGraph 프레임워크에 위임했는데, 이는 디스패치마다 컴파일 및 스케줄링 오버헤드를 더합니다. 고빈도(high-frequency)이면서 지연 시간에 민감한 연산의 경우 이 오버헤드가 실행 시간을 지배할 수 있습니다. 이번 릴리즈는 광범위한 연산 집합을 손으로 작성한 Metal 컴퓨트 커널로 이전합니다 — 복사/캐스트(copy/cast), uniform/normal/randint, 비교, 리덕션(sum/mean), cumsum/cumprod, 정렬(멀티 블록 및 안정 정렬), 임베딩 역방향, 그리고 경계 검사(bounds checking)를 포함한 scatter/gather가 그 대상입니다.
> PyTorch's MPS backend previously delegated most operations to Apple's MPSGraph framework, which adds compilation and scheduling overhead per dispatch. For high-frequency, latency-sensitive operations this overhead can dominate execution time. This release migrates a broad set of operations to hand-written Metal compute kernels — copy/cast, uniform/normal/randint, comparisons, reductions (sum/mean), cumsum/cumprod, sort (multi-block and stable), embedding backward, and scatter/gather with bounds checking.

네이티브 Metal 경로는 MPSGraph의 연산별 컴파일 비용을 제거하고, 스레드 디스패치와 메모리 접근 패턴에 대한 직접적인 제어권을 PyTorch에 부여하여, Apple Silicon에서 일반적인 학습 및 추론 워크로드 전반에 걸쳐 커널 실행 지연 시간을 줄입니다.
> The native Metal path eliminates MPSGraph's per-op compilation cost and gives PyTorch direct control over thread dispatch and memory access patterns, reducing kernel launch latency across common training and inference workloads on Apple Silicon.

API 안정성: Unstable
> API Unstable

(PR [#184740](https://github.com/pytorch/pytorch/pull/184740) 작성: Nikita Shulga, Meta, [#185609](https://github.com/pytorch/pytorch/pull/185609) 및 [#185119](https://github.com/pytorch/pytorch/pull/185119) 작성: Irakli Salia, EPAM.)
> (PR [#184740](https://github.com/pytorch/pytorch/pull/184740) by Nikita Shulga, Meta, [#185609](https://github.com/pytorch/pytorch/pull/185609) and [#185119](https://github.com/pytorch/pytorch/pull/185119) by Irakli Salia, EPAM.)

### Apple Silicon(MPS)에서의 FlexAttention / FlexAttention on Apple Silicon (MPS)

커스텀 어텐션 패턴을 평범한 Python 함수로 표현하여 융합된 커널로 컴파일하는 PyTorch의 통합 API인 FlexAttention을 이제 Metal/MPS에서 사용할 수 있습니다. MPS 구현은 희소 프리필(prefill) 경로와 디코드(decode) 경로 모두(GQA 및 캡처된 버퍼 포함)에 대해 손으로 작성한 Metal 커널을 제공합니다. 따라서 필요한 어텐션 변형마다 커스텀 CUDA 커널을 작성하는 대신, FlexAttention에서는 2줄짜리 Python 함수를 작성하면 컴파일러가 자동으로 빠른 커널을 빌드해 줍니다.
> FlexAttention, PyTorch's unified API for expressing custom attention patterns as plain Python functions compiled into fused kernels, is now available on Metal/MPS. The MPS implementation provides hand-written Metal kernels for both the sparse prefill and decode paths (including GQA and captured buffers). So instead of writing a custom CUDA kernel for every attention variant you need, FlexAttention will write a 2-line Python function, and the compiler builds a fast kernel for you automatically.

희소 마스크에 대한 벤치마크 수치는 인상적입니다. 길고 희소한 어텐션 패턴에서 SDPA 대비 속도 향상이 상당합니다 — 예를 들어 256개 요소 슬라이딩 윈도우(밀도 0.8%)를 사용하는 1×8×32768×64 shape에서 FlexAttention은 약 35ms에 실행되는 반면 SDPA는 약 431ms가 걸립니다(**약 12.3배**). 더 작은 8192 길이 / 64 윈도우 사례에서는 **약 4.15배**를 달성합니다. 예상대로 밀집(dense) 패턴은 여전히 SDPA가 유리합니다.
> The benchmark numbers are impressive for sparse masks. On long, sparse attention patterns the speedups over SDPA are substantial — e.g. on a 1×8×32768×64 shape with a 256-element sliding window (0.8% density), FlexAttention runs in ~35 ms vs ~431 ms for SDPA (**~12.3x**); a smaller 8192-length / 64-window case achieves **~4.15x**. Dense patterns continue to favor SDPA, as expected.

API 안정성: Unstable
> API Unstable

(PR [#182552](https://github.com/pytorch/pytorch/pull/182552), [#186215](https://github.com/pytorch/pytorch/pull/186215), [#181575](https://github.com/pytorch/pytorch/pull/181575) 작성: Irakli Salia, EPAM)
> (PR [#182552](https://github.com/pytorch/pytorch/pull/182552), [#186215](https://github.com/pytorch/pytorch/pull/186215), and [#181575](https://github.com/pytorch/pytorch/pull/181575) by Irakli Salia, EPAM)

## 핵심 기능 / Core Features

### nn.LinearCrossEntropyLoss

표준적인 대규모 어휘 학습(예: 10만 개 이상의 토큰 어휘를 가진 언어 모델)에서 교차 엔트로피(cross-entropy) 손실을 계산하려면 어휘 전체에 대한 로짓(logits) 행렬을 물리적으로 생성(materialize)해야 하며, 이는 수십 기가바이트의 GPU 메모리를 소비할 수 있습니다. `nn.LinearCrossEntropyLoss`(및 이에 대응하는 `linear_cross_entropy` 함수형)는 최종 선형 사영(linear projection)과 교차 엔트로피 계산을 하나의 모듈로 융합하여, 어휘 차원을 청크(chunk) 단위로 처리하며 전체 로짓 행렬을 결코 물리적으로 생성하지 않습니다. 이는 융합되지 않은(unfused) 경로와 수치적으로 동일한 결과를 유지하면서 대규모 어휘 워크로드의 최대 메모리 사용량을 최대 약 4배까지 줄입니다. 이 구현은 레이블 스무딩(label smoothing), 가중치 공유(weight tying), z-loss 정규화를 기본적으로 지원하며, 추가 최적화를 위해 `torch.compile`과 통합됩니다. 별도의 `nn.Linear` + `nn.CrossEntropyLoss`를 대체하는 드롭인(drop-in) 방식이므로, 도입 시 다른 코드 변경이 필요하지 않습니다.
> In standard large-vocabulary training (e.g., language models with 100K+ token vocabularies), computing the cross-entropy loss requires materializing the entire logits matrix over the vocabulary, which can consume tens of gigabytes of GPU memory. nn.LinearCrossEntropyLoss (and the corresponding linear\_cross\_entropy functional) fuses the final linear projection and cross-entropy computation into a single module that processes the vocabulary dimension in chunks, never materializing the full logits matrix. This reduces peak memory by up to ~4x for large-vocabulary workloads while maintaining numerical equivalence with the unfused path. The implementation supports label smoothing, weight tying, and z-loss regularization out of the box, and integrates with torch.compile for further optimization. As a drop-in replacement for separate nn.Linear + nn.CrossEntropyLoss, adoption requires no other code changes.

API 안정성: Unstable
> API Unstable

([#172446](https://github.com/pytorch/pytorch/pull/172446), [#172286](https://github.com/pytorch/pytorch/pull/172286), [#185852](https://github.com/pytorch/pytorch/pull/185852) 작성: Pearu Peterson, OpenTeams 참고.)
> (See [#172446](https://github.com/pytorch/pytorch/pull/172446), [#172286](https://github.com/pytorch/pytorch/pull/172286), and [#185852](https://github.com/pytorch/pytorch/pull/185852) by Pearu Peterson, OpenTeams.)

### torch.load로 Safetensors 직접 로드 / Load Safetensors Directly with torch.load

Safetensors는 메모리 매핑(memory-mapped) 로딩과 임의 코드 실행 위험이 없다는 점 덕분에 모델 가중치를 배포하는 형식으로 널리 채택되었습니다(Hugging Face, Stability AI 등에서 사용). 이전에는 safetensors 파일을 로드하려면 별도의 라이브러리를 설치하고 임포트해야 했습니다. 이제 `torch.load("foo.safetensors")`가 형식을 자동으로 감지하고 Tensor를 직접 반환하며 네이티브로 동작합니다. 이는 일반적인 워크플로우에서 의존성 하나를 제거하고, safetensors 형식으로 배포된 모델을 로드할 때 PyTorch를 매끄러운 드롭인 대체재로 만들어 줍니다.
> Safetensors has become a widely adopted format for distributing model weights (used by Hugging Face, Stability AI, and others) due to its memory-mapped loading and lack of arbitrary code execution risk. Previously, loading safetensors files required installing and importing a separate library. torch.load("foo.safetensors") now works natively, detecting the format automatically and returning tensors directly. This removes a dependency for common workflows and makes PyTorch a seamless drop-in for loading models distributed in safetensors format.

API 안정성: Unstable
> API Unstable

(PR [#170592](https://github.com/pytorch/pytorch/pull/170592) 작성: Nikita Shulga, Meta)
> (PR [#170592](https://github.com/pytorch/pytorch/pull/170592) by Nikita Shulga, Meta)

### Python 3.15 바이너리 지원 – 릴리즈 엔지니어링 / Python 3.15 Binary Support – Release Engineering

이제 PyTorch 휠이 실험적 자유 스레드(free-threaded) 3.15t 빌드를 포함하여 Python 3.15를 지원합니다. 중요 참고 사항: Python 3.15는 현재 사전 릴리즈(베타) 상태이며, 최종 안정 릴리즈는 2026년 10월로 예정되어 있습니다. 지원은 torch 휠에 한정되며(torchvision은 아직 3.15용으로 빌드되지 않음), x86\_64 및 aarch64의 Linux 빌드로 한정되어 CPU, CUDA, ROCm, XPU 변형을 포괄합니다. `torch.compile`은 Python 3.15에서 아직 지원되지 않습니다. 이번 릴리즈에는 이 Python 버전에 대한 Windows 또는 macOS 3.15 휠이 없습니다.
> PyTorch wheels now support Python 3.15, including the experimental free-threaded 3.15t build. Important Note: Python 3.15 is currently in pre-release (beta), with the final stable release scheduled for October 2026. Support is limited to torch wheels only (torchvision is not built for 3.15 yet) and to Linux builds on x86\_64 and aarch64, covering the CPU, CUDA, ROCm, and XPU variants. torch.compile is not yet supported on Python 3.15. There are no Windows or macOS 3.15 wheels for this python version in this release.

Python 3.15 및 3.15t 휠은 PyPI에 게시되지 않습니다 — 오직 download.pytorch.org를 통해서만 다음 명령 중 하나로 다운로드할 수 있습니다:
> Python 3.15 and 3.15t wheels are not published to PyPI — they are available to download only via download.pytorch.org, using any of the following commands:

```sh
# CPU
pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA (CUDA 버전으로 치환, 예: cu126 / cu130)
pip3 install torch --index-url https://download.pytorch.org/whl/cu130

# ROCm (ROCm 버전으로 치환)
pip3 install torch --index-url https://download.pytorch.org/whl/rocm7.2

# XPU
pip3 install torch --index-url https://download.pytorch.org/whl/xpu
```

동일한 명령을 자유 스레드 인터프리터 아래에서 실행하면 자유 스레드 3.15t 빌드가 설치됩니다.
> The same commands install the free-threaded 3.15t build when run under a free-threaded interpreter.

API 안정성: Unstable. 업데이트는 트래커 이슈 참고: [#184352](https://github.com/pytorch/pytorch/issues/184352)
> API Unstable. For updates See tracker issue: #[184352](https://github.com/pytorch/pytorch/issues/184352)

(PR [#182954](https://github.com/pytorch/pytorch/pull/182954) 작성: Nikita Shulga, PR [#184600](https://github.com/pytorch/pytorch/pull/184600) 및 [#186244](https://github.com/pytorch/pytorch/pull/186244) 작성: Andrey Talman, Meta, [#186017](https://github.com/pytorch/pytorch/pull/186017) 작성: Rob Timpe, OpenTeams)
> (PR [#182954](https://github.com/pytorch/pytorch/pull/182954) by Nikita Shulga, PR [#184600](https://github.com/pytorch/pytorch/pull/184600) and [#186244](https://github.com/pytorch/pytorch/pull/186244) by Andrey Talman, Meta, [#186017](https://github.com/pytorch/pytorch/pull/186017) by Rob Timpe, OpenTeams)

## 분산 학습 / Distributed Training

### torchcomms 백엔드 / torchcomms Backend

PyTorch Distributed는 지금까지 집합 통신(all-reduce, all-gather 등)을 위해 c10d의 ProcessGroup 추상화에 의존해 왔으며, 이는 주로 NCCL을 중심으로 설계되었습니다. 분산 학습이 더 복잡해짐에 따라(다차원 병렬화, 탄력적 확장, 이질적 인터커넥트), 기존 백엔드의 오류 처리 및 관측성(observability) 한계가 운영 팀에게 병목이 됩니다.
> PyTorch Distributed has historically relied on c10d's ProcessGroup abstraction for collective communications (all-reduce, all-gather, etc.), which was designed primarily around NCCL. As distributed training grows more complex (multi-dimensional parallelism, elastic scaling, heterogeneous interconnects), the original backend's error handling and observability limitations become bottlenecks for operations teams.

torchcomms는 PyTorch Distributed의 CI 및 디바이스 메시(device-mesh) 경로에 통합된 새로운 통신 백엔드로, 개선된 내결함성(정상적인 타임아웃 처리 및 부분 그룹 복구), 대규모 클러스터 전반에서의 향상된 확장성, 구조화된 로깅과 집합 통신 추적(collective tracing)을 통한 더 풍부한 디버깅 용이성을 제공합니다. 이는 기존 c10d 백엔드에 대한 현대적인 대안 역할을 하면서도 API 호환성을 유지합니다.
> torchcomms is a new communications backend integrated into PyTorch Distributed's CI and device-mesh paths, providing improved fault tolerance (graceful timeout and partial-group recovery), better scalability across large clusters, and richer debuggability through structured logging and collective tracing. It serves as a modern alternative to the existing c10d backends while maintaining API compatibility.

API 안정성: Unstable
> API Unstable

(PR [#181662](https://github.com/pytorch/pytorch/pull/181662) 작성: Tristan Rice, Meta, [#178533](https://github.com/pytorch/pytorch/pull/178533) 작성: Pangiotis Kourdis, Intel, [#182057](https://github.com/pytorch/pytorch/pull/182057) 작성: Kapil Sharma, Meta)
> (PR [#181662](https://github.com/pytorch/pytorch/pull/181662) by Tristan Rice, Meta, [#178533](https://github.com/pytorch/pytorch/pull/178533) Pangiotis Kourdis, Intel, and [#182057](https://github.com/pytorch/pytorch/pull/182057) by Kapil Sharma, Meta)

### FSDP2 별도 Reduce-Scatter 그룹 / FSDP2 Separate Reduce-Scatter Group

완전 분할 데이터 병렬(fully-sharded data-parallel, FSDP) 학습에서 all-gather와 reduce-scatter는 기본적으로 단일 NCCL 커뮤니케이터(communicator)를 공유합니다. NCCL이 동일한 커뮤니케이터의 연산을 직렬화하기 때문에, 이 두 집합 통신은 중첩될 수 없어 통신 대역폭이 충분히 활용되지 못합니다. `FSDPModule.set_separate_reduce_scatter_group(enable=True)`는 reduce-scatter에 자체 전용 NCCL 커뮤니케이터를 부여하여, all-gather 연산과 동시에 진행될 수 있게 합니다. 이는 AG/RS 중첩을 가능하게 하여, 모델 코드 변경 없이 완전 분할 워크로드의 학습 처리량을 향상시킵니다.
> In fully-sharded data-parallel (FSDP) training, all-gather and reduce-scatter share a single NCCL communicator by default. Because NCCL serializes operations on the same communicator, these two collectives cannot overlap, leaving communication bandwidth underutilized. FSDPModule.set\_separate\_reduce\_scatter\_group(enable=True) gives reduce-scatter its own dedicated NCCL communicator, allowing it to progress concurrently with all-gather operations. This enables AG/RS overlap, improving training throughput for fully-sharded workloads without any changes to model code.

API 안정성: Unstable
> API Unstable

(PR [#186335](https://github.com/pytorch/pytorch/pull/186335) 작성: Wei Feng, Meta)
> (PR [#186335](https://github.com/pytorch/pytorch/pull/186335) by Wei Feng, Meta)

## 컴파일 및 내보내기 / Compilation and Export

### torch.compiler.set_default_backend

커스텀 또는 트리 외(out-of-tree) 컴파일러 백엔드(예: 특수 하드웨어용)를 사용할 때, 사용자는 이전에 모든 `torch.compile()` 호출에 `backend=`를 명시적으로 전달해야 했으며, 이는 코드를 장황하고 오류가 발생하기 쉽게 만들었습니다. `torch.compiler.set_default_backend`는 `torch.set_default_dtype`과 `torch.set_default_device`가 확립한 패턴을 따라, 백엔드 작성자나 인프라 팀이 프로세스 전역 기본값을 한 번에 설정할 수 있게 합니다. 이후의 모든 `torch.compile()` 호출은 명시적인 `backend=` 인자가 이를 재정의하지 않는 한 자동으로 해당 백엔드를 사용합니다. 이는 대규모 코드베이스 전반에서 커스텀 백엔드의 도입을 단순화합니다.
> When using custom or out-of-tree compiler backends (e.g., for specialized hardware), users previously had to pass backend= explicitly to every torch.compile() call, making code verbose and error-prone. torch.compiler.set\_default\_backend mirrors the pattern established by torch.set\_default\_dtype and torch.set\_default\_device, letting backend authors or infrastructure teams set a process-wide default once. All subsequent torch.compile() calls automatically use that backend unless an explicit backend= argument overrides it. This simplifies adoption of custom backends across large codebases.

API 안정성: Unstable
> API Unstable

(PR [#178944](https://github.com/pytorch/pytorch/pull/178944) 작성: Angela Yi, Meta)
> (PR [#178944](https://github.com/pytorch/pytorch/pull/178944) by Angela Yi, Meta)

## 플랫폼 기능 및 업데이트 / Platform Features and Updates

### CUDA

#### Inductor를 위한 CuTeDSL "Native DSL" 백엔드 / CuTeDSL "Native DSL" Backend for Inductor

CuTeDSL은 NVIDIA의 CuTe(CUDA Templates) 라이브러리 위에 구축된 Python 네이티브 도메인 특화 언어(domain-specific language)로, GPU Tensor 레이아웃, 타일링(tiling) 전략, 메모리 접근 패턴에 대한 직접적인 제어권을 개발자에게 제공합니다. 이제 PyTorch의 Inductor 컴파일러는 Triton과 함께 CuTeDSL을 대체 코드 생성 백엔드로 사용할 수 있습니다 — 구체적으로는 트랜스포머 학습에서 가장 성능이 중요한 두 연산인 행렬 곱셈(GEMM)과 정규화(RMSNorm)에 대해서입니다. 이러한 Quack 기반의 커널 재정의(override)는 이 워크로드에 Triton을 요구하지 않으면서도 더 높은 품질의 행렬 곱셈 코드를 생성합니다. 커널 컴파일도 스레드 풀에서 서브프로세스 풀로 이전되어, Python의 GIL 병목을 제거하고 컴파일 시점의 병렬성을 개선했습니다.
> CuTeDSL is a Python-native domain-specific language built on NVIDIA's CuTe (CUDA Templates) library, giving developers direct control over GPU tensor layouts, tiling strategies, and memory access patterns. PyTorch's Inductor compiler can now use CuTeDSL as an alternative code-generation backend alongside Triton — specifically for matrix multiplication (GEMM) and normalization (RMSNorm), two of the most performance-critical operations in transformer training. These Quack-derived kernel overrides produce higher-quality matrix-multiply code without requiring Triton for these workloads. Kernel compilation has also moved from the thread pool to a subprocess pool, eliminating Python's GIL bottleneck and improving compile-time parallelism.

API 안정성: Unstable
> API Unstable

(PR [#181267](https://github.com/pytorch/pytorch/pull/181267) 작성: Michael Lazos, [#182108](https://github.com/pytorch/pytorch/pull/182108) 작성: Simon Layton, [#186310](https://github.com/pytorch/pytorch/pull/186310) 작성: Driss Guessous, Meta)
> (PR [#181267](https://github.com/pytorch/pytorch/pull/181267) by Michael Lazos, [#182108](https://github.com/pytorch/pytorch/pull/182108) by Simon Layton, and [#186310](https://github.com/pytorch/pytorch/pull/186310) by Driss Guessous, Meta)

### ROCm

#### AOTriton 0.12b, Origami GEMM 선택, 네이티브 HIP CMake / AOTriton 0.12b, Origami GEMM Selection, Native HIP CMake

ROCm 스택은 세 가지 개선을 얻습니다: AOTriton이 0.12b로 발전하여 비대칭 헤드 차원(asymmetric head dimensions), 결정론적 알고리즘, 새로운 GPU 타겟(gfx1100/gfx1151이 stable로 승격, gfx950에서 FlashAttention v3 부분 지원)을 지원합니다. Origami 도구는 무차별 대입(brute-force) 방식의 오토튜닝을 분석적 GEMM 구성 선택으로 대체하여 튜닝 시간을 단축합니다. 빌드 시스템은 이제 CMake의 네이티브 HIP 언어 지원을 사용합니다.
> The ROCm stack gains three improvements: AOTriton advances to 0.12b with support for asymmetric head dimensions, deterministic algorithms, and new GPU targets (gfx1100/gfx1151 promoted to stable, partial FlashAttention v3 on gfx950). The Origami tool replaces brute-force autotuning with analytical GEMM-configuration selection, cutting tune time. The build system now uses CMake's native HIP language support.

API 안정성: Unstable
> API Unstable

(PR [#184288](https://github.com/pytorch/pytorch/pull/184288) 및 [#172512](https://github.com/pytorch/pytorch/pull/172512) 작성: Xinya Zhang, Umesh Chand, AMD)
> (PR [#184288](https://github.com/pytorch/pytorch/pull/184288) and [#172512](https://github.com/pytorch/pytorch/pull/172512) by Xinya Zhang and Umesh Chand, AMD)

### Arm

#### torch.compile을 위한 Armv9-A 타겟 지원 / Armv9-A Target Support for torch.compile

AArch64에서의 `torch.compile`이 이제 Armv9-A CPU(예: AWS Graviton4에서 사용되는 Neoverse V2)를 인식하여, 올바른 타겟 트리플(target triple)과 기능 세트(128비트 및 256비트 SVE)를 Inductor 코드 생성 전반에 전달합니다. x86에서는 동작 변경이 없습니다.
> torch.compile on AArch64 now recognizes Armv9-A CPUs (e.g., Neoverse V2 used in AWS Graviton4), propagating the correct target triple and feature set (128-bit and 256-bit SVE) through Inductor codegen. No behavior change for x86.

API 안정성: Unstable
> API Unstable

(PR [#184555](https://github.com/pytorch/pytorch/pull/184555) 작성: Zhibo Li, Arm)
> (PR [#184555](https://github.com/pytorch/pytorch/pull/184555) by Zhibo Li, Arm)

### XPU (Intel GPU) / XPU (Intel GPUs)

#### 디바이스 텔레메트리 API / Device Telemetry APIs

Intel GPU를 위한 새로운 쿼리 API가 런타임 디바이스 상태를 노출합니다: 메모리 사용량(`torch.xpu.device_memory_used`), 활용률(utilization), 소비 전력(power draw), 클럭 속도(clock rate), 온도, 그리고 디바이스 전역 동기화 및 하드웨어 속성(마지막 수준 캐시(last-level cache) 크기, 통합 GPU(integrated-GPU) 감지)이 그 대상입니다.
> New query APIs for Intel GPUs expose runtime device state: memory usage (torch.xpu.device\_memory\_used), utilization, power draw, clock rate, and temperature, plus device-wide synchronization and hardware properties (last-level cache size, integrated-GPU detection).

API 안정성: Unstable
> API Unstable

(PR [#183431](https://github.com/pytorch/pytorch/pull/183431), [#183429](https://github.com/pytorch/pytorch/pull/183429), [#183428](https://github.com/pytorch/pytorch/pull/183428), [#183427](https://github.com/pytorch/pytorch/pull/183427) 작성: Guangye Yu, Intel)
> (PR [#183431](https://github.com/pytorch/pytorch/pull/183431), [#183429](https://github.com/pytorch/pytorch/pull/183429), [#183428](https://github.com/pytorch/pytorch/pull/183428), and [#183427](https://github.com/pytorch/pytorch/pull/183427) by Guangye Yu, Intel)

### C++ ABI

#### torch::stable::Generator

커스텀 커널 작성자가 이제 안정적인(stable) C++ ABI를 통해 `at::Generator`를 가져올 수 있게 되어(이전에는 항상 null로 전달됨), PyTorch 내부에 대한 재컴파일 없이 ABI 안정 확장(ABI-stable extensions)에서 RNG를 지원할 수 있습니다.
> Custom kernel authors can now retrieve an at::Generator through the stable C++ ABI (previously always passed as null), enabling RNG support in ABI-stable extensions without recompilation against PyTorch internals.

API 안정성: Unstable
> API Unstable

(PR [#186423](https://github.com/pytorch/pytorch/pull/186423) 및 [#183930](https://github.com/pytorch/pytorch/pull/183930) 작성: Jane Xu, Meta 및 Chris Leonard, Red Hat)
> (PR [#186423](https://github.com/pytorch/pytorch/pull/186423) and [#183930](https://github.com/pytorch/pytorch/pull/183930) by Jane Xu, Meta and Chris Leonard, Red Hat)

## 프로파일링 및 디버깅 / Profiling and Debugging

### 실험적 CUPTI 모니터 프로파일러 / Experimental CUPTI Monitor Profiler

PyTorch의 기존 프로파일러는 CUPTI의 activity API를 통해 GPU 활동을 수집하는데, 이는 타이밍을 왜곡하고 멀티 스레드 워크로드에서 GIL과 경합할 수 있는 동기화 지점을 필요로 합니다. 새로운 실험적 CUPTI 모니터 백엔드는 GPU 지표를 비동기적으로(GIL을 완전히 벗어나) 수집하여, 기존 CPU 프로파일러 경로를 재사용하면서 프로파일링으로 인한 오버헤드를 제거합니다. 그 결과 실제 실행 타이밍을 정확히 반영하는 병합된 Chrome 트레이스가 만들어지며, `record_function` 범위에 대한 GPU 측 사용자 어노테이션(annotation)이 자동으로 재생성됩니다. 이는 성능을 눈에 띄게 교란하지 않으면서 프로덕션 학습 루프를 프로파일링하는 것을 실용적으로 만듭니다.
> PyTorch's existing profiler collects GPU activity through CUPTI's activity API, which requires synchronization points that can distort timing and contend with the GIL in multi-threaded workloads. The new experimental CUPTI monitor backend collects GPU metrics asynchronously (completely off the GIL), eliminating profiling-induced overhead while reusing the existing CPU profiler path. The result is merged Chrome traces that accurately reflect real-world execution timing, with GPU-side user annotations automatically recreated for record\_function scopes. This makes it practical to profile production training loops without measurably perturbing their performance.

API 안정성: Unstable
> API Unstable

(PR [#186037](https://github.com/pytorch/pytorch/pull/186037) 및 [#186295](https://github.com/pytorch/pytorch/pull/186295) 작성: Natalia Gimelshein, Meta)
> (PR [#186037](https://github.com/pytorch/pytorch/pull/186037) and [#186295](https://github.com/pytorch/pytorch/pull/186295) by Natalia Gimelshein, Meta)

### CUDAGraph.get_graph_data()

CUDA 그래프 캡처의 성능 문제를 디버깅할 때, 내부 구조(어떤 커널이 실행되는지, 그 의존성, 실행 순서)를 이해하려면 이전에는 외부 도구나 수동 검사가 필요했습니다. `CUDAGraph.get_graph_data()`는 전체 그래프 토폴로지를 프로그래밍 방식으로 노출합니다: 노드 유형, 커널 이름, 의존성 엣지(dependency edges), 그리고 CUPTI 프로파일러 출력과 일치하도록 다시 매핑된 ID가 그것입니다. 이를 통해 개발자는 캡처된 그래프 구조와 프로파일링된 커널 실행을 상관 분석할 수 있어, 캡처된 그래프 내의 병목, 불필요한 직렬화, 최적이 아닌 커널 융합을 손쉽게 식별할 수 있습니다.
> When debugging performance issues in CUDA graph captures, understanding the internal structure (which kernels run, their dependencies, execution order) previously required external tools or manual inspection. CUDAGraph.get\_graph\_data() exposes the full graph topology programmatically: node types, kernel names, dependency edges, and IDs remapped to match CUPTI profiler output. This lets developers correlate captured graph structure with profiled kernel launches, making it straightforward to identify bottlenecks, unnecessary serialization, or suboptimal kernel fusion within a captured graph.

API 안정성: Unstable
> API Unstable

(PR [#183165](https://github.com/pytorch/pytorch/pull/183165) 작성: Natalia Gimelshein, Meta)
> (PR [#183165](https://github.com/pytorch/pytorch/pull/183165) by Natalia Gimelshein, Meta)

## 지원 중단 및 하위 호환성 변경 사항 / Deprecations and Backwards-Incompatible Changes

- **Named tensor 제거.** 지원이 중단되었던 named-tensor 기능(`Tensor.names` 및 관련 API)이 오버헤드와 코드 비대화를 줄이기 위해 완전히 제거(hard-removed)되었습니다. [#173895](https://github.com/pytorch/pytorch/pull/173895)을 참고하세요.
- **분산 집합 통신이 `_single` 방식으로 이름 변경.** `all_gather_into_tensor` → `all_gather_single`, `reduce_scatter_tensor` → `reduce_scatter_single`으로, torchcomms와 정렬됩니다. 기존 이름은 얇은 래퍼(thin wrapper)로 남아 `FutureWarning`을 통해 지원 중단 표시됩니다. [#186123](https://github.com/pytorch/pytorch/pull/186123)을 참고하세요.
- **Bazel 빌드 제거.** Bazel 빌드는 널리 채택된 적이 없었고 구식 Bazel 6에 의존했기에 제거되었습니다. [#180883](https://github.com/pytorch/pytorch/pull/180883)을 참고하세요.

> - **Named tensors removed.** The deprecated named-tensor feature (Tensor.names and associated APIs) has been hard-removed to cut overhead and code bloat. See [#173895](https://github.com/pytorch/pytorch/pull/173895).
> - **Distributed collectives renamed to a \_single scheme.** all\_gather\_into\_tensor → all\_gather\_single and reduce\_scatter\_tensor → reduce\_scatter\_single, aligning with torchcomms. The old names remain as thin wrappers marked deprecated via FutureWarning. See [#186123](https://github.com/pytorch/pytorch/pull/186123).
> - **Bazel build removed.** The Bazel build was never broadly adopted and depended on an antiquated Bazel 6; it has been removed. See [#180883](https://github.com/pytorch/pytorch/pull/180883).

## 기능 외 업데이트 / Non-Feature Updates

- **Python 지원**: CPython 3.13t가 Linux 바이너리 매트릭스에서 제외되었습니다. (예비 Python 3.15 및 3.15t 바이너리 지원은 위 내용을 참고하세요.) [#182951](https://github.com/pytorch/pytorch/pull/182951)을 참고하세요.
- **CUDA**: CUDA 13.0이 기본 빌드로 유지됩니다. 이제 CUDA Linux에 대해 소형 휠(small wheels)이 항상 빌드되며, CUDA 12.8/12.9 빌드는 제거되었습니다. ptxas는 더 이상 cu13 바이너리에 번들되지 않습니다. [#180612](https://github.com/pytorch/pytorch/pull/180612), [#174716](https://github.com/pytorch/pytorch/pull/174716)을 참고하세요.
- **Triton**: 핀(pin)이 3.7.1로 업데이트되었습니다. [#186792](https://github.com/pytorch/pytorch/pull/186792)를 참고하세요.
- **oneDNN**: 서브모듈이 v3.12로 업그레이드되었습니다. [#181222](https://github.com/pytorch/pytorch/pull/181222)를 참고하세요.

> - **Python support**: CPython 3.13t dropped from the Linux binary matrix. (Preliminary Python 3.15 and 3.15t binary support is found above.) See [#182951](https://github.com/pytorch/pytorch/pull/182951).
> - **CUDA**: CUDA 13.0 remains the default build; small wheels are now always built for CUDA Linux and the CUDA 12.8/12.9 builds were removed; ptxas is no longer bundled in the cu13 binary. See [#180612](https://github.com/pytorch/pytorch/pull/180612), [#174716](https://github.com/pytorch/pytorch/pull/174716).
> - **Triton**: pin advanced to 3.7.1. See [#186792](https://github.com/pytorch/pytorch/pull/186792).
> - **oneDNN**: submodule upgraded to v3.12. See [#181222](https://github.com/pytorch/pytorch/pull/181222).

*업데이트(7/8): 더 이상 사용하지 않는 지정(designation)인 prototype이 사용된 두 곳을 제거했습니다.*
> *Updated (7/8): Removed two instances of prototype which is a designation that we no longer use.*
