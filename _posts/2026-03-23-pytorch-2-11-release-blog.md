---
layout: blog_detail
title: "PyTorch 2.11 출시 공지"
image: /assets/blog/2026-03-23-pytorch-2-11-release-blog/hero.png
author: PyTorch Team
category: ["pytorch.org", "translation"]
org_title: "PyTorch 2.11 Release Blog"
org_link: https://pytorch.org/blog/pytorch-2-11-release-blog/
---

PyTorch 2.11([릴리즈 노트](https://github.com/pytorch/pytorch/releases/tag/v2.11.0))의 출시를 발표하게 되어 기쁩니다!
> We are excited to announce the release of PyTorch® 2.11 ([release notes](https://github.com/pytorch/pytorch/releases/tag/v2.11.0))!

PyTorch 2.11 릴리즈에는 다음과 같은 변경 사항이 포함되어 있습니다:
> The PyTorch 2.11 release features the following changes:

- **분산 학습을 위한 미분 가능한 집합 통신(Differentiable Collectives)**
- **FlexAttention에 Hopper 및 Blackwell GPU에서 FlashAttention-4 백엔드 추가**
- **MPS (Apple Silicon) 포괄적 연산자 확장**
- **RNN/LSTM GPU 내보내기(Export) 지원**
- **XPU Graph**

> - **Differentiable Collectives for Distributed Training**
> - **FlexAttention now has a FlashAttention-4 backend on Hopper and Blackwell GPUs.**
> - **MPS (Apple Silicon) Comprehensive Operator Expansion**
> - **RNN/LSTM GPU Export Support**
> - **XPU Graph**

이번 릴리즈는 PyTorch 2.10 이후 432명의 기여자로부터 2,723회의 커밋으로 구성되었습니다. 헌신적인 커뮤니티의 기여에 진심으로 감사드립니다. 언제나 그렇듯, 새로운 버전을 사용하시며 생기는 문제를 보고해주시면 PyTorch 2.11을 개선하는 데 도움이 됩니다. PyTorch 2 시리즈를 시작하는 방법에 대한 자세한 정보는 [시작하기](https://pytorch.org/get-started/pytorch-2.0/) 페이지에서 확인할 수 있습니다.
> This release is composed of 2723 commits from 432 contributors since PyTorch 2.10. We want to sincerely thank our dedicated community for your contributions. As always, we encourage you to try these out and report any issues as we improve 2.11. More information about how to get started with the PyTorch 2-series can be found at our [Getting Started](https://pytorch.org/get-started/pytorch-2.0/) page.

3월 31일 화요일 오전 10시에 Andrey Talman과 Nikita Shulga가 라이브 세션을 진행하여 2.11의 새로운 기능들을 소개합니다. 분산 학습을 위한 미분 가능한 집합 통신, Hopper 및 Blackwell GPU에서의 FlashAttention-4 백엔드를 활용한 FlexAttention, MPS 확장 등을 다루며, 이어서 라이브 Q&A가 진행됩니다. [참가 등록하기](https://streamyard.com/watch/zHmCTfH6Y3zQ)
> On Tuesday, March 31st at 10 am, Andrey Talman and Nikita Shulga will host a live session to walk through what's new in 2.11, including Differentiable Collectives for Distributed Training, FlexAttention with a FlashAttention-4 backend on Hopper and Blackwell GPUs, MPS expansion, and more, followed by a live Q&A. [Register to attend.](https://streamyard.com/watch/zHmCTfH6Y3zQ)

### API-불안정(UNSTABLE) 기능 / API-UNSTABLE Features

**분산 학습을 위한 미분 가능한 집합 통신 / Differentiable Collectives for Distributed Training**

함수형 집합 통신(functional collectives)에 미분 가능성(differentiability) 지원을 추가하여, 집합 통신 연산을 통해 역전파할 수 있는 학습 워크플로우를 구현할 수 있게 되었습니다. 이는 분산 딥러닝 연구 및 고급 학습 기법에 있어 중요한 발전으로, 사용자 정의 autograd 함수 없이도 구현할 수 있습니다.
> Added differentiability support for functional collectives, enabling training workflows that can backpropagate through collective operations. This is a significant advancement for distributed deep learning research and advanced training techniques, which may be implemented without the need for custom autograd functions.

**FlexAttention에 Hopper 및 Blackwell GPU에서 FlashAttention-4 백엔드 추가 / FlexAttention now has a FlashAttention-4 backend on Hopper and Blackwell GPUs.**

이 백엔드는 CuTeDSL 점수/마스크 수정 함수를 자동으로 생성하고 PyTorch에서 FlashAttention-4 커널을 JIT 인스턴스화하는 기능을 지원하여, 연산 바운드 워크로드에서 기존 Triton 구현 대비 1.2~3.2배의 속도 향상을 제공합니다. 이 기능은 아직 활발히 개발 중이며 안정화 과정에서 변경될 수 있습니다. 설정 방법과 현재 제한 사항은 [FlexAttention + FlashAttention-4 블로그 글](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)을 참고하세요.
> This backend adds support for automatically generating CuTeDSL score/mask modification functions and JIT-instantiating FlashAttention-4 kernels from PyTorch, enabling 1.2× to 3.2× speedups over the existing Triton implementation on compute-bound workloads. This feature is still under active development and may change as it stabilizes; for setup details and current limitations, see the [FlexAttention + FlashAttention-4 blog post](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/).

**MPS (Apple Silicon) 개발 개선 및 연산자 확장 / MPS (Apple Silicon) Development Improvements / Operator Expansion**

이번 릴리즈에는 MPS 백엔드의 오류 보고 지원과 함께 연산자 범위의 지속적인 확장이 포함되어 있습니다. 새로운 분포 함수(log\_normal, cauchy, geometric), 연산자 마이그레이션(erfcx, 모든 연산 모드를 지원하는 grid\_sampler\_2d), 정수 및 복소수 타입에 대한 baddbmm/addbmm 확장 등이 추가되었습니다.
> This release includes support for error reporting from MPS backend as well as continuous expansion of operator coverage, that includes new distributions functions (log\_normal, cauchy, geometric), operator migration (erfcx, grid\_sampler\_2d supports for all operation mode), extended baddbmm/addbmm for integer and complex types.

비동기 오류 보고 기능은 GPU 인덱싱 연산 중 발생하는 범위 초과 접근 시도를 감지할 수 있게 합니다. 예를 들면:
> Asynchronous error reporting enables detection of out-of-bounds access attempts that occur during GPU indexing operations, for example:

```python
import torch
x = torch.rand(10, 1, 10, device='mps')
y = x[:, [1]]
torch.mps.synchronize()  # 인덱스 범위 초과 오류 발생 / will raise index out of bounds error
```

**RNN/LSTM GPU 내보내기(Export) 지원 / RNN/LSTM GPU Export Support**

RNN 모듈(LSTM, GRU 등)을 이제 GPU에서 내보낼 수 있으며, 동적 형상(dynamic shapes)을 사용한 LSTM 추적(tracing)도 지원됩니다. 이를 통해 torch.export를 사용하여 프로덕션 추론을 위해 배포할 수 있는 모델 유형이 크게 확장되었습니다. GRU API는 변경되지 않았으며, 새로운 API는 LSTM에 적용됩니다.
> RNN modules (LSTM, GRU, etc.) can now be exported on GPUs, and tracing LSTM with dynamic shapes is now supported. This significantly expands the model types that can be deployed using torch.export for production inference. GRU API is unchanged; the new API is LSTM.

**ROCm 디바이스 측 어설션 및 TopK 최적화 / ROCm Device-Side Assertions & TopK Optimizations**

ROCm에서 디바이스 측 어설션(device-side assertions)을 지원하여 디버깅이 용이해졌으며, 공유 메모리에 데이터를 캐싱하는 방식으로 TopK 연산자 최적화와 기수 선택(radix select) 개선이 이루어졌습니다. AMD GPU에서의 개발자 경험과 성능이 모두 향상됩니다.
> Added support for device-side assertions on ROCm for better debugging, plus significant TopK operator optimizations and radix select improvements by caching data on shared memory. Improves both developer experience and performance on AMD GPUs.

**Intel GPU에서 실행을 최적화하는 XPUGraph 지원 / XPUGraph support to optimize execution on Intel GPUs**

XPUGraph를 사용하면 Intel GPU에서 일련의 XPU 연산을 런타임 실행 그래프로 캡처하고 여러 번 재생할 수 있습니다. 이를 통해 커널 실행 및 Python 런타임 오버헤드와 같은 CPU 오버헤드를 줄여 Intel GPU에서의 워크로드 성능을 향상시킵니다. 사용 방법은 [API 문서](https://docs.pytorch.org/docs/2.11/xpu.html#graphs)를 참고하세요.
> XPUGraph allows users to capture a sequence of XPU operations into a runtime execution graph on Intel GPUs and replay it multiple times. This reduces CPU overhead, such as kernel launch and Python runtime overhead, improving workload performance on Intel GPUs. See [API Doc](https://docs.pytorch.org/docs/2.11/xpu.html#graphs) for usage details.

**CPU에서 OpenBLAS를 통한 FP16 반정밀도 GEMM / FP16 Half-Precision GEMM On CPU Via OpenBLAS**

CPU에서 OpenBLAS를 통한 FP16 반정밀도 GEMM 지원을 추가하여, CPU 기반 배포 환경에서 더 빠른 FP16 추론이 가능합니다. 이는 엣지 디바이스 및 CPU 전용 추론 시나리오에 유용합니다.
> Added FP16 half-precision GEMM support via OpenBLAS on CPU, providing faster FP16 inference for CPU-based deployments. This is valuable for edge devices and CPU-only inference scenarios.

## 기능 외 업데이트 / Non-Feature Updates

**CUDA 버전 / CUDA version**

이번 릴리즈부터 CUDA 13이 x86\_64 및 ARM 플랫폼 모두에서 기본 설치 버전이 됩니다. 다른 빌드가 필요한 사용자는 https://download.pytorch.org/whl 의 해당 하위 폴더에서 CPU 전용 버전과 CUDA 12.8 빌드에 계속 접근할 수 있습니다.
> Starting with this release, CUDA 13 is now the default version installed for both x86\_64 and ARM platforms. Users who need an alternative build can still access the CPU-only version as well as CUDA 12.8 builds from the respective https://download.pytorch.org/whl subfolders.

**TorchScript 지원 중단(Deprecated) / Torchscript is now Deprecated**

TorchScript는 2.10에서 지원 중단(deprecated)되었으며, jit trace 및 script API를 대체하려면 [torch.export](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html)를, 임베디드 런타임을 대체하려면 [ExecuTorch](https://docs.pytorch.org/executorch/stable/index.html)를 사용해야 합니다. 자세한 내용은 PTC에서 진행한 [이 발표](https://youtu.be/X2YbbDmCsOI?si=8s6Ue3BKIa_FYUne&t=903)를 참고하세요.
> Torchscript was deprecated in 2.10, and [torch.export](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html) should be used to replace the jit trace and script APIs, and [Executorch](https://docs.pytorch.org/executorch/stable/index.html) should be used to replace the embedded runtime. For more details, see [this talk](https://youtu.be/X2YbbDmCsOI?si=8s6Ue3BKIa_FYUne&t=903) from PTC.

**2026년 릴리즈 주기 / 2026 Release Cadence**

2026년부터 릴리즈 주기가 분기별에서 2개월에 1회로 변경되었습니다. [공개된 릴리즈 일정](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-cadence)을 참고하세요.
> For 2026, the release cadence has been increased to 1 per 2 months, from quarterly. See the [published release schedule](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-cadence).
