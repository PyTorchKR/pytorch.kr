---
layout: blog_detail
title: "TPU 위의 Helion: 이기종 하드웨어 커널 작성을 향하여"
author: Dunfan Lu, Yifei Xu, Jongsok Choi, Ethan Che, Oguz Ulgen, Jason Ansel, Yarong Mu, Theotime Combes, Emilio Cota, Freya Azad
authors:
  - Dunfan Lu
  - Yifei Xu
  - Jongsok Choi
  - Ethan Che
  - Oguz Ulgen
  - Jason Ansel
  - Yarong Mu
  - Theotime Combes
  - Emilio Cota
  - Freya Azad
image: /assets/blog/2026-07-23-helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/hero.png
category: ["pytorch.org", "translation"]
org_title: "Helion on TPU: Towards Hardware Heterogeneous Kernel Authoring"
org_link: https://pytorch.org/blog/helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/
---

## 요약 / TL;DR

[Helion](https://helionlang.com/) 은 성능 이식성(performance portability)을 갖춘 ML 커널을 작성하기 위한 PyTorch의 고수준 DSL입니다. Google과 협력하여 Helion 커널을 [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) 로 컴파일하는 TPU 백엔드를 구축했으며, 이를 통해 PyTorch 친화적인 방식으로 고성능 TPU 커널을 작성할 수 있게 되었습니다. flash attention 워크로드에서 Helion이 생성한 커널은 TPU v7에서 838 TFLOPs(텐서 코어 하나 기준 약 79% MFU)를 달성합니다. 서로 다른 입력 형태(shape)에 대해 Helion은 여러 코드 생성 전략을 오토튜닝(autotuning)하여 최적의 파이프라이닝(pipelining) 방식을 선택하고, TPU가 사용할 수 있는 VMEM과 연산 자원을 최대한 활용합니다.
> [Helion](https://helionlang.com/) is PyTorch’s high-level DSL for writing performance-portable ML kernels. Partnering with Google, we have built a TPU backend that compiles Helion kernels to [Pallas](https://docs.jax.dev/en/latest/pallas/index.html), providing a PyTorch-friendly way to author performant TPU kernels. On a flash attention workload, the Helion-generated kernel achieves 838 TFLOPs (~79% MFU of one tensor core) on TPU v7. On different input shapes, Helion autotunes over different code-generation strategies to select the optimal pipelining schema, making the most use of TPU’s available VMEM and compute.

## 서론 / Introduction

TPU는 GPU를 보완하는 ML 연산 플랫폼으로서 점점 더 중요해지고 있습니다. Google의 최신 TPU v7(Ironwood)은 NVIDIA B200에 필적하는 성능을 제공하면서도 총소유비용(TCO, total cost of ownership)은 더 낮을 가능성이 있어, 대규모 학습 및 추론 워크로드에 매력적인 선택지가 됩니다. 하지만 전통적으로 TPU 커널을 작성하려면 Pallas에 대한 전문 지식이 필요했습니다. Pallas는 가파른 학습 곡선과 코드 복잡성을 동반하는 저수준(low-level) DSL입니다. [Helion](https://helionlang.com/) 은 이 간극을 메웁니다. ML 커널을 위한 PyTorch의 이식 가능한(portable) DSL로서, Helion은 사용자가 익숙한 PyTorch 스타일의 코드를 작성하면 이를 최적화된 TPU 코드로 컴파일해 줍니다. 오토튜너(autotuner)가 가져다주는 성능 이점과 결합되어, Helion은 TPU 커널 작성을 위한 매력적인 선택지로 발전하고 있습니다. 구체적으로 Helion TPU는 다음 세 가지 주요 사용 사례를 겨냥합니다:
> TPUs are increasingly important as an ML compute platform to complement GPUs. Google’s latest TPU v7 (Ironwood) delivers comparable performance to NVIDIA B200 with a potentially lower total cost of ownership (TCO), making TPUs an appealing option for large-scale training and inference workloads. However, authoring TPU kernels traditionally requires expertise in Pallas, a low-level DSL that comes with a steep learning curve and code complexity. [Helion](https://helionlang.com/) bridges this gap. As PyTorch’s portable DSL for ML kernels, Helion lets users write familiar PyTorch-style code and compiles it to optimized TPU code. Paired with performance wins brought by its autotuner, Helion is evolving towards an attractive option for authoring TPU kernels. Specifically, Helion TPU targets three main use cases:

- **성능이 중요한 사용 사례** — 설정 공간(configuration space)을 탐색하기 위해 오토튜닝이 필요한 경우
- **Pallas 비전문가** — TPU 커널 작성에 빠르게 입문하고자 하는 사용자
- **크로스 하드웨어(cross-hardware) 사용자** — TPU와 GPU에 걸쳐 동일한 커널 집합을 유지하고 싶어 하는 사용자

> - **Performance-critical use cases** where autotuning is required to explore the configuration space
> - **Non-Pallas experts** hoping to onboard TPU kernel authoring quickly
> - **Cross-hardware users** who prefer to maintain the same set of kernels across TPU and GPU

이번 글에서는 먼저 GPU와 비교하여 TPU의 하드웨어 특성과 프로그래밍 모델을 간략히 살펴본 뒤, Helion이 서로 다른 입력 형태에 대해 이상적인 파이프라이닝 특성을 갖춘 고성능 Pallas 코드를 어떻게 생성하는지 보여줍니다.
> This article starts with a brief overview of TPU’s hardware features and programming models as compared to GPUs, and then demonstrates how Helion generates performant Pallas code with ideal pipelining characteristics for different input shapes.

## TPU 기초 / TPU Primer

TPU는 머신러닝 워크로드에 특화되어 설계·최적화된 고도로 전문화된 가속기입니다. TPU의 [아키텍처](https://jax-ml.github.io/scaling-book/tpus/) 와 프로그래밍 모델은 GPU와 상당히 다릅니다. 가장 두드러진 차이는 TPU가 넓은 벡터 레지스터(vector register)와 연산 유닛을 갖춘 순차(sequential) 머신이라는 점입니다. 이는 대규모 병렬 실행(CUDA 코어)과 특화된 텐서 유닛(텐서 코어)을 함께 활용해 성능을 얻는 GPU와 대조됩니다.
> TPUs are highly specialized accelerators designed and optimized specifically for machine learning workloads. The [architecture](https://jax-ml.github.io/scaling-book/tpus/) and programming model of TPUs differ significantly from GPUs. The most prominent difference is that a TPU is a sequential machine featuring wide vector registers and compute units. This contrasts with GPUs, which achieve performance via both massively parallel execution (CUDA cores) and specialized tensor units (tensor cores).

| | TPU (Pallas) | GPU (CUDA) |
| --- | --- | --- |
| **스레딩 / Threading** | 순차 실행<br>소수의 큰 워커 | 병렬 SIMT (+ 텐서 코어)<br>다수의 작은 워커 |
| **메모리 계층 / Memory Hierarchy** | 명시적 메모리 공간(지속 메모리(persistent) vs 스크래치패드 메모리(scratchpad)),<br>파이프라이닝을 위해 비동기 메모리 복사 필요 | 암묵적 캐시,<br>하드웨어 관리 |

그 결과 TPU는 커널 작성자가 깊이 이해해야 하는 메모리 계층을 갖습니다. 그래야 작성한 커널이 오프칩(off-chip) HBM에서 빠른 온칩(on-chip) VMEM으로 데이터를 언제 어떻게 불러올지 조율할 수 있기 때문입니다. 고성능 Pallas 커널이라면 이러한 HBM↔VMEM 메모리 전송을, 행렬 유닛(MXU)과 벡터 연산 유닛에서 일어나는 부동소수점 연산과 겹쳐서 수행합니다.
> As a result, TPUs feature a memory hierarchy that kernel authors must deeply understand, so that the kernels they write can orchestrate when and how data is loaded from the off-chip HBM to the fast on-chip VMEM. A performant Pallas kernel would overlap these HBM<>VMEM memory transfers with floating point computation happening in the matrix (MXU) and vector compute units.

![TPU의 메모리 계층 / TPU memory hierarchy](/assets/blog/2026-07-23-helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/tpu-memory-hierarchy.png){:style="width:100%"}

이러한 아키텍처 차이에도 불구하고, 현재 세대의 TPU와 GPU는 순수 성능(raw performance) 면에서 매우 비슷합니다. TPU7x와 NVIDIA B200은 BF16 연산 TFLOPS와 HBM 대역폭이 매우 비슷한데, 이 둘은 현대 ML 워크로드에서 가장 중요한 두 가지 하드웨어 지표입니다.
> Despite the architectural differences, current-generation TPUs and GPUs are highly comparable in raw performance. TPU7x and NVIDIA B200 have very similar BF16 compute TFLOPS and HBM bandwidth — the two most important hardware metrics for modern ML workloads.

## Helion의 Pallas 코드 생성 / Helion’s Pallas Codegen

TPU에서 최대 성능을 끌어내기 위해, Helion의 Pallas 코드 생성(codegen)은 **소프트웨어 파이프라이닝(software pipelining)** 을 극대화하여 메모리 전송과 연산이 가능한 한 많이 겹치도록 하는 것을 목표로 합니다. 이 절에서는 파이프라이닝된 커널을 생성하기 위한 Helion의 세 갈래 전략을 설명합니다:
> To extract maximum performance out of a TPU, Helion’s Pallas codegen aims to maximize **software pipelining**, ensuring that memory transfers and computation overlap as much as possible. This section illustrates Helion’s three-fold strategy for generating pipelined kernels:

- 외부 루프(outer loop): Pallas가 제공하는 파이프라이닝된 디바이스 호출(`pallas_call`/`emit_pipeline`)
- 내부 루프(inner loop): 다음 두 가지 사이에서 오토튜닝:
    - Pallas가 제공하는 파이프라이닝된 디바이스 측 루프(`emit_pipeline`)
    - 가능하다면 모든 값을 VMEM으로 미리 가져오기(`unroll`)
- 오토튜닝된 파이프라인 버퍼 크기

> - Outer loop: pallas-provided pipelined device invocation (`pallas_call`/`emit_pipeline`)
> - Inner loop: autotuned between:
>     - pallas-provided pipelined device-side loop (`emit_pipeline`)
>     - Pre-fetching all values into VMEM, if possible (`unroll`)
> - Auto-tuned pipeline buffer sizes

### 예시: add / Example: add

간단한 예시로, 두 Tensor를 더하는 다음 Helion 커널을 살펴봅시다.
> As a simple example, consider the following helion kernel for adding two tensors.

```python
@helion.kernel
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out
```

Helion 컴파일러는 이를 두 개의 함수로 변환합니다. 하나는 입력을 타일로 나누고 디바이스 함수를 파이프라이닝된 방식으로 호출하는 호스트 측 런처(launcher)이고, 다른 하나는 VMEM에 상주하는 타일에 대해 동작하는 디바이스 함수입니다:
> The Helion compiler translates this into two functions: a host-side launcher that tiles the input and invokes the device function in a pipelined fashion, and a device function that operates on VMEM-resident tiles:

```python
def _helion_add(x, y, out):
    out[:] = x[:] + y[:]

def add(x: torch.Tensor, y: torch.Tensor):
    _BLOCK_SIZE_0 = <autotuner-selected value>
    out = torch.empty(...)
    out = launcher( # wraps around pallas_call
        _helion_add, 
        ((x.shape[0] + _BLOCK_SIZE_0 - 1) // _BLOCK_SIZE_0,), # grid size
        x, y, out, 
        _block_spec_info=[_BLOCK_SIZE_0, ...], ...
    )
    return out
```

생성된 코드에서:
> Within the generated code:

- Helion 소스의 `hl.tile` 루프는 호스트 측의 그리드(grid)가 됩니다. 런처(`pallas_call`을 감쌈)는 타일마다 한 번씩 `_helion_add`를 호출하며, 각 호출은 자동으로 파이프라이닝됩니다 — 한 타일이 계산되는 동안 다음 타일의 데이터가 HBM에서 VMEM으로 로드됩니다.
- 디바이스 함수 `_helion_add`는 단순합니다. HBM 포인터가 아니라 VMEM 참조를 받으므로, 커널 본문은 단순한 덧셈입니다.
- `_BLOCK_SIZE_0`(타일/버퍼 크기)은 오토튜너가 선택합니다. 오토튜너는 여러 크기를 탐색하여 대상 하드웨어에서 메모리 전송과 연산 사이의 최적 중첩(overlap)을 찾습니다.

> - The `hl.tile` loop in the Helion source becomes a grid on the host side. The launcher (wrapping `pallas_call`) invokes `_helion_add` once per tile, with each invocation automatically pipelined — while one tile is being computed, the next tile’s data is being loaded from HBM into VMEM.
> - The device function `_helion_add` is simple: it receives VMEM references (not HBM pointers), so the kernel body is a simple addition.
> - `_BLOCK_SIZE_0` (the tile/buffer size) is selected by the autotuner, which explores different sizes to find the best overlap between memory transfers and compute for the target hardware.

그 결과 아래 그림과 같이 파이프라이닝된 실행이 이루어집니다.
> This results in a pipelined execution as illustrated below.

![add 커널의 파이프라이닝된 실행 / Pipelined execution of the add kernel](/assets/blog/2026-07-23-helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/add-pipeline.png){:style="width:100%"}

### 예시: Flash Attention / Example: Flash Attention

어텐션(attention)은 현대 언어 모델의 핵심 연산 중 하나입니다. 실제 프로덕션 구현은 “Flash Attention” 패턴을 따르는데, 이는 전체 S×S 어텐션 행렬을 구체화(materialize)하지 않기 위해 어텐션을 타일 단위로 계산하는 메모리 효율적 기법입니다. Helion에서 flash attention 커널의 구조는 아래와 같습니다:
> Attention is one of the key operations in modern language models. Production implementations follow the “Flash Attention” pattern – a memory-efficient technique that computes attention in tiles to avoid materializing the full S×S attention matrix. The structure of a flash attention kernel in Helion is illustrated below:

```python
B, H, S, D = 8, 32, 8192, 256 # batch, head, sequence length, head dimension
@helion.kernel
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = torch.empty(...)
    for tile_b, tile_q in hl.tile(B * H, S):
        this_q = q[tile_b, tile_q, :]
        acc = ...
        for tile_kv in hl.tile(S):
            this_k = k[tile_b, tile_kv, :]
            this_v = v[tile_b, tile_kv, :]
            <qk matmul, online softmax, v matmul, update acc>
        out[tile_b, tile_q] = acc
    return out
```

앞서 다룬 “add” 예시와 비교하면, [flash attention 커널](https://github.com/pytorch/helion/blob/main/examples/attention.py) 에는 K와 V 시퀀스 전체에 걸쳐 타일 단위 접근을 수행하는 내부 루프가 추가로 있습니다. 이 내부 루프에서 메모리와 연산을 어떻게 파이프라이닝하는지가 이 커널의 성능을 좌우하는 핵심입니다.
> Compared to the “add” example discussed previously, the [flash attention kernel](https://github.com/pytorch/helion/blob/main/examples/attention.py) contains an additional inner loop which performs tiled accesses across the entire K and V sequences. How we pipeline the memory and compute within the inner loop is key to the performance of this kernel.

Helion에서 컴파일러는 이 커널을 Pallas로 변환하는 두 가지 서로 다른 전략을 놓고 오토튜닝합니다. 이는 `pallas_loop_type` 오토튜너 설정으로 제어됩니다.
> In Helion, the compiler autotunes over two different strategies for translating this kernel to Pallas. This is keyed on the `pallas_loop_type` autotuner config.

기본값인 `pallas_loop_type == emit_pipeline` 옵션에서는, 호스트 측 로직이 `pallas_call`로 디바이스 함수 호출을 파이프라이닝하는 것과 비슷하게, Helion이 Pallas의 디바이스 측 `emit_pipeline` API를 사용해 내부 루프 본문 함수를 파이프라이닝합니다:
> With the default `pallas_loop_type == emit_pipeline` option, Helion relies on Pallas’ device-side `emit_pipeline` API to pipeline an inner loop body function, similarly to how the host-side logic uses `pallas_call` to pipeline the device function invocation:

```python
def _helion_attention(q_VMEM, k_HBM, v_HBM, out_VMEM):
    acc = ...
    this_q = q_VMEM[:, :, :]
    def _inner_pipeline_body(k_VMEM, v_VMEM):
        this_k = k_VMEM[:, :, :]
        this_v = v_VMEM[:, :, :]
        <matmul, online softmax, matmul, update acc>
    pallas.tpu.emit_pipeline(_inner_pipeline_body, k_HBM, v_HBM, _block_spec_info=[BLOCK_SIZE_KV ,...], ... )
    out_VMEM = acc

def attention(q: torch.Tensor, k: torch.Tensor, v:torch.Tensor):
    out = torch.empty(...)
    out = launcher( # wraps around pallas_call
        _helion_attention, 
        q, k, v, out, 
        _block_spec_info=[BLOCK_SIZE_Q ,...], ...
    )
    return out
```

생성된 이 커널은 중첩된 파이프라인 구조를 따릅니다:
> This generated kernel follows a nested pipeline structure:

- (외부 파이프라인) 호스트는 `pallas_call`을 사용해 `_helion_attention`을 호출합니다. q의 HBM 참조는 타일로 나뉘고, `_helion_attention`의 각 호출은 q의 VMEM 타일을 받습니다. k와 v의 경우 `_helion_attention`은 HBM 참조를 직접 받습니다.
- (내부 파이프라인) `_helion_attention` 내부에서 디바이스는 `emit_pipeline`을 사용해 `_inner_pipeline_body`를 호출하며, 이 함수는 k와 v의 VMEM 타일을 받습니다.

> - (Outer pipeline) The host uses `pallas_call` to invoke `_helion_attention`. The HBM reference of q is tiled, and each invocation of `_helion_attention` receives a VMEM tile of q. For k and v, `_helion_attention` receives HBM references directly.
> - (Inner Pipeline) Within `_helion_attention`, the device uses `emit_pipeline` to invoke `_inner_pipeline_body`, which receives VMEM tiles of k and v.

그 결과 아래 그림과 같이 파이프라이닝된 실행이 이루어집니다:
> This results in a pipelined execution as illustrated in this image:

![emit_pipeline 전략의 파이프라이닝된 실행 / Pipelined execution with the emit_pipeline strategy](/assets/blog/2026-07-23-helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/attention-emit-pipeline.png){:style="width:100%"}

이 파이프라인에서 명백한 비효율 지점 하나는 연산 유닛에 버블(bubble)이 생긴다는 것입니다. 새로운 Q 타일마다 0번째 KV 타일을 가져오는 동안에는 연산 유닛이 할 일이 없기 때문입니다. 이는 새로운 Q 타일마다 KV 타일을 HBM에서 VMEM으로 다시 로드하기 때문입니다.
> One obvious point of inefficiency in this pipeline is that there are bubbles in the compute units – for every new Q tile, while we fetch the 0th KV tile, there is no work available for the compute units. This comes down to the fact that we are re-loading the KV tiles from HBM to VMEM for every new Q tile.

Helion은 이러한 버블을 피하는 대안으로 `pallas_loop_type == unroll` 설정을 제공합니다. `unroll`을 사용하면 내부 for 루프를 단순한 Python for 루프로 변환합니다:
> Helion offers an alternative `pallas_loop_type == unroll` config which avoids this bubbling. With `unroll`, we translate the inner for loop into a simple Python for loop:

```python
def _helion_attention(q_VMEM, k_VMEM_FULL, v_VMEM_FULL, out_VMEM):
    acc = ...
    this_q = q_VMEM[:, :, :]
    for offset in range(0, k_VMEM_FULL.size(1) , BLOCK_SIZE_KV):
        this_k = k_VMEM_FULL[:, pallas.dslice(offset, BLOCK_SIZE_KV), :]
        this_v = v_VMEM_FULL[:, pallas.dslice(offset, BLOCK_SIZE_KV), :]
        <matmul, online softmax, matmul, update acc>
    out_VMEM = acc

def attention(q: torch.Tensor, k: torch.Tensor, v:torch.Tensor):
    out = torch.empty(...)
    out = launcher( # wraps around pallas_call
        _helion_attention, 
        q, k, v, out, 
        _block_spec_info=[BLOCK_SIZE_Q, None, None], ...
    )
    return out
```

(“unroll”이라는 이름은 Pallas 디바이스 함수가 JAX의 JIT에 의해 트레이싱(tracing)된다는 사실을 반영합니다. Python for 루프는 트레이싱 시점에 사실상 펼쳐져(unroll) 평탄한 연산 시퀀스가 됩니다.)
> (The name “unroll” reflects the fact that Pallas device functions are traced by JAX’s JIT — the Python for loop is effectively unrolled at trace time into a flat sequence of operations.)

이 버전의 생성된 커널에서는:
> In this version of the generated kernel:

- **K와 V를 통째로 미리 가져옵니다:** 호스트는 K와 V의 블록 명세(block spec)로 `None`을 전달하여 `pallas_call`이 이들을 VMEM에 통째로 로드하도록 지시합니다. 이 전체 VMEM 참조는 모든 디바이스 함수 호출에 걸쳐 유지됩니다.
- **내부 루프는 로컬에서 슬라이싱합니다:** 각 반복은 `pallas.dslice`를 사용해 이미 상주 중인 VMEM 버퍼에서 해당 KV 타일을 선택합니다. 내부 루프 동안에는 HBM 트래픽이 발생하지 않습니다.

> - **K and V are pre-fetched in full:** The host passes `None` as the block spec for K and V, instructing `pallas_call` to load them entirely into VMEM. The full VMEM references persist across all device function invocations.
> - **The inner loop slices locally:** Each iteration uses `pallas.dslice` to select the relevant KV tile from the already-resident VMEM buffer. No HBM traffic occurs during the inner loop.

그 결과 아래 그림과 같이 다른 파이프라이닝 방식이 됩니다:
> This results in a different pipelining scheme as illustrated below:

![unroll 전략의 파이프라이닝 방식 / Pipelining scheme with the unroll strategy](/assets/blog/2026-07-23-helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/attention-unroll.png){:style="width:100%"}

이 방식에서는 더 이상 연산 파이프라인에 버블이 생기지 않습니다. 트레이드오프는 K와 V 시퀀스 전체가 존재해야 하므로 더 많은 VMEM을 사용한다는 점입니다. 즉, 더 성능이 좋긴 하지만 이 변환이 항상 가능한 것은 아닙니다. VMEM 사용량은 (타일 크기가 아니라) 입력 시퀀스 길이에 선형으로 비례하므로, 시퀀스가 길어지면 감당하기 어렵습니다. 이 두 전략 사이의 성능 차이는 상당합니다. 아래 표는 B=8, H=32, D=256 워크로드에서의 결과를 보여줍니다:
> In this workflow, there are no longer bubbles in the compute pipeline. The trade-off is that this requires more VMEM usage, as the entire K and V sequences need to be present. This means that although more performant, this translation isn’t always possible. The VMEM usage is linear with respect to the input sequence lengths (as opposed to tile size), which is prohibitive with longer sequences. The performance difference between these strategies is significant — the table below shows results on workloads with B=8, H=32, D=256:

| | S = 8k | S = 32k |
| --- | --- | --- |
| **emit_pipeline TFLOPs** | 653 | 695 |
| **unroll TFLOPs** | 892 | OOM |

Helion의 장점은 오토튜닝을 통해 최적의 오토튜너 설정을 선택하는 능력에 있습니다. 덕분에 시퀀스가 작을 때는 사용 가능한 VMEM을 활용하여 연산 버블이 없는 파이프라이닝된 코드를 생성합니다. 시퀀스가 길 때는 임의의 컨텍스트 길이까지 확장되는 `emit_pipeline`으로 대체(fallback)합니다. 아래 그래프는 다양한 시퀀스 길이에서 이 어텐션 커널의 성능을 다른 여러 Pallas 어텐션 구현과 비교하여 나타낸 것입니다:
> The benefit of Helion lies in its ability to autotune and select the best autotuner config. So that with smaller sequences, it makes use of the VMEM available and generates pipelined code with no compute bubbles. For longer sequences, it falls back to `emit_pipeline` which scales to arbitrary context lengths. The following graph plots the performance of this attention kernel compared to various other Pallas attention implementations, on varying sequence lengths:

![시퀀스 길이별 어텐션 커널 성능 비교 / Attention kernel performance across sequence lengths](/assets/blog/2026-07-23-helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/attention-perf.png){:style="width:100%"}

입력 길이에 따라 서로 다른 루프·파이프라이닝 전략을 코드 생성할 수 있는 오토튜너의 능력이야말로, Tokamax처럼 고도로 최적화된 구현과 비교해서도 Helion이 우위를 갖게 하는 요소입니다.
> The autotuner’s ability codegen different loop and pipelining strategies depending on the input length is what gives Helion its edge even when compared to highly optimized implementations such as Tokamax.

## 더 폭넓은 커널 벤치마크 / Broader Kernel Benchmarks

다양한 커널에 대해 Helion을 벤치마크하며, 그 결과는 [대시보드](https://helionlang.com/dashboard/) 에서 추적합니다. 아래 표는 여러 종류의 커널에 대해 Helion을 TorchTPU eager 및 (XLA를 사용하는) `torch.compile`과 비교합니다. Helion은 eager 대비 기하 평균 1.55배, 컴파일 버전 대비 1.12배의 속도 향상을 보입니다.
> We benchmark Helion across a variety of kernels, tracked on our [dashboard](https://helionlang.com/dashboard/). The table below compares Helion against TorchTPU eager and `torch.compile` (using XLA) across a range of different kernels. Helion shows a geometric average speed-up of 1.55x compared to eager, and 1.12x compared to compiled.

| 커널 / kernel | 형태 / shape | torch_tpu eager (ms) | torch.compile(tpu) (ms) | Helion (ms) | Helion vs torch_tpu eager | Helion vs torch.compile |
| --- | --- | --- | --- | --- | --- | --- |
| `attention` | [8,32,8192,256] | 87.77 | 88.28 | 19.72 | 4.45배 | 4.48배 |
| `softmax` | [65536,2560] | 0.712 | 0.743 | 0.477 | 1.49배 | 1.56배 |
| `batch_softmax` | [64,2048,4096] | 1.888 | 1.373 | 0.982 | 1.92배 | 1.40배 |
| `softmax_two_pass` | [8192,8192] | 0.386 | 0.417 | 0.334 | 1.16배 | 1.25배 |
| `bmm` | [64,2048,2048,2048] | 3.211 | 1.860 | 1.527 | 2.10배 | 1.22배 |
| `rms_norm-bwd` | [8192,8192] | 1.792 | 0.789 | 0.661 | 2.71배 | 1.19배 |
| `epilogue_subtiling` | [4096,4096,4096] | 0.850 | 0.462 | 0.417 | 2.04배 | 1.11배 |
| `matmul_layernorm` | [4096,4096,4096] | 0.535 | 0.523 | 0.489 | 1.10배 | 1.07배 |
| `welford` | [524288,512] | 1.330 | 1.357 | 1.316 | 1.01배 | 1.03배 |
| `swiglu` | [16,16384,4096] | 3.510 | 2.244 | 2.295 | 1.53배 | 0.98배 |
| `matmul` | [8192,8192,8192] | 1.552 | 1.527 | 1.597 | 0.97배 | 0.96배 |
| `geglu` | [16,8192,8192] | 3.779 | 2.240 | 2.424 | 1.56배 | 0.92배 |
| `cross_entropy` | [128,2048] | 0.363 | 0.264 | 0.320 | 1.13배 | 0.82배 |
| `broadcast_matmul` | [64,2048,2048,2048] | 1.817 | 1.440 | 1.806 | 1.01배 | 0.80배 |
| `layer_norm` | [16384,16384] | 1.253 | 0.779 | 1.126 | 1.11배 | 0.69배 |
| `rms_norm` | [8192,8192] | 1.194 | 0.419 | 0.617 | 1.94배 | 0.68배 |

Helion은 XLA가 자동으로 발견하기 어려운 융합(fusion)이나 최적화 패턴을 사용하는 커널에서 가장 큰 이득을 보이며, flash attention이 대표적인 예입니다. `matmul`이나 `layer_norm` 같은 더 표준적인 연산에서는 XLA 컴파일러가 이미 고품질 코드를 생성하므로, Helion도 비슷한 성능을 보입니다.
> Helion shows the largest gains on kernels that employ fusion or optimization patterns that are difficult for XLA to discover automatically — flash attention is a prominent example. For the more standard operations like `matmul` and `layer_norm`, XLA’s compiler already produces high-quality code, and Helion performs comparably.

![커널별 성능 비교 / Per-kernel performance comparison](/assets/blog/2026-07-23-helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/kernel-benchmarks.png){:style="width:100%"}

## 다음 단계 / What’s Next

TPU 위의 Helion은 활발히 개발 중입니다. 현재 작업 중인 항목을 전부는 아니지만 일부 나열하면 다음과 같습니다:
> Helion on TPU is under active development. Here’s a non-exhaustive list of things we are working on:

- 커널 커버리지 확대: 더 많은 Helion 예제를 TPU에서 동작하게 하기
- 추가적인 성능 개선
- 재기드(jagged) 및 희소(sparse) 연산에 대한 더 나은 지원
- 분산 TPU 컴퓨팅 지원

> - Expand kernel coverage: Get more Helion examples working on TPU
> - Further performance improvements
> - Better support for jagged and sparse operations
> - Support for distributed TPU computing

## 시작하기 / Getting Started

Helion은 오픈소스이며 GitHub에서 이용할 수 있습니다. TPU 백엔드는 [TorchTPU](https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/) 에 의존하며, TorchTPU는 올해 말에 공개될 예정입니다. 공개되면 TPU 위의 Helion을 꼭 사용해 보고 피드백을 공유해 주시기 바랍니다. 관련 자료:
> Helion is open source and available on GitHub. Its TPU backend has a dependency on [TorchTPU](https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/), which is expected to be released publicly later this year. When it does, we encourage you to try-out Helion on TPU and share your feedback. Resources:

- [Helion GitHub 저장소](https://github.com/pytorch/helion)
- [Helion 문서](https://helionlang.com/)
- [Helion TPU 예제](https://github.com/pytorch/helion/tree/main/examples)
- [성능 대시보드](https://helionlang.com/dashboard/)

> - [Helion GitHub Repository](https://github.com/pytorch/helion)
> - [Helion Documentation](https://helionlang.com/)
> - [Helion TPU Examples](https://github.com/pytorch/helion/tree/main/examples)
> - [Performance Dashboard](https://helionlang.com/dashboard/)

## 감사의 글 / Acknowledgements

이 프로젝트는 동료들의 값진 협업과 기술적 통찰 덕분에 가능했습니다. 특히 Google의 Joe Pamer, Robert Hundt, Claudio Basile, Adam Paszke, 그리고 Meta의 Jana van Greunen, Gregory Chanan, Peng Wu, Zongwei Zhou에게 이 결실을 맺도록 피드백과 지원을 준 데 대해 깊이 감사드립니다.
> This project was made possible through the invaluable collaboration and technical insights of our peers. A special thank you to Joe Pamer, Robert Hundt, Claudio Basile and Adam Paszke at Google, as well as Jana van Greunen, Gregory Chanan, Peng Wu, and Zongwei Zhou at Meta, for their feedback and support in bringing this to fruition.
