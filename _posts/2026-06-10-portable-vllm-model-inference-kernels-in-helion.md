---
layout: blog_detail
title: "Helion으로 작성한 이식 가능한 vLLM 모델 추론 커널"
author: Sean Chen (Red Hat) and Yanan Cao (PyTorch, Meta Platforms)
category: ["pytorch.org", "translation"]
org_title: "Portable vLLM Model Inference Kernels in Helion"
org_link: https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/
---

### TL;DR

*Qwen3 모델을 사용하는 FP8 추론(inference)을 위해 Helion 커널을 vLLM에 통합하고, NVIDIA H100과 B200 GPU에서 평가했습니다. 실험 결과, Helion은 양자화(quantization), 정규화(normalization), 그리고 융합이 많은(fusion-heavy) 추론 커널 다수에서 성능을 끌어올리면서도, 융합형 GPU 커널을 개발하는 데 생산적인 PyTorch 네이티브 워크플로우를 제공한다는 점이 드러났습니다. 엔드투엔드(end-to-end) 벤치마크에서는 여러 서빙 시나리오에 걸쳐 처리량(throughput) 향상이 확인되었으며, Blackwell GPU에서의 GEMM 성능을 위한 추가 최적화 작업이 진행 중입니다.*
> *Helion kernels were integrated into vLLM for FP8 inference using Qwen3 models and evaluated across NVIDIA H100 and B200 GPUs. The experiments show that Helion provides a productive PyTorch-native workflow for developing fused GPU kernels while delivering performance improvements for many quantization, normalization, and fusion-heavy inference kernels. End-to-end benchmarks demonstrated throughput gains across multiple serving scenarios, with additional optimization work underway for GEMM performance on Blackwell GPUs.*

## vLLM과 Helion에 대한 간략한 배경 / Brief Background on vLLM and Helion

[vLLM](https://docs.vllm.ai/en/latest/)은 대규모 언어 모델(LLM)을 위한 고성능 추론·서빙 프레임워크입니다. 강력한 처리량 성능, 효율적인 KV 캐시 관리, 연속 배칭(continuous batching) 아키텍처, 그리고 추측 디코딩(speculative decoding)·양자화·분산 서빙 같은 고급 추론 기능 지원 덕분에 프로덕션 LLM 서빙에 널리 쓰입니다. 내부적으로 vLLM은 다양한 하드웨어 플랫폼에서 높은 추론 효율을 달성하기 위해 커스텀 GPU 커널, TorchInductor 융합, 그리고 CUTLASS·DeepGEMM 같은 최적화된 GEMM 백엔드에 크게 의존합니다.
> [vLLM](https://docs.vllm.ai/en/latest/) is a high-performance inference and serving framework for large language models (LLMs). It is widely used for production LLM serving due to its strong throughput performance, efficient KV-cache management, continuous batching architecture, and support for advanced inference features such as speculative decoding, quantization, and distributed serving. Internally, vLLM relies heavily on custom GPU kernels, TorchInductor fusion, and optimized GEMM backends such as CUTLASS and DeepGEMM to achieve high inference efficiency across different hardware platforms.

[Helion](https://helionlang.com/index.html)은 타일 프로그래밍(tile-programming) 모델을 사용해 고성능 커널을 작성하도록 설계된, PyTorch 네이티브이면서 하드웨어에 구애받지 않는 커널 DSL입니다. 저수준 CUDA 프로그래밍과 달리, Helion은 메모리 레이아웃·타일링 전략·커널 스케줄링에 대한 저수준 제어권을 그대로 노출하면서도 더 자연스러운 PyTorch 문법 중심의 개발 경험을 제공합니다. 타일을 다루는 PyTorch라고 생각하면 됩니다. PyTorch나 Triton을 안다면 Helion의 대부분을 이미 아는 셈입니다. 매끄러운 작성 경험 외에 Helion의 또 다른 강점은 강력한 사전 컴파일(ahead-of-time, AOT) 자동 튜닝(autotuning) 인프라로, 방대한 커널 구성 공간을 탐색하여 특정 워크로드와 하드웨어 타깃에 최적화된 구현을 자동으로 선택할 수 있습니다.
> [Helion](https://helionlang.com/index.html) is a PyTorch-native hardware agnostic kernel DSL designed for writing high-performance kernels using a tile-programming model. Unlike lower-level CUDA programming, Helion provides a more natural PyTorch-syntax-centric development experience while still exposing low-level control over memory layout, tiling strategy, and kernel scheduling. You can think of it as PyTorch with tiles. If you know PyTorch or Triton, you already know most of Helion. Other than smooth authoring experience, another strength of Helion is its powerful ahead-of-time (AOT) autotuning infrastructure, which can explore a large kernel configuration space and automatically select optimized implementations for specific workloads and hardware targets.

## Helion 커널을 사용한 vLLM 모델 추론 / vLLM Model Inference with Helion Kernels

먼저 FP8 활성화(activation) 양자화를 켠 상태에서 Qwen3 모델 패밀리를 사용해 텐서 병렬화 없는(tensor-parallel-free) 추론에 집중했습니다.
> We began by focusing on tensor-parallel-free inference using the Qwen3 model family with FP8 activation quantization enabled.

목표는 Helion 커널이 기존 vLLM 구현 대비 추론 성능을 개선할 수 있는지 평가하는 것이었습니다.
> Our goal was to evaluate whether Helion kernels can improve inference performance compared to the existing vLLM implementations.

이 실험에서는 양자화 추론에 관여하는 거의 모든 순전파(forward-pass) 커널을 Helion 구현으로 교체하고, 커널 수준과 엔드투엔드 서빙 수준 모두에서 벤치마크했습니다.
> For this experiment, we replaced nearly all forward-pass kernels involved in quantized inference with Helion implementations and benchmarked them at both kernel level and end-to-end serving level.

### vLLM 순전파 융합 패턴 / vLLM Forward Pass Fusion Pattern

Qwen3 모델의 경우, vLLM에서 융합되지 않은 순전파는 다음 커널 시퀀스를 실행합니다:
> For Qwen3 models, the unfused forward pass in vLLM executes the following sequence of kernels:

1. input_norm
2. fp8_quant
3. scaled_mm (qkv_proj)
4. split_qkv
5. q_norm
6. k_norm
7. rope
8. attention
9. fp8_quant
10. scaled_mm (out_proj)
11. post_attention_norm
12. fp8_quant
13. scaled_mm (gate_up)
14. silu_and_mul
15. fp8_quant
16. scaled_mm (down_proj)

**동적 토큰별 활성화 양자화 / Dynamic Per-Token Activation Quantization**

torch.compile과 TorchInductor 융합 패스가 적용되고 나면, 실행 패턴은 다음과 같이 바뀝니다:
> After torch.compile and TorchInductor fusion passes are applied, the execution pattern becomes:

1. rms_norm + fp8_quant
2. scaled_mm (qkv_proj)
3. split_qkv + q_norm + k_norm
4. rope
5. attention
6. fp8_quant
7. scaled_mm (out_proj)
8. rms_norm + fp8_quant
9. scaled_mm (gate_up)
10. silu_and_mul + fp8_quant
11. scaled_mm (down_proj)

*역주: 원문 3번 항목은 `split_qkv + q_norm + v_norm`으로 표기되어 있으나, 이 단계를 융합한 커널 이름 `fused_qk_norm_rope`(QK-Norm)와 융합 전 패턴이 `q_norm`·`k_norm`을 나열하는 점에 비추어 `k_norm`의 오기로 판단해 바로잡았습니다.*

`scaled_mm`과 attention은 모두 [PyTorch 커스텀 연산자(Custom Operators)](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)로 등록된다는 점에 유의하세요. 이들 연산자는 TorchInductor에게 불투명(opaque)하기 때문에, 컴파일러 측의 추가 융합을 막는 단단한 경계(hard boundary)를 형성합니다.
> Note that both `scaled_mm` and attention are registered as [PyTorch Custom Operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html). Since these operators are opaque to TorchInductor, they form hard boundaries that prevent further compiler-side fusion.

**동적 그룹별 활성화 양자화 / Dynamic Per-Group Activation Quantization**

동적 그룹별 활성화 양자화를 켜고 `scaled_mm_blockwise`에 DeepGEMM이 선택되면, 실행 패턴은 다음과 같이 바뀝니다:
> When dynamic per-group activation quantization is enabled and DeepGEMM is selected for `scaled_mm_blockwise`, the execution pattern changes to:

1. rms_norm
2. fp8_quant (ue8m0)
3. scaled_mm (qkv_proj, DeepGEMM)
4. split_qkv + q_norm + k_norm
5. rope
6. attention
7. fp8_quant (ue8m0)
8. scaled_mm (out_proj, DeepGEMM)
9. rms_norm
10. fp8_quant (ue8m0)
11. scaled_mm (gate_up, DeepGEMM)
12. silu_and_mul
13. fp8_quant (ue8m0)
14. scaled_mm (down_proj, DeepGEMM)

*역주: 위 토큰별 패턴과 마찬가지로, 원문 4번 항목의 `v_norm`을 `k_norm`의 오기로 판단해 바로잡았습니다.*

DeepGEMM은 내부적으로 UE8M0 활성화 양자화를 사용합니다. 현재 vLLM 구현에서는 `fuse_act_quant`와 `fuse_norm_quant` 패스가 UE8M0 양자화에 대해 지원되지 않아, 이러한 추가 융합이 일어나지 못합니다.
> DeepGEMM uses UE8M0 activation quantization internally. In the current vLLM implementation, `fuse_act_quant` and `fuse_norm_quant` passes are not supported for UE8M0 quantization, which prevents these additional fusions from occurring.

DeepGEMM을 쓸 수 없어 CUTLASS 기반 커널이 대신 사용되면, 실행 패턴은 동적 토큰별 양자화 경우와 비슷해집니다.
> If DeepGEMM is unavailable and CUTLASS-based kernels are used instead, the execution pattern becomes similar to the dynamic per-token quantization case.

### Helion 커널 구현 / Helion Kernels Implementation

이번 작업에서는 다음 Helion 커널을 구현했습니다:
> For this work, we implemented the following Helion kernels:

- dynamic_per_token_scaled_fp8_quant
- rms_norm_dynamic_per_token_quant
- silu_and_mul_dynamic_per_token_quant
- fused_qk_norm_rope
- per_token_group_fp8_quant
- rms_norm_per_block_quant
- silu_and_mul_per_block_quant
- scaled_mm
- scaled_mm_blockwise

`scaled_mm`과 `scaled_mm_blockwise` 커널은 vLLM의 기존 Triton 구현([triton_scaled_mm](https://github.com/vllm-project/vllm/blob/v0.21.1rc0/vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py#L141), [w8a8_triton_block_scaled_mm](https://github.com/vllm-project/vllm/blob/v0.21.1rc0/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L835))을 따릅니다. `silu_and_mul_dynamic_per_token_quant`는 `silu_and_mul`과 `dynamic_per_token_quant`를 단일 커널 실행(launch)으로 결합한 새로운 융합 커널입니다. 나머지 커널은 vLLM이 사용하는 기존 `torch.ops._C` CUDA 커널을 Helion으로 재구현한 것입니다.
> The `scaled_mm` and `scaled_mm_blockwise` kernels follow the existing Triton implementations in vLLM ([triton_scaled_mm](https://github.com/vllm-project/vllm/blob/v0.21.1rc0/vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py#L141), [w8a8_triton_block_scaled_mm](https://github.com/vllm-project/vllm/blob/v0.21.1rc0/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L835)). `silu_and_mul_dynamic_per_token_quant` is a new fused kernel that combines `silu_and_mul` and `dynamic_per_token_quant` into a single kernel launch. The remaining kernels are Helion reimplementations of the existing `torch.ops._C` CUDA kernels used by vLLM.

### vLLM-Helion 커널 통합 / vLLM Helion Kernel Integration

이 커널들은 [vLLM Helion 커널 통합 프레임워크](https://github.com/vllm-project/vllm/issues/32219)를 사용해 통합했으며, 이 프레임워크는 다음을 제공합니다:
> We integrated these kernels using the [vLLM Helion kernel integration framework](https://github.com/vllm-project/vllm/issues/32219) which provided:

- 자동 튜닝 인프라
- 구성(config) 관리
- 커널 등록
- 런타임 디스패칭(dispatching)

> - Autotuning infrastructure
> - Config management
> - Kernel registration
> - Runtime dispatching

Helion 커널을 활성화하기 위해, 해당 커널을 대응되는 Helion 융합 커널로 교체하도록 vLLM 융합 패스를 직접 수정했습니다. 융합 후 순전파 실행 패턴은 다음과 같아졌습니다:
> To enable the Helion kernels, we manually updated vLLM fusion passes to replace the corresponding kernels with corresponding Helion fused kernels. After fusion, the forward-pass execution patterns became the following:

토큰별 활성화 양자화의 경우:
> For per-token activation quantization:

1. rms_norm_dynamic_per_token_quant (helion)
2. scaled_mm (helion)
3. fused_qk_norm_rope (helion)
4. attention (default)
5. dynamic_per_token_scaled_fp8_quant (helion)
6. scaled_mm (helion)
7. rms_norm_dynamic_per_token_quant (helion)
8. scaled_mm (helion)
9. silu_and_mul_dynamic_per_token_quant (helion)
10. scaled_mm (helion)

그룹별 활성화 양자화의 경우:
> For per-group activation quantization:

1. rms_norm_per_block_quant (helion)
2. scaled_mm_blockwise (helion)
3. fused_qk_norm_rope (helion)
4. attention (default)
5. per_token_group_fp8_quant (helion)
6. scaled_mm_blockwise (helion)
7. rms_norm_per_block_quant (helion)
8. scaled_mm_blockwise (helion)
9. silu_and_mul_per_block_quant (helion)
10. scaled_mm_blockwise (helion)

### 자동 튜닝 / Autotuning

Helion의 기본 [LFBOTreeSearch](https://helionlang.com/api/autotuner.html#helion.autotuner.surrogate_pattern_search.LFBOTreeSearch) 알고리즘을 다음 구성으로 사용했습니다:
> We used the Helion's default [LFBOTreeSearch](https://helionlang.com/api/autotuner.html#helion.autotuner.surrogate_pattern_search.LFBOTreeSearch) algorithm with the following configuration:

```
initial_population=FROM_RANDOM, copies=5, max_generations=20, similarity_penalty=1.0
```

성능을 최대화하기 위해, 은닉 크기(hidden size)나 중간 크기(intermediate size)처럼 각 모델의 컴파일 타임 정적 차원과 정확히 일치하는 형상(shape)을 사용해 커널을 자동 튜닝했습니다. 이것이 vLLM-Helion 통합의 장점인데, Helion이 다양한 형상에 대해 구성을 자동 튜닝·저장·디스패치할 수 있게 해 주며, 동일한 장점이 실제 프로덕션 사용 사례에도 적용됩니다.
> To maximize performance, we autotuned kernels using shapes that exactly match the compile-time static dimensions of each model, such as hidden size and intermediate size. This is the advantage of vLLM-Helion integration – it allows Helion to autotune/store/dispatch configs for many different shapes, the same advantage would apply to real world production use cases too.

동적 차원(`num_tokens`)에 대해서는 1부터 8192까지 2의 거듭제곱 값에 걸쳐 자동 튜닝했습니다.
> For the dynamic dimension (`num_tokens`), we autotuned across power-of-two values ranging from 1 to 8192.

예를 들어, 입력 텐서 `[M, K] x [K, N]`에 대해 `scaled_mm` 커널을 자동 튜닝했으며, 여기서
> For example, we autotuned `scaled_mm` kernel for input tensors `[M, K] x [K, N]`, where

- M은 1부터 8192까지의 범위를 갖고,
- (K, N) 쌍은 각 Qwen3 모델의 프로젝션 계층(projection layer)에 대응합니다.

> - M ranges from 1 to 8192
> - (K, N) pairs correspond to the projection layers of each Qwen3 model.

| 모델 / Model | qkv_proj | out_proj | gate_up | down_proj |
|------|----------|----------|---------|-----------|
| Qwen3-1.7B | [2048, 4096] | [2048, 2048] | [2048, 12288] | [6144, 2048] |
| Qwen3-8B | [4096, 6144] | [4096, 4096] | [4096, 24576] | [12288, 4096] |
| Qwen3-32B | [5120, 10240] | [5120, 5120] | [5120, 51200] | [25600, 5120] |

*표 1: 각 Qwen3 모델의 프로젝션 계층 [K, N] 차원. / Tab. 1: Projection layer [K, N] dimensions for each Qwen3 model.*

테스트 대상인 각 하드웨어 플랫폼에 대해 모든 커널을 독립적으로 자동 튜닝했습니다.
> We independently autotuned all kernels for each hardware platform under test.

### 런타임 디스패칭 / Runtime Dispatching

런타임에는 [Helion 통합 프레임워크](https://github.com/vllm-project/vllm/issues/32219)가 입력 형상에 가장 적합한 자동 튜닝 구성으로 요청을 디스패치했습니다.
> At runtime, the [Helion integration framework](https://github.com/vllm-project/vllm/issues/32219) dispatched requests to the autotuned config most appropriate for the input shape.

예를 들어 scaled_mm 디스패칭은 두 입력 행렬의 형상(M, K, N)을 기준으로 수행되는데, 여기서 M은 각 요청 배치의 런타임 `num_tokens`에 따라 다음 2의 거듭제곱으로 올림됩니다. 다른 커널에도 비슷한 전략이 적용됩니다.
> For example, scaled_mm dispatching is performed based on shapes of two input matrices (M, K, N), where M is rounded up to the next power of two according to runtime `num_tokens` of each batch of requests. Similar strategy is applied to other kernels as well.

## 성능 평가 — 커널 수준 / Performance Evaluation – Kernel Level

커널 수준 벤치마킹은 각 개별 Helion 커널이 베이스라인 대비 만들어내는 국소적(local) 속도 향상을 평가하는 것이 목적입니다. 구체적으로 `scaled_mm`과 `scaled_mm_blockwise`에 대해서는 CUTLASS를 베이스라인으로 사용했습니다. 반면 다른 연산은 torch.compile된 vLLM 구현 및 기존 `torch.ops._C` 커널과 비교했습니다. 그 이유는 다음과 같습니다:
> Kernel level benchmarking aims to evaluate the local speedups produced by each individual Helion kernel against their baselines. Specifically, we used CUTLASS as the baseline for `scaled_mm` and `scaled_mm_blockwise`. While other ops are compared against torch.compile 'ed vLLM implementation and existing `torch.ops._C` kernels. This is because:

- vLLM에서 토큰별 양자화는 기본적으로 `torch.compile`을 사용하고,
- 그룹별 양자화는 이 [성능 이슈](https://github.com/vllm-project/vllm/issues/25094) 때문에 기본적으로 `torch.ops._C` CUDA 구현을 사용합니다.

> - per-token quantization in vLLM uses `torch.compile` by default,
> - per-group quantization uses `torch.ops._C` CUDA implementations by default due to this [performance issue](https://github.com/vllm-project/vllm/issues/25094).

torch.compile 베이스라인의 경우, vLLM 컴파일 설정에 맞췄습니다:
> For the torch.compile baseline, we matched the vLLM compilation setup:

```python
torch.compile(
    native_torch_impl,
    fullgraph=True,
    dynamic=False,
    backend="inductor",
    options={
        'enable_auto_functionalized_v2': False,
        'size_asserts': False,
        'alignment_asserts': False,
        'scalar_asserts': False,
        'combo_kernels': True,
        'benchmark_combo_kernel': True
    }
)
```

특히 `'combo_kernels': True`를 켜는 것이 중요한데, 이는 TorchInductor가 여러 독립적인 커널을 하나의 실행으로 융합할 수 있게 해 주기 때문입니다.
> Notably, enabling `'combo_kernels': True` is important because it allows TorchInductor to fuse multiple independent kernels into a single launch

커널 수준 벤치마킹에서는 `triton.testing.do_bench_cudagraph`를 통해 `CudaGraph` 모드를 켰고, 적절한 워밍업과 반복 테스트를 거쳐 디스패치 오버헤드, 콜드 캐시(cold cache), 측정 시간 변동 같은 노이즈를 제거했습니다.
> For kernel-level benchmarking, we enabled `CudaGraph` mode via `triton.testing.do_bench_cudagraph` with proper warmup and repetitive testing to get rid of noises like dispatch overhead or cold cache and variations in timing.

| 커널 \ 베이스라인 대비 속도 향상 (하드웨어) / Kernel \ Speedup against baseline (Hardware) | torch.compile 대비 (H100) | torch.ops._C 대비 (H100) | CUTLASS 대비 (H100) | torch.compile 대비 (B200) | torch.ops._C 대비 (B200) | CUTLASS 대비 (B200) |
|---|---|---|---|---|---|---|
| dynamic_per_token_scaled_fp8_quant | 1.237x | 1.405x | N/A | 1.311x | 1.495x | N/A |
| rms_norm_dynamic_per_token_quant | 1.180x | 1.802x | N/A | 1.240x | 1.969x | N/A |
| silu_and_mul_dynamic_per_token_quant | 1.256x | N/A | N/A | 1.420x | N/A | N/A |
| fused_qk_norm_rope | 1.383x | 1.204x | N/A | 1.133x | 1.155x | N/A |
| per_token_group_fp8_quant | 1.423x | 1.408x | N/A | 1.150x | 1.446x | N/A |
| rms_norm_per_block_quant | 1.674x | 2.055x | N/A | 1.424x | 2.128x | N/A |
| silu_and_mul_per_block_quant | 1.731x | 2.269x | N/A | 1.483x | 2.325x | N/A |
| scaled_mm | N/A | N/A | 1.080x | N/A | N/A | 0.739x |
| scaled_mm_blockwise | N/A | N/A | 0.957x | N/A | N/A | 0.782x |

*표 2: Helion 커널이 달성한 기하 평균(geometric-mean) 속도 향상 요약. / Tab. 2: A summary of the geometric-mean speedups achieved by Helion kernels.*

GEMM이 아닌 커널에서 Helion은 일관되게 강력한 성능을 보이며, TorchInductor가 생성한 커널과 기존 vLLM CUDA 구현을 모두 능가합니다.
> For non-GEMM kernels, Helion consistently demonstrates strong performance and outperforms both TorchInductor-generated kernels and the existing vLLM CUDA implementations.

GEMM 워크로드(`scaled_mm`, `scaled_mm_blockwise`)에서는 결과가 더 엇갈렸습니다:
> For GEMM workloads (`scaled_mm` and `scaled_mm_blockwise`), results were more mixed:

- H100에서는 scaled_mm이 CUTLASS를 능가했습니다.
- B200에서는 두 GEMM 커널 모두 현재 CUTLASS에 뒤처졌습니다.

> - On H100, scaled_mm outperformed CUTLASS.
> - On B200, both GEMM kernels currently lagged behind CUTLASS

B200의 주된 제약 요인은 Helion 프로그래밍 모델 자체가 아니라 Blackwell GPU에서의 Triton 생성 GEMM 커널 성능입니다. Helion은 현재 이 커널들에 대해 Triton 코드 생성에 의존하며, 관찰된 성능 격차는 대체로 Blackwell 하드웨어에서의 현재 Triton GEMM 성능 수준을 반영합니다. Helion의 CuteDSL 백엔드에 대한 진행 중인 작업을 통해 Blackwell에서의 GEMM 성능이 더욱 개선될 것으로 보입니다.
> The primary limiting factor for B200 is the performance of Triton-generated GEMM kernels on Blackwell GPUs rather than the Helion programming model itself. Helion currently relies on Triton code generation for these kernels, and the observed performance gap largely reflects the current state of Triton GEMM performance on Blackwell hardware. Ongoing work on Helion's CuteDSL backend is expected to further improve GEMM performance on Blackwell.

## 성능 평가 — 엔드투엔드 모델 수준 / Performance Evaluation – End-to-End Model Level

반면 엔드투엔드 모델 수준 벤치마킹은 Helion 커널이 사용자에게 미치는 가시적 영향을 부각합니다. 이를 위해 Qwen3 모델의 세 가지 변형을 선택했습니다:
> End-to-end model level benchmarking, on the other hand, highlights the user-visible impact of Helion kernels. We picked 3 different variants of Qwen3 models for this purpose:

- Qwen3-1.7B
- Qwen3-8B
- Qwen3-32B

세 Qwen3 모델 모두에 대해, 1부터 8192까지 2의 거듭제곱 간격으로 num_tokens 값을 변화시키는 모든 모델 수준 벤치마킹 트래픽 패턴에서 `CudaGraph`를 켰습니다.
> `CudaGraph` is enabled for all model-level benchmarking traffic patterns, which varies num_tokens values ranging from 1 to 8192 at power-of-two intervals for all three Qwen3 models.

트래픽 패턴을 구성하기 위해 무작위 입력 데이터를 사용하는 vLLM 내장 서빙 벤치마크를 사용했습니다.
> To construct the traffic pattern, we used the built-in vLLM serving benchmark with the random input data.

프리픽스 캐싱(prefix caching) 효과로 인한 노이즈를 최소화하기 위해 다음을 수행했습니다:
> To minimize noise from prefix caching effects, we:

- 프롬프트 셔플링을 비활성화했고,
- 각 벤치마크 실행 전에 vLLM 서버를 재시작했습니다.

> - disabled prompt shuffling,
> - restarted the vLLM server before each benchmark run.

예시 명령은 다음과 같습니다:
> Here is an example command:

```sh
vllm serve --model $MODEL --max-num-seqs $BATCH_SIZE --tensor-parallel-size 1 --compilation-config '{"max_cudagraph_capture_size": 8192, "custom_ops": ["+quant_fp8"], "pass_config": {"fuse_norm_quant": true, "fuse_act_quant": true, "enable_qk_norm_rope_fusion": true}}'

vllm bench serve \
  --backend vllm \
  --model $MODEL \
  --endpoint /v1/completions \
  --dataset-name random \
  --num-prompts $NUM_PROMPTS \
  --max-concurrency $BATCH_SIZE \
  --input-len 512 \
  --output-len 600 \
  ----num-warmups $NUM_WARMUPS \
  --disable-shuffle
```

`max_cudagraph_capture_size`는 기본 `max_num_batched_tokens`에 맞춰 8192로 설정해, 모든 실행 경로가 CUDA 그래프로 캡처되도록 했습니다.
> `max_cudagraph_capture_size` was set to 8192 to match the default `max_num_batched_tokens`, ensuring all execution paths were CUDA-graph captured.

모든 워크로드는 두 NVIDIA GPU 플랫폼에서 평가했습니다:
> All workloads are evaluated on two NVidia GPU platforms:

- NVIDIA H100
- NVIDIA B200

성능 개선이 어디에서 나오는지 더 깊이 파악하기 위해, Helion 커널을 세 가지 범주로 묶어 독립적으로, 그리고 조합한 형태로도 벤치마크했습니다.
> To gain more insight into where performance improvements come from, we grouped the Helion kernels into three categories and benchmarked them independently as well as in combinations.

- **fp8_quant**: fp8 양자화 커널 및 융합 양자화 커널
- **qk_norm_rope**: `fused_qk_norm_rope` 커널
- **scaled_mm**: `scaled_mm` 또는 `scaled_mm_blockwise` 커널.

> - **fp8_quant**: fp8 quantization kernels and fused quant kernels
> - **qk_norm_rope**: `fused_qk_norm_rope` kernel
> - **scaled_mm**: `scaled_mm` or `scaled_mm_blockwise` kernel.

#### 동적 토큰별 활성화 양자화 / Dynamic per-token activation quantization

다음 체크포인트를 사용했습니다:
> We used the following checkpoints:

- RedHatAI/Qwen3-1.7B-FP8-dynamic
- RedHatAI/Qwen3-8B-FP8-dynamic
- RedHatAI/Qwen3-32B-FP8-dynamic

![H100에서 토큰별 활성화 양자화를 켰을 때의 총 처리량 속도 향상 / Total throughput speedup on H100 with per-token activation quantization](/assets/blog/2026-06-10-portable-vllm-model-inference-kernels-in-helion/fig1-h100-per-token-throughput.jpg){:style="width:100%"}

*그림 1: 기본 vLLM 설정을 베이스라인으로 하여, H100에서 토큰별 활성화 양자화를 켰을 때의 총 처리량 속도 향상. / Fig. 1: Total throughput speedup on H100 with per-token activation quantization enabled, using the default vLLM setup as the baseline.*

1.7B 모델의 경우, 모든 Helion 커널 그룹을 켰을 때 H100에서 약 1.05배의 엔드투엔드 처리량 향상을 보입니다. 8B 모델의 경우 배치 크기 32 부근에서 개선이 가장 두드러지는데, 이는 Helion scaled_mm이 `num_tokens = 32` 부근에서 가장 강력한 성능을 내는 커널 수준 관찰 결과와 일치합니다.
> For the 1.7B model, the results show approximately 1.05x end-to-end throughput improvement on H100 when all Helion kernel groups are enabled. For the 8B model, the improvement is most pronounced around batch size 32, which aligns with the kernel-level observations where Helion scaled_mm achieves its strongest performance around `num_tokens = 32`.

또한 디코딩 단계의 유효 `num_tokens`가 자연스럽게 이 성능 스위트 스폿(sweet spot)에 들어가는 추측 디코딩 시나리오도 평가했습니다.
> We also evaluated speculative decoding scenarios where the effective decode-phase `num_tokens` naturally falls into this performance sweet spot.

다음을 사용해:
> Using:

- RedHatAI/Qwen3-8B-speculator.eagle3
- RedHatAI/Qwen3-32B-speculator.eagle3

모든 Helion 커널을 켰을 때 최대 약 1.09배의 엔드투엔드 처리량 향상을 관찰했습니다.
> we observed up to approximately 1.09x end-to-end throughput improvement when all Helion kernels were enabled.

| 배치 크기 / Batch Size | 모델 / Model | 추측 토큰 수 (위치별 수용률) / # Speculative Tokens (per-pos acc rate) | Helion TTFT (평균, ms) | Default TTFT (평균, ms) | **TTFT 속도 향상 / Speedup** | Helion TPOT (평균, ms) | Default TPOT (평균, ms) | **TPOT 속도 향상 / Speedup** | Helion 총 처리량 (tok/s) | Default 총 처리량 (tok/s) | **총 처리량 속도 향상 / Total Throughput Speedup** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | Qwen3-8B | 1 (47%) | 34.75 | 39.93 | **1.15x** | 4.63 | 5.01 | **1.08x** | 6,314.86 | 5817.23 | **1.09x** |
| 16 | Qwen3-8B | 3 (35%, 25%, 15%) | 38.46 | 51.18 | **1.33x** | 4.40 | 4.63 | **1.05x** | 6,616.60 | 6261.1 | **1.06x** |
| 8 | Qwen3-32B | 2 (24%, 10%) | 81.92 | 100.93 | **1.23x** | 13.29 | 14.37 | **1.08x** | 1,101.61 | 1018.32 | **1.08x** |
| 8 | Qwen3-32B | 3 (24%, 10%, 4%) | 83.01 | 104.73 | **1.26x** | 13.33 | 14.21 | **1.07x** | 1,100.04 | 1030.51 | **1.07x** |

*표 3: H100에서 토큰별 활성화 양자화와 추측 디코딩을 켠 엔드투엔드 벤치마크 결과. 추측 토큰의 수용률은 괄호 안에 표기. / Tab. 3: End-to-end benchmark results on H100 with per-token activation quantization and speculative decoding enabled. Acceptance rates for speculative tokens are reported in parentheses.*

NVIDIA B200에서는 엔드투엔드 평가 동안 `fp8_quant` 커널 그룹만 켰습니다. 나머지 커널 그룹은 다음 중 하나에 해당했기 때문입니다:
> On NVIDIA B200, we enabled only the `fp8_quant` kernel group during end-to-end evaluation. The remaining kernel groups either:

- 베이스라인 대비 성능이 떨어졌거나(Blackwell GEMM에 대한 Triton의 한계),
- 트래픽 패턴 전반에서 일관되지 않은 이득을 보였습니다.

> - underperformed relative to the baseline (Triton limitation for Blackwell GEMMs)
> - or showed inconsistent gains across traffic patterns.

양자화 관련 커널만 켰을 때조차, 테스트한 모든 Qwen3 모델 크기에서 의미 있는 처리량 향상을 여전히 관찰했습니다.
> Even with only the quantization-related kernels enabled, we still observed meaningful throughput improvements across all tested Qwen3 model sizes.

![B200에서 토큰별 활성화 양자화를 켰을 때의 총 처리량 속도 향상 / Total throughput speedup on B200 with per-token activation quantization](/assets/blog/2026-06-10-portable-vllm-model-inference-kernels-in-helion/fig2-b200-per-token-throughput.jpg){:style="width:100%"}

*그림 2: 기본 vLLM 설정을 베이스라인으로 하여, B200에서 토큰별 활성화 양자화를 켰을 때의 총 처리량 속도 향상. / Fig. 2: Total throughput speedup on B200 with per-token activation quantization enabled, using the default vLLM setup as the baseline.*

#### 동적 그룹별 활성화 양자화 / Dynamic per-group activation quantization

그룹별 활성화 양자화의 경우, 다음 체크포인트를 사용했습니다:
> For per-group activation quantization, we used the following checkpoints:

- Qwen/Qwen3-1.7B-FP8
- Qwen/Qwen3-8B-FP8
- Qwen/Qwen3-32B-FP8

그룹별 활성화 양자화에서는 H100과 B200 모두에서 블록 단위(blockwise) FP8 GEMM의 기본 백엔드가 DeepGEMM입니다. 그러나 현재 그룹별 Helion 양자화 커널은 DeepGEMM이 요구하는 UE8M0 양자화 포맷과 아직 호환되지 않습니다. 따라서 이 실험에서는 vLLM이 선형 계층(linear) 백엔드로 CUTLASS를 사용하도록 강제했습니다.
> For per-group activation quantization, DeepGEMM is the default backend for blockwise FP8 GEMM on both H100 and B200. However, our current per-group Helion quantization kernels are not yet compatible with the UE8M0 quantization format required by DeepGEMM. Therefore, for this experiment, we forced vLLM to use CUTLASS as the linear backend.

이는 이 절의 베이스라인이 기본 vLLM 구성이 **아니라는** 뜻입니다. 그렇더라도 모든 실행에서 선형 계층에 일관된 CUTLASS 커널을 사용할 수 있으므로 비교는 여전히 유의미합니다. 결과적으로 측정된 차이는 선형 백엔드의 변화가 아니라, 평가 대상인 비-GEMM 커널(FP8 양자화 및 융합 양자화 커널 등)에서 나옵니다.
> This means the baseline in this section is **not** the default vLLM configuration. However, the comparison is still meaningful because we are able to use consistent CUTLASS kernels for the linear layer for all runs. As a result, the measured differences come from the non-GEMM kernels being evaluated, such as FP8 quantization and fused quantization kernels, rather than from changes in the linear backend.

다음 그림들은 작은 Helion 커널만 켰을 때에도 모든 워크로드에서 약 1.05배의 엔드투엔드 처리량 향상이 나타났음을 보여줍니다.
> The following figures show enabling only the small Helion kernels still produced approximately 1.05x end-to-end throughput improvement across all workloads.

![H100과 B200에서 그룹별 활성화 양자화를 켰을 때의 총 처리량 속도 향상 / Total throughput speedup on H100 and B200 with per-group activation quantization](/assets/blog/2026-06-10-portable-vllm-model-inference-kernels-in-helion/fig3-per-group-throughput.jpg){:style="width:100%"}

*그림 3: 선형 계층 백엔드를 CUTLASS로 교체한 기본 vLLM 설정을 베이스라인으로 하여, H100과 B200에서 그룹별 활성화 양자화를 켰을 때의 총 처리량 속도 향상. / Fig. 3: Total throughput speedup on H100 and B200 with per-group activation quantization enabled, using the default vLLM setup with the linear layer backend replaced by CUTLASS as the baseline.*

## 자료 / Resources

재현성과 추가 탐색을 위해, 이번 글에서 다룬 모든 Helion 커널 구현은 해당 GitHub [이슈](https://github.com/vllm-project/vllm/issues/32962)에 링크되어 있습니다. 같은 이슈에는 보고된 엔드투엔드 벤치마크 결과를 재현하기 위해 실험에 사용한 vLLM 브랜치도 포함되어 있습니다.
> For reproducibility and further exploration, all Helion kernel implementations discussed in this post are linked in the corresponding GitHub [issue](https://github.com/vllm-project/vllm/issues/32962). The same issue also includes the vLLM branches used in our experiments for reproducing the reported end-to-end benchmark results.

## 유의 사항 / Caveats

실험을 진행하는 동안 엔지니어링 시간의 대부분은 커널 자동 튜닝에 들어갔습니다. scaled_mm 같은 대형 커널의 경우, 총 [168](https://github.com/xiaohongchen1991/vllm/blob/91142591ec0b2da967c600599421ee60fed4f6ca/vllm/kernels/helion/ops/scaled_mm.py#L33-L50)개의 서로 다른 입력 형상을 아우르는 세 모델 크기 전체에 대해 전력(full-effort) 자동 튜닝 스윕을 돌리면 하루가 통째로 걸릴 수 있는데, Helion이 각 형상마다 수천 개의 후보 커널 구현을 자동으로 생성하고 벤치마크하기 때문입니다. 초기 [연구](https://github.com/vllm-project/vllm/commit/5bc478ccee9bae4056aeae9953861fe587265e3f#diff-be77e79f35962c7bc20c44638613a5fdca7bb745b987888b4c63dd7557dd4207)에 따르면, 형상별 완전 자동 튜닝과 디스패칭이 항상 필요한 것은 아닐 수 있으며, 특화(specialization) 버킷 수를 줄이면 성능 저하를 최소화하면서 자동 튜닝 비용과 런타임 성능 사이에서 더 나은 절충을 달성할 수 있다고 합니다. Helion 팀은 탐색 공간 축소 전략과 LLM 기반 자동 튜닝 접근법을 포함해, 튜닝 시간을 더 줄이기 위한 추가 기법을 적극적으로 탐색하고 있습니다.
> During our experiments, the majority of engineering time was spent on kernel autotuning. For large kernels such as scaled_mm, running a full-effort autotuning sweep across all three model sizes, covering a total of [168](https://github.com/xiaohongchen1991/vllm/blob/91142591ec0b2da967c600599421ee60fed4f6ca/vllm/kernels/helion/ops/scaled_mm.py#L33-L50) distinct input shapes, can take an entire day, as Helion automatically generates and benchmarks thousands of candidate kernel implementations for each shape. Initial [research](https://github.com/vllm-project/vllm/commit/5bc478ccee9bae4056aeae9953861fe587265e3f#diff-be77e79f35962c7bc20c44638613a5fdca7bb745b987888b4c63dd7557dd4207) suggests that exhaustive per-shape autotuning and dispatching may not always be necessary, and that reducing the number of specialization buckets may achieve a better tradeoff between autotuning cost and runtime performance with minimal performance degradation. The Helion team is actively exploring additional techniques to further reduce tuning time, including search-space reduction strategies and LLM-guided autotuning approaches.

또 다른 유의 사항은 Helion 런타임 디스패칭 자체가 커널 실행마다 수십 마이크로초의 CPU 오버헤드를 유발한다는 점입니다. 작은 커널의 경우 이 오버헤드가 엔드투엔드 지연 시간을 지배할 수 있습니다. 그 결과 Helion 커널로 최적의 성능을 얻으려면 CUDA 그래프 캡처와 재생(replay)이 필수입니다. Helion 팀은 CudaGraph 모드 없이도 디스패치 지연 시간을 줄이는 작업을 적극적으로 진행하고 있습니다.
> Another caveat is that Helion runtime dispatching itself introduces tens of microseconds of CPU overhead per kernel launch. For small kernels, this overhead can dominate the end-to-end latency. As a result, CUDA graph capture and replay are essential for achieving optimal performance with Helion kernels. The Helion team is actively reducing the dispatch latency without CudaGraph mode.

## 결론 / Conclusion

Helion은 타일 프로그래밍 스타일로 커널을 작성하는 자연스럽고 PyTorch 문법 중심인 접근법을 제공합니다. 이는 커널 개발을 크게 단순화하고 구현 노력을 줄여 줍니다. 실험에서 대부분의 커널은 하루 안에 구현하고 검증할 수 있었으며, 이는 Helion이 새로운 커널을 빠르게 개발하고 커널 융합 기회를 탐색하는 데 실용적인 DSL임을 보여 줍니다.
> Helion provides a natural, PyTorch-syntax-centric approach for writing kernels in a tile-programming style. It significantly simplifies kernel development and reduces implementation effort. In our experiments, most kernels could be implemented and validated within a single day, demonstrating that Helion is a practical DSL for rapidly developing new kernels and exploring kernel fusion opportunities.

강력한 AOT 자동 튜닝 역량과 결합되어, Helion은 높은 성능을 달성할 강력한 잠재력을 보였습니다. 실험 결과 Helion 커널은 다수의 커널에서 강력한 성능을 내며 대부분의 경우 기본 vLLM 구현을 일관되게 능가합니다. GEMM 커널의 경우, 특히 Blackwell GPU에서 CUTLASS 성능에 맞먹거나 능가하기까지는 아직 개선의 여지가 있으며, 팀은 Triton 코드 생성을 개선하고 CuteDSL 같은 대체 백엔드를 도입해 이를 적극적으로 개선하고 있습니다.
> Combined with its powerful AOT autotuning capability, Helion demonstrated strong potential for achieving high performance. Our experiments show that Helion kernels deliver strong performance for many kernels and consistently outperform the default vLLM implementations in most cases. For GEMM kernels, there is still room for improvement to match or exceed CUTLASS performance, particularly on Blackwell GPUs, the teams are actively working to improve it by improving Triton code gen and introducing alternative backends like CuteDSL.

## 감사의 말 / Acknowledgments

이 작업은 Red Hat의 OCTO 및 vLLM 팀, 그리고 Meta의 Helion 팀 등 여러 기여자의 지원을 받았습니다. 특히 이 작업 전반에 걸쳐 피드백과 지원을 아끼지 않은 동료 Luka Govedič, Richard Zou, Will Feng에게 감사드립니다.
> This work was supported by many contributors across the OCTO and vLLM teams at Red Hat, as well as the Helion team at Meta. In particular, we would like to thank our colleagues: Luka Govedič, Richard Zou and Will Feng for their feedback and support throughout this work.
