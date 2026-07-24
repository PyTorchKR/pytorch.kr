---
layout: blog_detail
title: "PyTorch 2.1에 새로 추가된 성능 향상 기능 소개"
image: /assets/blog/2023-11-29-new-features-for-ai/hero.png
org_title: "PyTorch 2.1 Contains New Performance Features for AI Developers"
author: 인텔(Intel)
category: ["pytorch.org", "translation"]
org_link: https://pytorch.org/blog/new-features-for-ai/
---

PyTorch(파이토치) 2.1을 출시하여 매우 기쁩니다. 이번 글에서는 PyTorch 2.1에 인텔(Intel)이 크게 기여한 다섯가지 기능들에 대해서 설명하겠습니다:
> We are excited to see the release of PyTorch 2.1. In this blog, we discuss the five features for which Intel made significant contributions to PyTorch 2.1:

1. torch.compile()시 bfloat16 추론 경로를 포함하는 TorchInductor-CPU 최적화
2. torch.compile()에서 CPU용 동적 쉐입(dynamic shape) 추론 경로
3. C++ 래퍼 (wrapper, 프로토타입 기능)
4. CPU용 플래시 어텐션(flash-attention) 기반 스케일드-닷-프로덕트(scaled-dot-product) 알고리즘
5. Inductor를 통해 x86 백엔드를 사용한 PyTorch 2의 학습 후 양자화 내보내기 기능
> 1. TorchInductor-CPU optimizations including Bfloat16 inference path for torch.compile
> 2. CPU dynamic shape inference path for torch.compile
> 3. C++ wrapper (prototype)
> 4. Flash-attention-based scaled dot product algorithm for CPU
> 5. PyTorch 2 export post-training qauantization with an x86 back end through an inductor

인텔이 PyTorch 커뮤니티의 일원이 된 것을 기쁘게 생각하며, Meta*의 동료들과 이러한 기능들을 함께 개발하고 피드백을 주신 것에 감사드립니다.
> At Intel, we are delighted to be part of the PyTorch community and appreciate the collaboration with and feedback from our colleagues at Meta* as we co-developed these features.

이제 시작해보겠습니다.
> Let’s get started.

## TorchInductor-CPU 최적화 / TorchInductor-CPU Optimizations

이 기능은 TorchInductor(토치인덕터)의 bfloat16 추론 성능을 최적화합니다. 3세대 및 4세대의 Intel® Xeon® Scalable 프로세서에는 bfloat16 데이터 타입의 닷-프로덕트(dot-product) 연산을 가속화하는 하드웨어 가속기가 내장되어 있습니다. 아래 그림 1은 BF16 추론 경로를 지정하는 코드 조각입니다.
> This feature optimizes bfloat16 inference performance for TorchInductor. The 3rd and 4th generation Intel® Xeon® Scalable processors have built-in hardware accelerators for speeding up dot-product computation with the bfloat16 data type. Figure 1 shows a code snippet of how to specify the BF16 inference path.

```python
user_model = ...

user_model.eval()
with torch.no_grad(), torch.autocast("cpu"):
	compiled_model = torch.compile(user_model)
	y = compiled_model(x)
```

그림 1. TorchInductor에서 BF16 추론 사용 예시 코드 / Figure 1. Code snippet showing the use of BF16 inference with TorchInductor <br />


TorchBench, Hugging Face*, TIMM의 3종을 벤치마크 대상으로하여 TorchInductor의 성능을 측정했으며 그 결과는 표 1과 같습니다. 그래프 모드(TorchInductor)의 성능이 Eager 모드보다 1.25배에서 2.35배*까지 뛰어난 것을 확인할 수 있습니다.
> We measured the performance on three TorchInductor benchmark suites—TorchBench, Hugging Face*, and TIMM—and the results are as follows in Table 1. Here we see that performance in graph mode (TorchInductor) outperforms eager mode by factors ranging from 1.25x to 2.35x.*

표 1. 그래프 모드와 Eager 모드에서의 bfloat16 성능 향상 (기하 평균, geometric mean) / Table 1. Bfloat16 performance geometric mean speedup in graph mode, compared with eager mode

<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>Bfloat16 Geometric Mean Speedup (Single-Socket Multithreads)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
torchbench
   </td>
   <td>
huggingface
   </td>
   <td>
timm_models
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.81x
   </td>
   <td>
1.25x
   </td>
   <td>
2.35x
   </td>
  </tr>
</table>



<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>Bfloat16 Geometric Mean Speedup (Single-Core Single Thread)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
torchbench
   </td>
   <td>
huggingface
   </td>
   <td>
timm_models
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.74x
   </td>
   <td>
1.28x
   </td>
   <td>
1.29x
   </td>
  </tr>
</table>

Intel® Advanced Matrix Extensions(Intel® AMX) 기능을 활용하여 `torch.compile()` 성능을 최대치로 끌어냄으로써 4세대 인텔 Xeon 프로세서에서 모델을 완전히 배포할 수 있습니다. 인텔 AMX에는 타일(Tile)과 타일드 매트릭스 곱셈(TMUL; Tiled Matric Multiplication)이라는 두 가지 주요 구성 요소가 있습니다. 타일(tile)은 각각 1KB 크기를 갖는 2차원 레지스터 8개에 대량의 데이터를 저장합니다. TMUL은 한 번의 연산(single operation)으로 더 큰 행렬을 계산하기 위한 명령어들을 지원하는, 타일에 연결된 가속 엔진입니다.
> Developers can fully deploy their models on 4th generation Intel Xeon processors to take advantage of the Intel® Advanced Matrix Extensions (Intel® AMX) feature to get peak performance for `torch.compile`. Intel AMX has two primary components: tiles and tiled matrix multiplication (TMUL). The tiles store large amounts of data in eight two-dimensional registers, each one kilobyte in size. TMUL is an accelerator engine attached to the tiles that contain instructions to compute larger matrices in a single operation.


## torch.compile()에서 CPU용 동적 쉐입(dynamic shape) 추론 경로 / CPU Dynamic Shapes Inference Path for torch.compile


동적 쉐입(Dynamic Shape)은 PyTorch 2.0의 핵심 기능들 중 하나입니다. PyTorch 2.0은 기본적으로 모든 것이 정적이라고 가정하고 있습니다. 크기가 변경되어 다시 컴파일을 해야 할 때는 동적 크기를 갖도록 다시 컴파일하려고  시도합니다. (한 번 변경된 크기는 앞으로 변경될 가능성이 높습니다.) 동적 쉐입을 지원하는 경우, conv/gemm 연산자에 대한 후처리 퓨전(post-op fusion)과 conv/gemm 연산자가 아닌 연산자(non-conv-gemm operators)에 대한 벡터화 코드 생성(vectorization code-gen)을 지원합니다.
> Dynamic shapes is one of the key features in PyTorch 2.0. PyTorch 2.0 assumes everything is static by default. If we recompile because a size changed, we will instead attempt to recompile that size as being dynamic (sizes that have changed are likely to change in the future). Dynamic shapes support is required for popular models like large language models (LLM). Dynamic shapes that provide support for a broad scope of models can help users get more benefit from torch.compile. For dynamic shapes, we provide the post-op fusion for conv/gemm operators and vectorization code-gen for non-conv/gemm operators.

동적 쉐입은 CUDA*용 인덕터 트리튼 백엔드와 CPU용 C++ 백엔드에서 모두 지원하고 있습니다. 여기에는 기능(모델 통과율로 측정)과 성능(추론 지연/처리량으로 측정) 모두에 대한 개선이 포함하고 있습니다. 그림 2는 TorchInductor에서 동적 쉐입 추론을 사용하는 코드 조각을 보여줍니다.
> Dynamic shapes is supported by both the inductor Triton back end for CUDA* and the C++ back end for CPU. The scope covers improvements for both functionality (as measured by model passing rate) and performance (as measured by inference latency/throughput). Figure 2 shows a code snippet for the use of dynamic shape inference with TorchInductor.


```python
user_model = ...

# 학습 예시 / Training example
compiled_model = torch.compile(user_model)
y = compiled_model(x_size1)
# 여기서 입력 크기가 변경되어 재컴파일을 시작합니다
# Here trigger the recompile because the input size changed
y = compiled_model(x_size2)


# 추론 예시 / Inference example
user_model.eval()
compiled_model = torch.compile(user_model)
with torch.no_grad():
	y = compiled_model(x_size1)
 # 여기서 입력 크기가 변경되어 재컴파일을 시작합니다
 # Here trigger the recompile because the input size changed
 y = compiled_model(x_size2)
```

그림 2. TorchInductor로 동적 쉐입 추론을 사용하는 코드 조각 / Figure 2. Code snippet showing the use of dynamic shape inference with TorchInductor

TorchInductor 벤치마크 제품군 3종(TorchBench, Hugging Face, TIMM)에 대해서 다시 성능을 측정한 결과를 표 2에 정리하였습니다. 여기서 그래프 모드의 성능이 Eager 모드보다 1.15배에서 1.79배까지 뛰어난 것을 확인할 수 있습니다.
> We again measured the performance on the three TorchInductor benchmark suites—TorchBench, Hugging Face, and TIMM—and the results are in Table 2. Here we see that performance in graph mode outperforms eager mode by factors ranging from 1.15x to 1.79x.

표 2. 동적 쉐입에서 Eager 모드 대비 성능 향상 (기하 평균, geometric mean) / Table 2. Dynamic shape geometric mean speedup compared with Eager mode

<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>Dynamic Shape Geometric Mean Speedup (Single-Socket Multithreads)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
torchbench
   </td>
   <td>
huggingface
   </td>
   <td>
timm_models
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.35x
   </td>
   <td>
1.15x
   </td>
   <td>
1.79x
   </td>
  </tr>
</table>


<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>Dynamic Shape Geometric Mean Speedup (Single-Core Single-Thread)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
torchbench
   </td>
   <td>
huggingface
   </td>
   <td>
timm_models
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.48x
   </td>
   <td>
1.15x
   </td>
   <td>
1.48x
   </td>
  </tr>
</table>


## C++ 래퍼(프로토타입 기능) / C++ Wrapper (Prototype)

이 기능은 TorchInductor의 생성된 커널 및 외부 커널을 호출하는 Python* 코드 대신 C++ 코드를 생성하여 Python 오버헤드를 줄입니다. 또한 Python이 없는 환경에서의 배포를 지원하기 위한 중간 단계이기도 합니다.
> The feature generates C++ code instead of Python* code to invoke the generated kernels and external kernels in TorchInductor to reduce Python overhead. It is also an intermediate step to support deployment in environments without Python.

다음과 같이 설정하면 이 기능을 사용할 수 있습니다:
> To enable this feature, use the following configuration:


```python
import torch
import torch._inductor.config as config
config.cpp_wrapper = True
```

Python 래퍼에서의 오버헤드가 더 큰, 가벼운 워크로드의 경우에는 C++ 래퍼가 더 높은 성능 향상 비율을 보여줍니다. TorchBench, Hugging Face, TIMM의 모델들을 평균 추론 시간에 따라 Small, Medium, Large의 세 가지 그룹으로 정리하였습니다. 표 3은 C++ 래퍼가 기본 Python 래퍼 대비 얻은 기하 평균 성능 향상 비율을 보여줍니다.
> For light workloads where the overhead of the Python wrapper is more dominant, C++ wrapper demonstrates a higher performance boost ratio. We grouped the models in TorchBench, Hugging Face, and TIMM per the average inference time of one iteration and categorized them into small, medium, and large categories. Table 3 shows the geometric mean speedups achieved by the C++ wrapper in comparison to the default Python wrapper.

표 3. Eager 모드와 비교한 C++ 래퍼의 속도 향상 (기하 평균, geometric mean) / Table 3. C++ wrapper geometric mean speedup compared with Eager mode

<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>FP32 Static Shape Mode Geometric Mean Speedup (Single-Socket Multithreads)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
Small (t &lt;= 0.04s)
   </td>
   <td>
Medium (0.04s &lt; t &lt;= 1.5s)
   </td>
   <td>
Large (t > 1.5s)
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.06x
   </td>
   <td>
1.01x
   </td>
   <td>
1.00x
   </td>
  </tr>
</table>






<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>FP32 Static Shape Mode Geometric Mean Speedup (Single-Core Single-Thread)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
Small (t &lt;= 0.04s)
   </td>
   <td>
Medium (0.04s &lt; t &lt;= 1.5s)
   </td>
   <td>
Large (t > 1.5s)
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.13x
   </td>
   <td>
1.02x
   </td>
   <td>
1.01x
   </td>
  </tr>
</table>






<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>FP32 Dynamic Shape Mode Geometric Mean Speedup (Single-Socket Multithreads)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
Small (t &lt;= 0.04s)
   </td>
   <td>
Medium (0.04s &lt; t &lt;= 1.5s)
   </td>
   <td>
Large (t > 1.5s)
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.05x
   </td>
   <td>
1.01x
   </td>
   <td>
1.00x
   </td>
  </tr>
</table>






<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>FP32 Dynamic Shape Mode Geometric Mean Speedup (Single-Core Single-Thread)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
Small (t &lt;= 0.04s)
   </td>
   <td>
Medium (0.04s &lt; t &lt;= 1.5s)
   </td>
   <td>
Large (t > 1.5s)
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.14x
   </td>
   <td>
1.02x
   </td>
   <td>
1.01x
   </td>
  </tr>
</table>






<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>BF16 Static Shape Mode Geometric Mean Speedup (Single-Socket Multithreads)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
Small (t &lt;= 0.04s)
   </td>
   <td>
Medium (0.04s &lt; t &lt;= 1.5s)
   </td>
   <td>
Large (t > 1.5s)
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.09x
   </td>
   <td>
1.03x
   </td>
   <td>
1.04x
   </td>
  </tr>
</table>






<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>BF16 Static Shape Mode Geometric Mean Speedup (Single-Core Single-Thread)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
Small (t &lt;= 0.04s)
   </td>
   <td>
Medium (0.04s &lt; t &lt;= 1.5s)
   </td>
   <td>
Large (t > 1.5s)
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.17x
   </td>
   <td>
1.04x
   </td>
   <td>
1.03x
   </td>
  </tr>
</table>


## CPU용 플래시 어텐션(flash-attention) 기반 스케일드-닷-프로덕트(scaled-dot-product) 알고리즘 / Flash-Attention-Based Scaled Dot Product Algorithm for CPU

스케일-닷-프로덕트 어텐션(SDPA)은 Transformer 모델들의 속도 향상을 지원하는 PyTorch 2.0의 핵심 기능 중 하나입니다. 이 기능은 최적의 CUDA 커널을 사용하여 가속하지만, 최적화된 CPU의 커널은 아직 없었습니다. 이 플래시-어텐션(flash-attention) 구현은 FP32 및 bfloat16 데이터 타입에서의 학습과 추론 모두를 지원합니다. 사용자는 이 SDPA 최적화를 활용하기 위해 코드(frontend)를 변경할 필요는 없습니다. SDPA 호출 시에 자동적으로 구현체를 선택하는데, 이 새로운 구현체를 포함하고 있습니다.
> Scaled dot product attention (SDPA) is one of the flagship features of PyTorch 2.0 that helps speed up transformer models. It is accelerated with optimal CUDA kernels while still lacking optimized CPU kernels. This flash-attention implementation targets both training and inference, with both FP32 and Bfloat16 data types supported. There is no front-end use change for users to leverage this SDPA optimization. When calling SDPA, a specific implementation will be chosen automatically, including this new implementation.


허깅페이스의 SDPA 관련 모델들을 측정한 결과, 이 기능이 없는 SDPA(unfused SDPA)보다 효과적인 것을 확인하였습니다. 표 4에는 SDPA 최적화에 대한 기하 평균 속도 향상 비율을 보여줍니다.
> We have measured the SDPA-related models in Hugging Face, and they are proven effective when compared to the unfused SDPA. Shown in Table 4 are the geometric mean speedups for SDPA optimization. <br />

표 4. SDPA 최적화 속도 향상 (기하 평균, geometric mean) / Table 4. SDPA optimization performance geometric mean speedup

<table class="table table-bordered">
  <tr>
   <td colspan="3" >
<strong>SDPA Geometric Mean Speedup (Single-Socket Multithreads)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
Geometric Speedup FP32
   </td>
   <td>
Geometric Speedup BF16
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.15x, 20/20
   </td>
   <td>
1.07x, 20/20
   </td>
  </tr>
</table>


<table class="table table-bordered">
  <tr>
   <td colspan="3" >
<strong>SDPA Geometric Mean Speedup (Single-Core Single-Thread)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
Geometric Speedup FP32
   </td>
   <td>
Geometric Speedup BF16
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.02x, 20/20
   </td>
   <td>
1.04x, 20/20
   </td>
  </tr>
</table>



## Inductor를 통해 x86 백엔드를 사용한 학습 후 양자화 PyTorch 2 내보내기/ PyTorch 2 Export Post-Training Quantization with x86 Back End through Inductor


PyTorch는 PyTorch 2.0 내보내기(export)에서 새로운 양자화 흐름(quantization flow)을 제공하고 있습니다. 이 기능은 x86 CPU 장치를 백엔드로 하는 TorchInductor를 사용하여, 새로운 양자화 흐름을 적용한 학습 후 정적 양자화(post-training quantization)를 수행합니다. 예시 코드 조각은 그림 3과 같습니다.
> PyTorch provides a new quantization flow in the PyTorch 2.0 export. This feature uses TorchInductor with an x86 CPU device as the back end for post-training static quantization with this new quantization flow. An example code snippet is shown in Figure 3.


```python
import torch
import torch._dynamo as torchdynamo
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq

model = ...

model.eval()
with torch.no_grad():
 # 1단계: 모델을 평탄화(flatten)한 ATen 연산자의 FX 그래프로 추적합니다
 # Step 1: Trace the model into an FX graph of flattened ATen operators
 exported_graph_module, guards = torchdynamo.export(
	 model,
	 *copy.deepcopy(example_inputs),
	 aten_graph=True,
 )

 # 2단계: 관찰자(observers) 또는 가짜 양자화 모듈을 삽입합니다
 # Step 2: Insert observers or fake quantize modules
 quantizer = xiq.X86InductorQuantizer()
 operator_config = xiq.get_default_x86_inductor_quantization_config()
 quantizer.set_global(operator_config)
 prepared_graph_module = prepare_pt2e(exported_graph_module, quantizer)

 # 여기서 캘리브레이션을 수행합니다
 # Doing calibration here.

 # 3단계: 모델을 양자화합니다
 # Step 3: Quantize the model
 convert_graph_module = convert_pt2e(prepared_graph_module)

 # 4단계: 백엔드로 양자화된 모델을 내보냅니다
 # Step 4: Lower Quantized Model into the backend
 compile_model = torch.compile(convert_graph_module)
```

그림 3. Inductor를 PyTorch 2의 학습 후 양자화 내보내기를 위한 백엔드로 사용하는 코드 조각 / Figure 3. Code snippet showing the use of Inductor as back end for PyTorch 2 export post-training quantization

모든 합성곱 신경망(CNN; Convolutional Neural Network) 모델들은 TorchBench 벤치마크 테스트 스윗(suite)에서 측정하였으며, Inductor FP32 추론 경로와 비교하여 효과적임이 증명되었습니다. 성능 지표는 표 5와 같습니다.
> All convolutional neural networks (CNN) models from the TorchBench test suite have been measured and proven effective when compared with the Inductor FP32 inference path. Performance metrics are shown in Table 5.

표 5. Inductor를 통해 x86 백엔드를 사용한 학습 후 양자화 내보내기 성능 향상 (기하 평균, geometric mean) / Table 5. PyTorch 2 export post-training quantization performance geometric mean speedup with x86 back end through Inductor

<table class="table table-bordered">
  <tr>
   <td>
<strong>Compiler</strong>
   </td>
   <td>
<strong>Geometric Speedup</strong>
   </td>
   <td>
<strong>Geometric Related Accuracy Loss</strong>
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
3.25x, 12/12
   </td>
   <td>
0.44%, 12/12
   </td>
  </tr>
</table>


## 다음 단계 / Next Steps


### 소프트웨어 다운로드 / Get the Software


[PyTorch 2.1](https://github.com/pytorch/pytorch/releases/tag/v2.1.0)을 사용해보고 인텔이 기여한 이러한 기능들의 성능 이점을 직접 확인해보세요.
> Try out [PyTorch 2.1](https://github.com/pytorch/pytorch/releases/tag/v2.1.0) and realize the performance benefits for yourself from these features contributed by Intel.

인텔의 다른 [AI 도구들](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/tools.html)과 [프레임워크](https://www.intel.com/content/www/us/en/developer/tools/frameworks/overview.html) 최적화를 확인하고 인텔의 AI 소프트웨어 포트폴리오의 기반이 되는 오픈, 표준 기반 [oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) 멀티아키텍처, 멀티벤더 프로그래밍 모델에 대해 알아보세요.
> We encourage you to check out Intel’s other [AI Tools](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/tools.html) and [framework](https://www.intel.com/content/www/us/en/developer/tools/frameworks/overview.html) optimizations and learn about the open, standards-based [oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) multiarchitecture, multivendor programming model that forms the foundation of Intel’s AI software portfolio.

4세대 인텔 Xeon Scalable 프로세서에 대한 자세한 내용은 [AI 플랫폼](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/platform.html)에서 확인할 수 있으며, 개발자들이 고성능의 효율적인 엔드-투-엔드 AI 파이프라인을 실행할 수 있도록 지원하고 있는 방법에 대해 알아보세요.
> For more details about the 4th generation Intel Xeon Scalable processor, visit the [AI platform](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/platform.html) where you can learn how Intel is empowering developers to run high-performance, efficient end-to-end AI pipelines.


### 파이토치 리소스 / PyTorch Resources

* [파이토치 시작하기 (한국어)](http://pytorch.kr/get-started/pytorch-2.0/)
* [파이토치 개발자 포럼](http://dev-discuss.pytorch.org/t/pytorch-release-2-0-execution-update/1077)
* [파이토치 2.0 문서](http://pytorch.org/docs/2.0/)
> * [PyTorch Get Started](http://pytorch.org/get-started/pytorch-2.0/)
> * [Dev Discussions](http://dev-discuss.pytorch.org/t/pytorch-release-2-0-execution-update/1077)
> * [Documentation](http://pytorch.org/docs/2.0/)

### 제품 및 성능 정보 / Product and Performance Information

1 Amazon EC2* m7i.16xlarge: 1-node, Intel Xeon Platinum 8488C processor with 256 GB memory (1 x 256 GB DDR5 4800 MT/s), microcode 0x2b000461, hyperthreading on, turbo on, Ubuntu* 22.04.3 LTS, kernel 6.2.0-1011-aws, GCC* 11.3.0, Amazon Elastic Block Store 200 GB, BIOS Amazon EC2 1.0 10/16/2017; Software: [PyTorch 2.1.0_rc4](https://github.com/pytorch/pytorch/tree/release/2.1), [Intel® oneAPI Deep Neural Network Library (oneDNN) version 3.1.1](https://github.com/oneapi-src/oneDNN/tree/v3.1.1), [TorchBench](https://github.com/pytorch/benchmark/commit/ffbbebb9), [TorchVision](https://github.com/pytorch/vision/commit/8636bf3), [TorchText](https://github.com/pytorch/text/commit/142d029), [TorchAudio](https://github.com/pytorch/audio/commit/475b6ae), [TorchData](https://github.com/pytorch/data/commit/eb9bf61), [TorchDynamo Benchmarks](https://github.com/pytorch/pytorch/tree/release/2.1/benchmarks/dynamo), tested by Intel on 9/12/2023.


2 Amazon EC2 c6i.16xlarge: 1-node, Intel Xeon Platinum 8375C processor with 128 GB memory (1 x 128 GB DDR4 3200 MT/s), microcode 0xd0003a5, hyperthreading on, turbo on, Ubuntu 22.04.2 LTS, kernel 6.2.0-1011-aws, gcc 11.3.0, Amazon Elastic Block Store 200 GB, BIOS Amazon EC2 1.010/16/2017; Software: [PyTorch 2.1.0_rc4](https://github.com/pytorch/pytorch/tree/release/2.1), [oneDNN version 3.1.1](https://github.com/oneapi-src/oneDNN/tree/v3.1.1), [TorchBench](https://github.com/pytorch/benchmark/commit/ffbbebb9), [TorchVision](https://github.com/pytorch/vision/commit/8636bf3), [TorchText](https://github.com/pytorch/text/commit/142d029), [TorchAudio](https://github.com/pytorch/audio/commit/475b6ae), [TorchData](https://github.com/pytorch/data/commit/eb9bf61), [TorchDynamo Benchmarks](https://github.com/pytorch/pytorch/tree/release/2.1/benchmarks/dynamo), [TorchBench cpu userbenchmark](https://github.com/pytorch/benchmark/tree/chuanqiw/inductor_quant/userbenchmark/cpu), tested by Intel on 9/12/2023.
