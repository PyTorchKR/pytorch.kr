---
layout: blog_detail
title: "PyTorch Compile은 왜 이렇게 빠를까: 커널 융합(Kernel Fusion)"
author: Morrison Turnansky
ext_author: Junghwan Park (박정환)
category: ["pytorch.org", "translation"]
date: 2026-05-27 12:00:00
org_title: "Why Is PyTorch Compile So Fast: Kernel Fusion"
org_link: https://pytorch.org/blog/why-is-pytorch-compile-so-fast-kernel-fusion/
---

PyTorch의 컴파일러를 사용하면 모델이 최대 10배까지 빠르게 실행됩니다. 그런데 실제로 무슨 일이 일어나는 걸까요? 컴파일을 하지 않으면 GPU는 코드에 있는 각 torch 연산마다 커널(kernel), 즉 GPU에서 동작하는 함수를 실행합니다. 이로 인해 두 가지 큰 속도 저하가 발생합니다. 메모리에서 데이터를 옮기는 데 드는 시간과, 매번 새로운 커널을 시작할 때 드는 오버헤드입니다. GPU가 커널을 실행할 때마다 오버헤드 비용을 치르며, 중간 결과가 생길 때마다 메모리에 쓰고 다시 읽어야 합니다.
> When you use PyTorch's compiler, your model runs faster, up to 10x faster. But what's actually happening? Without compilation, the GPU runs a kernel, a function on the GPU, for each torch operation in your code. This creates two big slowdowns: the time spent moving data in memory, and the overhead of starting each new kernel. Every time the GPU launches a kernel, it pays an overhead cost, and every intermediate result means writing to and reading from memory.

바로 이 지점에서 융합(fusion)이 등장합니다. PyTorch의 Inductor 컴파일러는 서로 의존하는 연산들을 자동으로 묶어 단일하고 효율적인 Triton 커널로 만듭니다. 이렇게 하면 데이터를 레지스터(register)에 가까운 더 빠른 메모리에 유지하면서 커널 오버헤드를 줄일 수 있습니다. 이번 글에서는 융합의 구체적인 예시를 살펴본 뒤, 더 깊이 읽어볼 만한 주제들을 정리합니다. `torch.compile`이 여러분의 PyTorch 연산을 어떻게 최적화된 GPU 코드로 변환하는지 정확하게 확인할 수 있습니다.
> This is where fusion comes in. PyTorch's Inductor compiler automatically groups dependent operations together into single, efficient Triton kernels. This keeps data in faster memory close to the register and cuts down on kernel overhead. In this article, we'll look at a concrete example of fusion, and then outline topics for further reading. You'll see exactly how torch.compile transforms your PyTorch operations into optimized GPU code.

이번 글을 최대한 잘 활용하려면 PyTorch에 대한 기본적인 이해와 GPU 프로그래밍 개념에 대한 전반적인 이해가 있으면 좋습니다.
> To get the most out of this article, you should have basic familiarity with PyTorch and a general understanding of GPU programming concepts.

## 수직 융합이란 무엇인가? / What is Vertical Fusion?

수직 융합(vertical fusion)은 여러 단계를 "연결"하여 한 단계의 출력이 곧바로 다음 단계로 들어가게 하는 방법이라고 생각하면 됩니다. 이를 "수직(vertical)"이라고 부르는 이유는, 계산 그래프를 떠올려 보면 이러한 연산들이 수직으로 쌓이기 때문입니다. 각 연산이 이전 단계의 결과에 의존하는 구조입니다.
> Think of vertical fusion as a way to "link" steps, so the output of one goes straight into the next. It's called "vertical" because if you picture the computation graph, these operations stack vertically – each one depends on the result of the previous step.

이것은 딥러닝에서 가장 흔한 융합 패턴입니다. 신경망은 정규화, 그다음 선형 계층, 그다음 활성화 함수처럼 연산들이 사슬처럼 이어진 구조이기 때문입니다. 가장 큰 이점은 중간 결과를 없앨 수 있다는 것입니다. 이러한 임시 Tensor들은 전역 메모리(global memory)에 쓰거나 읽을 필요가 전혀 없습니다. GPU가 더 빠르게 접근할 수 있는 빠른 레지스터에 그대로 머무릅니다.
> This is the most common fusion pattern in deep learning because neural networks are chains of operations: normalization, then linear layers, then activation functions, and so on. The big win is eliminating intermediate results – those temporary tensors never need to be written to or read from global memory. They stay in fast registers where the GPU can reach them more quickly.

수직 융합의 한 예시인 점별 융합(pointwise fusion)을 자세히 살펴보겠습니다.
> Let's dive into an example of vertical fusion, namely pointwise fusion.

## 점별 융합 예시 / Pointwise Fusion Example

점별(pointwise) 연산은 각 요소에 대해 동작하는 단순한 수학 커널로, 덧셈, 곱셈, 활성화 함수 등이 여기에 해당합니다. 신경망 계층에서 볼 수 있는 패턴 하나를 살펴보겠습니다.
> Pointwise operations are simple math kernels that work on each element: addition, multiplication, activation functions, and more. Let's look at a pattern you might see in a neural network layer:

*점별 PyTorch 예시 / Pointwise PyTorch Example*

```python
import torch

def pointwise_example(x, w, b):
    # 여러 요소별(element-wise) 연산
    tmp = x * w        # 곱셈
    tmp = tmp + b      # 덧셈
    tmp = tmp.sigmoid() # 시그모이드 활성화
    return tmp
```

### 융합하지 않은 경우: 세 개의 분리된 커널 / Unfused: Three Separate kernels

융합을 하지 않으면 Inductor는 세 개의 분리된 Triton 커널을 만듭니다. Triton 문법이 어렵게 보이더라도 걱정하지 마세요. 중요한 것은 문법을 외우는 것이 아니라 패턴을 이해하는 것입니다. 각 커널은 데이터를 불러오고, 하나의 연산을 수행하고, 결과를 씁니다.
> Without fusion, Inductor creates three separate Triton kernels. Don't worry if the Triton syntax looks intimidating. The important part isn't memorizing the syntax, but understanding the pattern: each kernel loads data, does one operation, and writes the result.

*커널 1: 곱셈 / Kernel 1: Multiply*

```python
@triton.jit
def mul_kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)
```

간결함을 위해 다음 커널들은 거의 동일하므로 시그니처(signature)만 포함합니다. 전체 소스 코드는 [Git 저장소](https://gist.github.com/morrison-turnansky/0cc51b498c674aa23d4718ae200e6209)를 참고하세요.
> For succinctness, we include just the signatures of the next kernels as they are nearly identical, see our [Git Repository](https://gist.github.com/morrison-turnansky/0cc51b498c674aa23d4718ae200e6209) for the full source code.

*커널 2: 덧셈 / Kernel 2: Add*

```python
@triton.jit
def add_kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr)
```

*커널 3: 시그모이드 / Kernel 3: Sigmoid*

```python
@triton.jit
def sigmoid_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr)
```

세 커널에 걸쳐 총 여덟 번의 메모리 연산이 수행됩니다. 곱셈을 위해 입력을 두 번 읽고, 덧셈을 위해 곱셈 결과와 편향(bias)을 읽고, 시그모이드를 위해 덧셈 결과를 읽고, 세 결과를 모두 씁니다. 메모리 트래픽이 상당히 많습니다.
> Across the three kernels you're performing eight memory operations: reading inputs twice for multiply, reading multiply's result and the bias for add, reading add's result for sigmoid, and writing all three results. That's a lot of memory traffic.

### 융합한 경우: 하나의 커널 / Fused: One Kernel

융합을 하면 `torch.compile`은 단일 커널을 만듭니다.
> With fusion, torch.compile creates a single kernel:

*커널 4: 융합된 커널 / Kernel 4: Fused*

```python
@triton.jit
def triton_poi_fused_add_mul_sigmoid_0(in_ptr0, in_ptr1, in_ptr2,
                                        out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # 모든 입력을 한 번에 불러오기
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (x0), xmask)

    # 융합된 점별 연산: mul -> add -> sigmoid
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)

    # 최종 결과만 저장
    tl.store(out_ptr0 + (x0), tmp5, xmask)
```

차이를 눈여겨 보세요. 모든 입력을 한 번만 불러오고, 세 연산을 연달아 수행한 뒤, 최종 결과만 저장합니다. 중간 값(`tmp2`와 `tmp4`)은 GPU에서 가장 빠른 메모리인 레지스터에 그대로 머무릅니다. 더 느린 전역 메모리에는 전혀 닿지 않습니다.
> Notice the difference: we load all inputs once, do all three operations in a row, and store only the final result. The intermediate values (`tmp2` and `tmp4`) stay in registers – the fastest memory on the GPU. They never touch the slower global memory.

### 이점 / Benefits

- **커널 실행(Kernel launches)**: 3회에서 1회로 감소
- **중간 버퍼(Intermediate buffers)**: 2개 제거 (곱셈 결과와 덧셈 결과)
- **메모리 대역폭(Memory bandwidth)**: 전체 Tensor 5개를 읽고 3개를 쓰던 것(메모리 연산 8회)에서 Tensor 3개를 읽고 1개를 쓰는 것(메모리 연산 4회)으로 감소 — 메모리 트래픽 50% 절감

> -   **Kernel launches**: 3 reduced to 1
> -   **Intermediate buffers**: 2 eliminated (multiply result and add result)
> -   **Memory bandwidth**: Reading 5 full tensors and writing 3 full tensors (8 memory operations) reduced to reading 3 tensors and writing 1 (4 memory operations) – a 50% reduction in memory traffic

## 그 밖의 융합 유형 / Other Fusion Types

점별 융합은 수직 융합의 한 유형일 뿐입니다. Inductor는 GPU를 효율적으로 유지하기 위해 다른 형태의 수직 융합도 사용합니다.
> Pointwise fusion is just one type of vertical fusion. Inductor uses other forms of vertical fusion to keep your GPU efficient:

**리덕션 융합(Reduction Fusion)**: max, mean, sum 같은 리덕션(reduction) 연산을 그 앞뒤에서 일어나는 연산들과 결합합니다. 이는 배치 정규화(batch normalization) 같은 연산에서 매우 중요합니다.
> **Reduction Fusion**: Combines reducing operations like max, mean, or sum, with the operations that happen before and after them. This is critical for operations like batch normalization.

**GEMM + 에필로그 융합(GEMM + Epilogue Fusion)**: 무거운 행렬 계산의 끝에 간단한 연산을 붙입니다. 행렬 곱을 한 뒤 결과를 메모리에 쓰고 다시 읽어 편향을 더하고 ReLU를 적용하는 대신, 편향과 활성화가 곱셈 직후 같은 커널 안에서 일어납니다.
> **GEMM + Epilogue Fusion**: Attaches simple math to the end of heavy matrix calculations. Instead of doing a matrix multiply, writing the result to memory, then reading it back to add bias and apply ReLU, the bias and activation happen right after the multiply in the same kernel.

**프롤로그 융합(Prologue Fusion)**: 에필로그의 반대로, 데이터를 불러오는 동안 전처리가 일어납니다. 예를 들어 행렬 곱 전에 입력을 정규화하는 작업을 데이터가 들어오는 즉시 즉석에서 처리할 수 있습니다.
> **Prologue Fusion**: The opposite of epilogue – preprocessing happens as data loads. For instance, normalizing input before matrix multiplication can happen on-the-fly as the data comes in.

가장 대표적인 융합 유형인 수직 융합 외에도, Inductor는 수평 융합(horizontal fusion)도 사용합니다.
> In addition to vertical fusion, the most prominent type of fusion, Inductor also uses horizontal fusion.

**수평 융합(Horizontal Fusion)**: 동일한 입력에 대해 서로 독립적인 여러 연산을 한 번에 실행합니다. 예를 들어 `sin(x)`와 `cos(x)`를 하나의 커널에서 계산하면, `x`를 두 번이 아니라 한 번만 불러옵니다.
> **Horizontal Fusion**: Runs multiple independent operations on the same input at once. For example, computing both `sin(x)` and `cos(x)` in a single kernel, loading `x` only once instead of twice.

## 시작하기: 내 코드에서 융합 확인하기 / Get Started: See Fusion in Your Own Code

리덕션 패턴을 사용하는 완전한 예시를 단계별로 살펴보겠습니다.
> Let's walk through a complete example using a reduction pattern.

### 1단계: 간단한 리덕션 예시 만들기 / Step 1: Create a Simple Reduction Example

`fusion_example.py` 라는 파일을 만듭니다.
> Create a file called `fusion_example.py`:

```python
import torch

def reduction_example(x):
    # 점별 연산 다음에 리덕션
    tmp = x * 2.0
    result = tmp.sum(dim=-1)
    result = result + 1.0
    return result

# 테스트 입력 생성
x = torch.randn(1024, 1024, device='cuda')

compiled_fn = torch.compile(reduction_example)
result_fused = compiled_fn(x)
```

### 2단계: 생성된 코드 보기 / Step 2: View the Generated Code

`TORCH_LOGS` 환경 변수를 지정하여 스크립트를 실행하면 Inductor가 생성한 코드를 확인할 수 있습니다.
> Run your script with the `TORCH_LOGS` environment variable to see what Inductor generated:

```sh
TORCH_LOGS="output_code" python fusion_example.py
```

이렇게 하면 생성된 Triton 커널이 터미널에 출력됩니다. `triton_per_fused_add_mul_sum_0` 와 같은 이름의 커널을 찾아보세요. `per` 접두사는 "리덕션별(per-reduction)" 커널을 의미하며, 이름을 보면 add, mul, sum이 모두 함께 융합되었음을 알 수 있습니다.
> This outputs the generated Triton kernels to your terminal. Look for a kernel named something like `triton_per_fused_add_mul_sum_0`. The `per` prefix means "per-reduction" kernel, and the name tells you that add, mul, and sum were all fused together.

## 맺음말 / Conclusion

융합은 `torch.compile`이 수행하는 가장 중요한 최적화 중 하나입니다. 서로 의존하는 연산들을 단일 커널로 연결함으로써, GPU 작업에서 주된 속도 저하 요인인 메모리 트래픽과 커널 오버헤드를 줄입니다.
> Fusion is one of the most important optimizations that torch.compile does. By linking dependent operations into single kernels, it cuts down memory traffic and kernel overhead – often the main slowdowns in GPU work.

torch compile로 여러분의 코드를 직접 가속해 보세요. 구현을 바꿀 필요 없이, torch 컴파일러 데코레이터(decorator)만 추가하면 컴파일러가 알아서 처리해 줍니다.
> Try accelerating your own code with torch compile. No need to change your implementation, just add a torch compiler decorator and let the compiler do the work.

**더 알아보기**: PyTorch 문서 [pytorch.org/docs/stable/torch.compiler.html](http://pytorch.org/docs/stable/torch.compiler.html) 에는 컴파일과 최적화 전략에 대한 완전한 가이드가 있습니다. 전체 소스 코드는 [Git 저장소](https://gist.github.com/morrison-turnansky/0cc51b498c674aa23d4718ae200e6209)를 참고하세요.
> **Learn more**: PyTorch documentation at [pytorch.org/docs/stable/torch.compiler.html](http://pytorch.org/docs/stable/torch.compiler.html) has complete guides on compilation and optimization strategies. Reference our [Git Repository](https://gist.github.com/morrison-turnansky/0cc51b498c674aa23d4718ae200e6209) for the full source code.
