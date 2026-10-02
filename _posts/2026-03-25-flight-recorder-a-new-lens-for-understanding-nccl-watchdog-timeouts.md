---
layout: blog_detail
title: "Flight Recorder: NCCL 워치독 타임아웃을 이해하는 새로운 렌즈"
author: Phillip Liu, Uttam Thakore, Junjie Wang, Justin Yang
ext_author: Junghwan Park (박정환)
category: ["pytorch.org", "translation"]
date: 2026-03-25 12:00:00
org_title: "Flight Recorder: A New Lens for Understanding NCCL Watchdog Timeouts"
org_link: https://pytorch.org/blog/flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/
---

![Flight Recorder: NCCL 워치독 타임아웃을 이해하는 새로운 렌즈 대표 이미지 / Flight Recorder: A New Lens for Understanding NCCL Watchdog Timeouts](/assets/blog/2026-03-25-flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/hero.png){:style="width:100%"}

대규모 AI 모델을 학습시켜 본 적이 있고 다음과 같은 오류로 실패한 경험이 있다면:
> If you've ever trained a large AI model and had it fail with an error like:

```
[Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=12345, 
OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600029
milliseconds before timing out.
Exception raised from checkTimeout at .../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:692 (most recent call 
first):

...

# 2  c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >)
# 3  c10d::ProcessGroupNCCL::Watchdog::runLoop()
# 4  c10d::ProcessGroupNCCL::Watchdog::run()
# 5  execute_native_thread_routine
# 6  start_thread
# 7  __clone3
```

그 악명 높은 NCCL 워치독 타임아웃(watchdog timeout)을 만난 것입니다. 이 오류는 디버깅하기 어렵습니다 — 오류 메시지가 두루뭉술하고, 디버깅에는 랭크(rank) 간 텔레메트리(telemetry) 분석이 필요하며, 근본 원인이 여러 계층에 걸쳐 있어 인과 관계가 복잡할 수 있습니다.
> You've encountered the infamous NCCL watchdog timeout. Debugging this error can be hard – the error message is generic, debugging requires cross-rank telemetry analysis, and root causes are multi-layered and can have a complex causal chain.

이번 글에서는 NCCL 워치독 타임아웃에 대한 핵심적인 통찰을 다음과 같이 다룹니다:

- 이 오류가 왜 발생하고 왜 그토록 디버깅하기 어려운지
- 이 오류의 가장 흔한 근본 원인(예: CPU 측 실행 분기(divergence), GPU 멈춤(hang), 잘못 설정된 집합 통신)에 대한 심층 분석
- PyTorch Flight Recorder로 무엇이 잘못됐는지 빠르게 짚어내고 고치는 방법, 그리고 Meta 내부에서 이를 어떻게 활용하는지에 대한 이야기

> This post provides key insights on NCCL watchdog timeouts, including:
> - Why this error happens and why it's so hard to debug;
> - A deep dive into the most common root causes for the error (e.g., CPU-side divergence, GPU hang, misconfigured collectives);
> - How to use PyTorch Flight Recorder to quickly pinpoint what went wrong and how to fix it, including insights about how it is used within Meta.

이 글을 다 읽고 나면 NCCL 워치독 타임아웃을 효율적으로, 더 확신을 갖고 진단하고 해결하는 데 필요한 지식과 실전 도구를 갖추게 될 것입니다.
> By the end of this post, you should be equipped with the knowledge and practical tools needed to diagnose and resolve NCCL watchdog timeouts efficiently and with greater confidence.

## 들어가며: PyTorch의 집합 통신이란? / Intro: What are collectives in PyTorch?

먼저 PyTorch가 분산 환경에서 어떻게 동작하는지 살펴봅시다.
> Let's begin by examining how PyTorch functions in a distributed setting.

단일 랭크(즉, 단일 GPU) 학습 시대에는 사용자가 `torch.matmul(tensor1, tensor2)` 같은 텐서 연산을 호출하면, 그 요청이 C++ 디스패처(dispatcher)를 거쳐 최종적으로 백엔드 커널(예: GPU용 CUDA 커널, CPU용 C++ 구현)을 호출했습니다. 하지만 분산 학습에서는 낙오(straggling)를 피하거나 랭크 간에 계산 결과를 공유하기 위해, 특정 텐서 연산 뒤에 동기화를 수행해야 합니다. 예를 들어 `dist.all_reduce(tensor1)` 을 호출하면 분산 환경 전체에 있는 `tensor1` 의 모든 인스턴스를 합한 뒤 그 결과를 다시 `tensor1` 에 할당합니다. 이런 종류의 동기화를 **집합 통신(collective)** 연산이라고 합니다. 집합 통신은 동기화가 필요한 랭크들의 집단을 나타내는 [프로세스 그룹(process group)](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)에서 실행됩니다. 집합 통신의 예로는 [DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)에서 쓰이는 `all-reduce`, [FSDP](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)에서 쓰이는 `all-gather`/`reduce-scatter`, [TorchRec](https://docs.pytorch.org/tutorials/intermediate/torchrec_intro_tutorial.html)에서 쓰이는 `all-to-all` 등이 있습니다.
> In the era of single-rank (i.e., single-GPU) training, when a user calls a tensor operation, such as `torch.matmul(tensor1, tensor2)`, the request goes through the C++ dispatcher and eventually invokes backend kernels (e.g., CUDA kernels for GPU or C++ implementation for CPU). However, in distributed training users must perform synchronization after certain tensor operations to either avoid straggling or share computational results across ranks. For instance, a user might call `dist.all_reduce(tensor1)`, which sums up all instances of `tensor1` across the distributed setup and assigns the result back to `tensor1`. This type of synchronization is referred to as a "collective" operation. Collectives are executed in a "[process group](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)", which represents a cohort of ranks that require synchronization. Examples of collectives include `all-reduce` (used in [DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)), `all-gather/reduce-scatter` (used in [FSDP](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)), and `all-to-all` (used in [TorchRec](https://docs.pytorch.org/tutorials/intermediate/torchrec_intro_tutorial.html)).

이제 사용자가 `dist.all_reduce(tensor1)` 을 호출할 때 무슨 일이 벌어지는지 더 자세히 살펴봅시다. 이 호출은 C++ PyTorch 디스패처와 PyBind 계층을 모두 통과하고, PyBind 계층이 최종적으로 PyTorch c10d 계층을 호출합니다. c10d 계층에서 PyTorch는 여러 통신 백엔드 라이브러리의 집합 통신 API — GPU 통신용 [NCCL API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html), CPU 통신용 [Gloo](https://github.com/pytorch/gloo?tab=readme-ov-file) 등 — 를 호출합니다. *이후 논의는 주로 이 c10d 계층에 집중합니다.*
> Now, let's take a closer look at what happens when a user calls `dist.all_reduce(tensor1)`. The call passes through both the C++ PyTorch dispatcher and a PyBind layer, where the latter ultimately calls into the PyTorch c10d layer. Within the c10d layer, PyTorch then invokes the collective communication APIs from various communication backend libraries, such as [NCCL APIs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) for GPU communication and [Gloo](https://github.com/pytorch/gloo?tab=readme-ov-file) for CPU communication. *The remainder of this discussion will primarily focus on this c10d layer.*

그러면 이런 질문이 이어질 수 있습니다: PyTorch에는 왜 이 c10d 계층이 필요할까요? 사용자가 NCCL API를 직접 호출하게 두면 안 될까요? 답은 PyTorch가 요청을 통신 라이브러리에 넘기기 전에 더 많은 제어 정보를 필요로 한다는 것입니다. c10d 계층은 몇 가지 기능을 제공하는데, 이번 글의 맥락에서는 그중 가장 두드러진 기능인 **PyTorch c10d 워치독** 에 집중합니다. 오늘날 대부분의 학습이 GPU에서 이뤄지므로 여기서는 논의를 NCCL로 더 좁혀 이 워치독을 아래에서 *NCCL 워치독* 이라고 부르겠습니다. 다만 NCCL은 NVIDIA GPU에 한정되지만, c10d 워치독 메커니즘 자체는 어떤 분산 백엔드든 모니터링할 수 있다는 점을 기억해 두세요.
> A subsequent question can be: Why does PyTorch need this c10d layer? Why not let users call into the NCCL API directly? The answer is that PyTorch needs more control information before handing the request over to the communication library. The c10d layer provides a couple of features; in the context of this post, we focus on the most notable one: the **PyTorch c10d watchdog**. Since most modern training occurs on GPUs, we further narrow our discussion here to NCCL only and refer to this watchdog below as the *NCCL watchdog*, but note that while NCCL is specific to Nvidia GPUs, the c10d watchdog mechanism is capable of monitoring any distributed backend.

## 문제 정의: NCCL 워치독 타임아웃 오류 / Problem statement: The NCCL watchdog timeout error

### NCCL 워치독 타임아웃이란? / What is the NCCL watchdog timeout?

집합 통신 사용에 버그나 장애가 없는 이상적인 상황이라면, 집합 통신은 CPU 측에서 스케줄링되고 GPU에서 비동기로 실행된 뒤 그 결과가 이후 사용을 위해 반환됩니다. 하지만 NCCL API 자체에는 내장된 오류 검사가 전혀 없습니다. 집합 통신이 "잘못 사용"되면(유효하지 않은 인자로 호출하거나, 랭크마다 다른 집합 통신을 호출하는 경우 등) 그 집합 통신은 GPU에서 무한정 멈춰 있게 됩니다.
> In an ideal scenario with no bugs or failures in collective usage, collectives are scheduled on the CPU side, executed asynchronously on the GPU, and the result is returned for subsequent use. However, the NCCL API itself does not provide any built-in error checking. If a collective is "misused" (including calling with invalid arguments, calling different collective across ranks, etc.), the collective execution will hang indefinitely on the GPU.

이 멈춤을 감지하기 위해 PyTorch는 c10d 계층에 CPU 측 [`Work`](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp#L313) 객체와 NCCL 워치독 모니터링 스레드를 도입했습니다. `Work` 객체는 NCCL API 호출을 [앞](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L3498)과 [뒤](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L3509)에서 두 개의 `CudaEvent` 로 감싸 GPU에서 집합 통신의 생애주기를 추적합니다. 이 메커니즘 덕분에 PyTorch NCCL 워치독은 집합 통신의 상태를 주기적으로 폴링해, 사용자가 정한 타임아웃(기본값 10분) 안에 GPU에서 완료되는지 확인할 수 있습니다. 타임아웃을 넘기면 NCCL 워치독은 예외를 던져 학습 프로세스를 중단시키며, 이 예외가 바로 잘 알려진 "NCCL 워치독 타임아웃" 오류이고 흔히 "NCCL 워치독 타임아웃" 오류라고도 불립니다. (이렇게 줄여 부르는 것이 혼동을 일으키는 경우가 많은데, "NCCL 워치독 타임아웃" 은 NCCL 라이브러리가 아니라 *PyTorch NCCL 워치독* 이 발생시키기 때문입니다. 오류 이름에 "NCCL" 이 들어 있는 것은 NCCL 백엔드로 집합 통신을 실행하는 중에 타임아웃이 발생하기 때문입니다.)
> To detect this hang, PyTorch introduces a CPU-side [Work](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp#L313) object and a NCCL watchdog monitoring thread on the c10d layer. The `Work` object tracks the collective's lifecycle on the GPU by wrapping the NCCL API calls with two `CudaEvent`s [before](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L3498) and [after](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L3509). This mechanism enables the PyTorch NCCL watchdog to periodically poll the status of a collective and check if it is completed on the GPU within a user-defined timeout (default is 10 minutes). If the timeout is exceeded, the NCCL watchdog throws an exception to interrupt the training process – this exception is the well-known "NCCL watchdog timeout" error, also commonly referred to as the "NCCL watchdog timeout" error. (Note that this abbreviation often causes confusion, as the "NCCL watchdog timeout" is *not* raised from the NCCL library *but from the PyTorch NCCL watchdog*. The error name contains "NCCL" because the timeout occurs while the collective is launched with the NCCL backend.)

아래 그림 1은 NCCL 워치독 타임아웃을 유발하는 사건의 순서를 보여줍니다:
> Figure 1 below shows the sequence of events that triggers a NCCL watchdog timeout:

![PyTorch가 NCCL 집합 통신을 모니터링하는 방식 / How PyTorch monitors NCCL collectives](/assets/blog/2026-03-25-flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/unnamed-4-1.png){:style="width:100%"}

*그림 1: PyTorch가 NCCL 집합 통신을 모니터링하는 방식을 보여주는 시퀀스 다이어그램 / Figure 1: Sequence diagram showing how PyTorch monitors NCCL collectives*

### NCCL 워치독 타임아웃은 왜 디버깅하기 어려운가? / Why is the NCCL watchdog timeout hard to debug?

NCCL 워치독 타임아웃 오류가 디버깅하기 어려운 데는 두 가지 큰 이유가 있습니다.
> NCCL watchdog timeout errors are hard to debug for two major reasons.

첫째, NCCL 워치독 타임아웃은 이른바 "포괄(catch-all)" 오류입니다 — 어떤 랭크가 집합 통신에서 무한정 기다리게 만드는 모든 것이 NCCL 워치독 타임아웃으로 이어질 수 있으며, *집합 통신이 느리거나 네트워크/NCCL 라이브러리에 문제가 있는 경우만이 아닙니다*. 다음 절에서 설명하듯 여기에는 CPU 측 멈춤, GPU 멈춤, CUDA 교착 상태(deadlock), 유효하지 않은 집합 통신 인자, 하드웨어/네트워크 문제 등이 포함됩니다.
> First, the NCCL watchdog timeout is what we call a "catch-all" error – anything that can cause a rank to wait indefinitely on a collective can lead to a NCCL watchdog timeout, *not just collective slowness or network/NCCL library issues*. As we describe in the next section, this includes CPU-side hang, GPU hang, CUDA deadlock, invalid collective arguments, hardware/network issues, etc.

둘째, 오류 메시지가 거의 도움이 되지 않습니다. 오류가 워치독 스레드에서 발생하기 때문에, 워치독 스레드 자체가 멈출 가능성을 줄이려고 집합 통신에 대한 제한된 메타데이터만 오류 메시지에 기록합니다. 집합 통신이 스케줄링된 PyTorch 메인 스레드의 호출 스택(call stack)처럼 결정적인 디버깅 정보는 오류 메시지에서 빠져 있습니다(오류 메시지에 담긴 스택 트레이스는 NCCL 워치독 스레드의 것이라 진단에는 무관합니다).
> Second, the error message is largely unhelpful. Since the error is raised from the watchdog thread, to reduce the potential for the watchdog thread to itself hang, only limited metadata about the collective is logged in the error message. Crucial debugging information, such as the call stack on the PyTorch main thread from which the collective was scheduled, is missing from the error message (the stack trace in the error message is from the NCCL watchdog thread, which is diagnostically irrelevant).

게다가 다음 절에서 설명하듯:

- NCCL 워치독 타임아웃을 가장 먼저 일으킨 랭크가 타임아웃을 유발한 범인인 경우는 드물고,
- NCCL 워치독 타임아웃 시점에 실행 중이던 집합 통신이 타임아웃의 원인이 아닐 수도 있습니다.

> Furthermore, as we explain in the next section:
> - The rank that first raises the NCCL watchdog timeout is rarely the culprit that caused the timeout, and
> - The collective being executed at the time of NCCL watchdog timeout may not be the cause of the timeout.

그래서 PyTorch Flight Recorder 없이 NCCL 워치독 타임아웃 오류를 디버깅하면 몇 시간 이상 걸릴 수 있고, 추가 디버깅 플래그(예: `CUDA_LAUNCH_BLOCKING`)를 켠 채 작업을 다시 실행해야 하는 경우도 잦습니다.
> Consequently, debugging NCCL watchdog timeout errors without the use of PyTorch Flight Recorder can take hours or longer, often necessitating job reruns with additional debugging flags enabled (e.g., `CUDA_LAUNCH_BLOCKING`).

## 심층 분석: NCCL 집합 통신은 왜 타임아웃되는가? / Deep dive: What causes NCCL collectives to time out?

NCCL 워치독 타임아웃의 원인을 설명하려면, 먼저 PyTorch가 분산 환경에서 ([Monarch](https://pytorch.org/blog/introducing-pytorch-monarch/) 같은 중앙 집중식 단일 컨트롤러가 없을 때) 집합 통신을 어떻게 스케줄링하고 실행하는지 설명해야 합니다.
> To explain the causes of NCCL watchdog timeouts, we first need to describe how PyTorch schedules and executes collectives in a distributed setting (absent a centralized single controller like [Monarch](https://pytorch.org/blog/introducing-pytorch-monarch/)).

앞서 말했듯 PyTorch NCCL 집합 통신은 CPU가 스케줄링하고 GPU에서 비동기로 실행됩니다. 대부분의 학습 프레임워크에서 학습 자원이 효율적으로 쓰이고 있다면 워크로드는 GPU에 의해 제한되므로(GPU-bound), CPU는 GPU 연산 커널(PyTorch 연산)과 NCCL 집합 통신을 차례로 GPU에 스케줄링한 뒤, 다음 CPU-GPU 동기화 지점에서 GPU가 스케줄링된 커널을 모두 끝낼 때까지 간헐적으로 블로킹 대기합니다. 그림 2는 2개 랭크로 이뤄진 프로세스 그룹에서 이 정상 경로(happy path) 동작을 보여줍니다.
> As mentioned above, PyTorch NCCL collectives are scheduled by the CPU and executed asynchronously on GPU. In most training frameworks, when training resources are being used efficiently, the workload is GPU-bound, so the CPU will schedule a sequence of GPU compute kernels (PyTorch operations) and NCCL collectives onto the GPU, then intermittently blocking wait for the GPU to complete all the scheduled kernels at the next CPU-GPU synchronization point. Figure 2 illustrates this happy path behavior for a 2-rank process group.

![2개 랭크 프로세스 그룹에서 CUDA 커널과 NCCL 집합 통신이 실행되는 과정 / How CUDA kernels and NCCL collectives are launched and executed in a 2-rank process group](/assets/blog/2026-03-25-flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/unnamed-5-1.png){:style="width:100%"}

*그림 2: 2개 랭크 프로세스 그룹에서 CUDA 커널과 NCCL 집합 통신이 어떻게 실행 요청되고 실행되는지를 보여주는 시퀀스 다이어그램. GPU에서 커널 실행 사이의 간격은 읽기 쉽도록 과장했습니다. / Figure 2: Sequence diagram for how CUDA kernels and NCCL collectives are launched and executed in a 2-rank process group. Gaps between kernel executions on the GPU are exaggerated for readability.*

이 과정에서 GPU 상의 동기 연산은 두 종류입니다:

1. **랭크 간 GPU-GPU 동기화**(즉, NCCL 집합 통신). 프로세스 그룹의 모든 GPU가 다른 커널을 계속 실행하기 전에 서로의 상태를 동기화합니다. 모든 NCCL 집합 통신이 배리어(barrier)로 동작하는 것은 아니지만, 배리어 성격의 집합 통신(예: `all_reduce`)이 NCCL 워치독 타임아웃의 가장 흔한 원인입니다. (PyTorch NCCL 프로세스 그룹 초기화/종료처럼 랭크 간 GPU 동기화가 필요한 다른 연산도 있지만, 이번 글의 범위를 벗어납니다.)
2. **단일 랭크 CPU-GPU 동기화**. 특정 GPU에 연결된 CPU 스레드가 그 GPU에 스케줄링된 모든 연산(연산 커널과 통신 커널 포함)이 끝날 때까지 블로킹됩니다. 이 CPU-GPU 동기화는 명시적일 수도 있고(예: [torch.cuda.synchronize](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html) 호출) 암묵적일 수도 있습니다(예: 텐서를 CPU에서 GPU로, 또는 그 반대로 옮기기).

> During this process, there are two types of synchronous operations on the GPU:
> 1. Inter-rank GPU-GPU synchronization (i.e., NCCL collectives), in which all GPUs in a process group synchronize their state before continuing with other kernels. Not all NCCL collectives function as barriers, but barrier collectives (e.g., all\_reduce) are the most common source of NCCL watchdog timeouts. (There are some other operations such as PyTorch NCCL process group initialization/teardown that require inter-rank GPU sync, but they are beyond the scope of this blog.)
> 2. Single-rank CPU-GPU synchronization, in which the CPU thread associated with a given GPU blocks until the GPU finishes all of its scheduled operations (including compute kernels and communication kernels). This CPU-GPU sync can either be explicit (e.g. a call to [torch.cuda.synchronize](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html)) or implicit (e.g. moving a tensor from CPU to GPU or vice versa).

집합 통신이 타임아웃되는 경로는 두 가지입니다: 1) 집합 통신 커널 자체가 타임아웃보다 오래 걸리거나 멈추는 경우, 또는 2) 랭크 간 동기화가 깨져서(desynchronize) 타임아웃 시점에 랭크마다 집합 통신 메타데이터나 상태가 달라지는 경우입니다.
> There are two ways a collective can time out: either 1) the collective kernel itself takes longer than the timeout to execute or hangs, or 2) ranks desynchronize, causing the collective metadata or state to differ across ranks at timeout.

Meta의 서버 플릿(fleet) 전반에서 NCCL 워치독 타임아웃을 디버깅해 온 경험에 따르면, **거의 모든 NCCL 워치독 타임아웃 오류는 집합 통신의 비동기화(desync, 즉 불일치)에서 비롯되며, 집합 통신이 느리거나 멈춰서가 *아닙니다*.** 즉 타임아웃 값을 늘리는 것으로는 문제가 해결되지 않으며, 비동기화의 원인을 반드시 해결해야 합니다.
> Based on our experience debugging NCCL watchdog timeouts across Meta's fleet, **almost all NCCL watchdog timeout errors are caused by collective desync (i.e., mismatch),** ***not*** **collective slowness/hang**. This means that increasing the timeout value will not fix the issue – the cause of the desync must be resolved.

그림 3은 Meta 플릿에서 4개의 서로 다른 학습 프레임워크(3개는 추천 시스템(RecSys)용, 1개는 LLM용)를 대상으로 관측한 NCCL 워치독 타임아웃의 가장 흔한 근본 원인 분포를 보여줍니다. 이 절의 나머지에서는 관측된 4개의 큰 원인 범주 — CPU 측 문제, GPU 연산 커널 멈춤, 잘못 설정된 NCCL 집합 통신 인자, 네트워크/하드웨어 문제 — 를 설명합니다.
> Figure 3 shows the breakdown of the most common root causes for NCCL watchdog timeouts that we have observed within Meta's fleet for 4 different training frameworks (three for recommendation systems, i.e., RecSys, and one for LLMs). In the rest of this section, we explain the 4 broad categories of root causes that we have observed: CPU-side issues, GPU compute kernel hang, misconfigured NCCL collective arguments, and network/hardware issues.

![Meta 내부 여러 학습 스택에서 관측된 NCCL 워치독 타임아웃 근본 원인 분포 / Observed breakdown of NCCL watchdog timeout root causes across various training stacks within Meta](/assets/blog/2026-03-25-flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/unnamed-6-1.png){:style="width:100%"}

*그림 3: Meta 내부 여러 학습 스택에서 관측된 NCCL 워치독 타임아웃 근본 원인 분포 / Figure 3: Observed breakdown of NCCL watchdog timeout root causes across various training stacks within Meta*

### 범주 1: CPU 측 문제 / Category 1: CPU-side issues

(중앙 집중식 단일 컨트롤러가 없는) 오늘날 대부분의 모델 아키텍처에서 NCCL 집합 통신이 성공적으로 완료되려면, 프로세스 그룹에 속한 모든 랭크가 같은(또는 상호 보완적인) NCCL 집합 통신을 *정확히 같은 순서로* 실행해야 합니다.
> In most modern model architectures (without a centralized single controller), for NCCL collectives to complete successfully, all ranks in a process group must execute the same (or complementary) NCCL collectives *in the exact same order.*

형식적으로 쓰면, 주어진 NCCL 프로세스 그룹 *G* 에 대해 *G* 를 그 프로세스 그룹에 참여하는 모든 랭크의 집합이라고 하겠습니다. 프로세스 그룹 *G* 안에서 실행되는 모든 집합 통신 중 랭크 *p* ∈ *G* 에서 실행되는 *i* 번째 집합 통신을 *C*<sub>*ip*</sub><sup>*G*</sup> 라고 하겠습니다. 그러면 랭크 *p*, *q* ∈ *G* 에 대해, 모든 *i* 에 대하여(∀ *i*) *C*<sub>*ip*</sub><sup>*G*</sup> 는 *C*<sub>*iq*</sub><sup>*G*</sup> 와 같아야 합니다. 이 기대가 깨지면 NCCL 라이브러리는 CPU-GPU 동기화 중 위반이 발생한 집합 통신에서 무한정 멈추고, 결국 NCCL 워치독 타임아웃을 일으킵니다.
> Formally, for a given NCCL process group *G*, let *G* represent the set of all ranks participating in the process group. Let *C<sub>ip</sub><sup>G</sup>* represent the *i*<sup>th</sup> collective executed on rank *p* ∈ *G* among all collectives executed within process group *G*. Then, given ranks *p*, *q* ∈ *G*, ∀ *i*, *C<sub>ip</sub><sup>G</sup>* must be the same as *C<sub>iq</sub><sup>G</sup>*. Violation of this expectation will cause the NCCL library to hang indefinitely on the violating collective during CPU-GPU sync, eventually triggering NCCL watchdog timeout.

어떤 이유로든 — GPU나 네트워크, 하드웨어와는 전혀 무관하게 — CPU가 일부 랭크에 *집합 통신을 아예 스케줄링하지 않거나 다른 집합 통신을* 스케줄링하게 되면, 집합 통신 비동기화가 발생해 NCCL 워치독 타임아웃으로 이어질 수 있습니다. **Meta 플릿 내부에서는 이 CPU 측 문제가 NCCL 워치독 타임아웃의 지배적인 근본 원인** 이었습니다(모든 학습 프레임워크에서 60% 초과).
> If, for some reason – entirely unrelated to the GPU, network, or hardware – the CPU ends up scheduling *no collective or different collectives* on some ranks, that can cause collective desync that leads to NCCL watchdog timeout. **Inside Meta's fleet, we have observed these CPU-side issues to be the dominant root cause for NCCL watchdog timeouts** (>60% across all training frameworks).

CPU 측 문제는 크게 다음 두 부류로 나눕니다:

1. CPU 측 연산에서의 멈춤/지연
2. 랭크 간 CPU 실행 분기

> We broadly divide CPU-side issues into the following two classes:
> 1. Stuckness/slowness in a CPU-side operation
> 2. Cross-rank CPU execution divergence

#### CPU 측 연산에서의 멈춤/지연 / Stuckness/slowness in a CPU-side operation

그림 4처럼, 일부 랭크에서 CPU가 어떤 CPU 측 연산(예: 데이터 로딩, 체크포인팅, PT2 컴파일)에 멈춰 있거나 NCCL 워치독 타임아웃 시간보다 오래 걸리면, 그 랭크들은 GPU에 집합 통신을 스케줄링하지 못하고, 집합 통신을 스케줄링한 다른 랭크 중 하나가 NCCL 워치독 타임아웃을 일으킵니다.
> As shown in Figure 4, if for a subset of ranks, the CPU gets stuck or takes longer than the NCCL watchdog timeout duration on some CPU-side operation (e.g. data loading, checkpointing, or PT2 compilation), those ranks will fail to schedule the collective on the GPU, causing one of the other ranks that did schedule the collective to raise a NCCL watchdog timeout.

![CPU 측 지연/멈춤으로 발생하는 NCCL 워치독 타임아웃 / NCCL watchdog timeout caused by CPU-side slowness/hang](/assets/blog/2026-03-25-flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/unnamed-7.png){:style="width:100%"}

*그림 4: CPU 측 지연/멈춤으로 발생하는 NCCL 워치독 타임아웃의 시퀀스 다이어그램 / Figure 4: Sequence diagram for NCCL watchdog timeout caused by CPU-side slowness/hang*

이런 지연의 대표적인 예가 PT2 컴파일입니다. 컴파일 시간은 데이터에 의존하는 것으로 알려져 있고, 컴파일러 캐시와 동적 형태(dynamic shapes) 재컴파일을 사용하면 랭크 간 편차가 더 커질 수 있습니다. 랭크 간 컴파일 시간 차이가 NCCL 워치독 타임아웃 임계값을 넘으면 NCCL 워치독 타임아웃이 발생할 수 있습니다. 이 문제가 [PT2 컴파일러 집합 통신(compiler collectives)](https://github.com/pytorch/pytorch/pull/130935) 도입의 직접적인 계기가 되었습니다.
> One notable example of such slowness is in PT2 compilation, where compilation times are known to be data-dependent and can vary even further across ranks when using compiler cache and dynamic shapes recompilation. If the compile time difference between ranks exceeds the NCCL watchdog timeout threshold, it can lead to NCCL watchdog timeouts. This problem directly motivated the introduction of [PT2 compiler collectives](https://github.com/pytorch/pytorch/pull/130935).

#### 랭크 간 CPU 실행 분기 / Cross-rank CPU execution divergence

서로 다른 랭크가 서로 다른 코드 경로로 진입하면, 결국 다른 집합 통신을 스케줄링하거나(그림 5) 집합 통신을 아예 스케줄링하지 못할 수 있습니다(그림 4). 두 경우 모두 일부 랭크가 이후의 CPU-GPU 동기화 지점에서 멈춰 NCCL 워치독 타임아웃을 일으킵니다.
> If different ranks entered different code paths, they could end up scheduling different collectives (Figure 5) or failing to schedule a collective (Figure 4). In both cases, some ranks will get stuck at the subsequent CPU-GPU synchronization point and raise a NCCL watchdog timeout.

![CPU 실행 분기로 발생하는 NCCL 워치독 타임아웃 / NCCL watchdog timeout caused by CPU execution divergence](/assets/blog/2026-03-25-flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/unnamed-8.png){:style="width:100%"}

*그림 5: CPU 실행 분기로 발생하는 NCCL 워치독 타임아웃의 시퀀스 다이어그램 / Figure 5: Sequence diagram for NCCL watchdog timeout caused by CPU execution divergence*

경험상 CPU 실행 분기의 흔한 원인은 다음과 같습니다:
> Here are some common causes for CPU execution divergence based on our experience:

##### 비대칭 PT2 컴파일 / Asymmetric PT2 compilation

모델 코드가 데이터에 의존한다면, PT2 컴파일 시점에 각 랭크에 주어지는 데이터의 편차가 비대칭 컴파일을 일으켜 랭크마다 다른 코드가 컴파일될 수 있습니다. 그 결과 컴파일된 코드에 서로 다른 NCCL 집합 통신이 들어가면 그림 5와 같은 CPU 실행 분기가 발생합니다. PT2 컴파일을 켠 모델에서는 이것이 지금까지 본 CPU 실행 분기의 비교적 흔한 원인 중 하나였습니다.
> If model code is data-dependent, variance in the data provided to ranks at PT2 compilation time can cause asymmetric compilation, resulting in different compiled code across ranks. When this results in different NCCL collectives in the compiled code, it causes CPU execution divergence that looks like Figure 5. In models with PT2 compilation enabled, this has been one of the more common causes of CPU execution divergence we have seen.

##### 랭크 간 데이터 불균형 또는 이질성 / Cross-rank data imbalance or heterogeneity

모델이나 학습 프레임워크에 데이터에 의존하는 조건 분기 로직이 있으면, 랭크 간 데이터 불균형이나 이질성이 NCCL 워치독 타임아웃으로 이어질 수 있습니다. 관측한 사례 중 하나는 한 랭크가 다른 랭크보다 데이터를 먼저 소진해 반복(iteration)을 한 번 덜 수행하고, 그 결과 학습 루프를 먼저 빠져나가면서 그림 4와 같은 CPU 실행 분기가 발생한 경우입니다.
> If the model or training framework contains data-dependent conditional logic, data imbalance or heterogeneity across ranks can lead to NCCL watchdog timeouts. One such scenario we have observed is when one rank exhausts data earlier than others and ends up running one fewer iteration, causing it to jump out of the training loop earlier, leading to CPU execution divergence that looks like Figure 4.

##### 부적절한 오류 처리 / Improper error handling

학습 루프 중 치명적인 예외가 발생하면 보통 오류가 난 랭크의 워커가 종료되고, PyTorch의 오류 전파(error propagation)가 나머지 워커에게도 — NCCL 집합 통신을 기다리는 중이더라도 — 똑같이 종료하라는 신호를 보냅니다. 하지만 `except` 블록 안의 코드는 본질적으로 랭크별 조건 분기 로직이므로, 다음과 같은 상황에서는 *오류가 나지 않은 랭크에서* NCCL 워치독 타임아웃이 발생할 수 있습니다:

- `except` 절이 NCCL 워치독 타임아웃 임계값보다 오래 CPU에서 멈춰 있는 경우(그림 4와 같은 형태).
- `except` 절에 랭크 간 GPU 동기화(예: 또 다른 NCCL 집합 통신이나 [destroy\_process\_group](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommdestroy))가 들어 있어, 종료 과정이 오류가 나지 않은 랭크와 교착 상태에 빠지는 경우(그림 5와 같은 형태).
- `except` 절이 예외를 *삼켜 버리고* 학습을 계속 진행해, 오류가 난 랭크가 새 NCCL 집합 통신을 발행하고 교착 상태에 빠지거나(그림 5) CPU 측 배리어나 CPU 위주 연산에서 멈추는 경우(그림 4).

> Fatal exceptions during the training loop should normally cause the worker(s) for the erroring rank(s) to tear down, and PyTorch's error propagation to signal the remaining workers to do the same, even if they are waiting on a NCCL collective. But code in the except blocks are essentially rank-specific conditional logic, which can lead to NCCL watchdog timeouts being raised *from a non-erroring rank* in any of the following scenarios:
> - Except clause gets stuck on the CPU for longer than the NCCL watchdog timeout threshold (looks like Figure 4).
> - Except clause contains inter-rank GPU synchronization (e.g., another NCCL collective or [destroy\_process\_group](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommdestroy)), causing teardown to deadlock with non-erroring ranks (looks like Figure 5).
> - Except clause *swallows* the exception and proceeds with training, causing the erroring rank(s) to either issue new NCCL collectives and deadlock (Figure 5) or get stuck at a CPU-side barrier or in a CPU-bound operation (Figure 4).

##### 엣지 케이스: 집합 통신의 GPU 실행 순서 뒤바뀜 / Edge case: Collective GPU execution reordering

대부분의 경우 집합 통신의 CPU 스케줄링 순서는 GPU 실행 순서와 같습니다. 하지만 N차원 병렬화(N-D parallelism, 예: FSDP)를 사용하는 모델에서 서로 다른 프로세스 그룹이 적절한 동기화 없이 같은 GPU에 집합 통신을 연달아(back-to-back) 스케줄링하면, 낮은 확률로 GPU 통신 커널의 실행 순서가 랭크마다 어긋나 그림 5와 같은 교착 상태가 발생하고 NCCL 워치독 타임아웃으로 이어질 수 있습니다. 이 문제를 완화하기 위해 NCCL 2.26은 GPU 통신 순서를 스케줄링 순서와 정확히 같게 강제하는 환경 변수 [NCCL\_LAUNCH\_ORDER\_IMPLICIT](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently) 을 도입했습니다.
> In most cases, collective CPU scheduling order is the same as the GPU execution order. However, in models using N-D parallelism (e.g., FSDP), when collectives are scheduled to the same GPU by different process groups without proper synchronizations (i.e. back-to-back), there is a small chance that the GPU communication kernel execution order can become inconsistent across rank, causing deadlock that looks like Figure 5 and leads to NCCL watchdog timeout. To alleviate this issue, NCCL 2.26 introduces an environment variable [NCCL\_LAUNCH\_ORDER\_IMPLICIT](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently) to enforce the GPU communication order to be the exact same as the scheduling order.

### 범주 2: GPU 연산 커널 멈춤 / Category 2: GPU compute kernel hang

하나의 CUDA 스트림 안에서 GPU 실행은 순차적이므로, 연산 커널을 실행하는 중 GPU가 멈추면 GPU는 스케줄링된 집합 통신을 실행할 수 없고, 아래 그림 6처럼 작업이 해당 스트림의 이후 CPU-GPU 동기화 지점에서 멈춥니다. 다만 증상은 동기화 시점에 따라 달라집니다. CPU-GPU 동기화가 연산 커널이 멈춘 *뒤* 이지만 NCCL 집합 통신이 스케줄링되기 *전* 에 일어나면, 동작은 오히려 그림 4처럼 보입니다.
> Since GPU execution is sequential within a single CUDA stream, any hang on the GPU while executing compute kernels will prevent the GPU from executing scheduled collectives, causing the job to get stuck at the subsequent CPU-GPU synchronization on the stream, as shown in Figure 6 below. Note the symptom depends on the timing of the synchronization: if the CPU-GPU sync occurs *after* the compute kernel hangs but *before* the NCCL collective is scheduled, the behavior looks like Figure 4 instead.

![GPU 멈춤으로 발생하는 NCCL 워치독 타임아웃 / NCCL watchdog timeout caused by GPU hang](/assets/blog/2026-03-25-flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/unnamed-9.png){:style="width:100%"}

*그림 6: GPU 멈춤으로 발생하는 NCCL 워치독 타임아웃의 시퀀스 다이어그램 / Figure 6: Sequence diagram for NCCL watchdog timeout caused by GPU hang*

GPU 멈춤은 무수히 많은 원인으로 발생할 수 있습니다. 이 범주에는 특정 커널 구현이나 특정 작업·모델에 국한된 일시적인 GPU 문제로 인한 멈춤을 넣고, 결함 있는 GPU 하드웨어에서 비롯된 멈춤은 아래 범주 4에서 설명합니다.
> GPU hangs can be caused by myriad issues; we designate this category for those caused by specific kernel implementations or transient GPU issues isolated to a particular job or model, and describe faulty GPU hardware-induced hangs in Category 4 below.

### 범주 3: 잘못 설정된 집합 통신 인자 / Category 3: Misconfigured collective arguments

보통 NCCL 집합 통신은 입력 텐서 하나를 받아 출력 텐서 하나를 내놓습니다. 대부분의 경우 이 입력/출력 텐서의 데이터 타입과 형태(shape)는 프로세스 그룹의 모든 랭크에서 같아야 합니다. 또한 [all\_to\_all\_single](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single), [broadcast](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast), [gather](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.gather) 처럼 프로세스 그룹(PG) 전체에 걸쳐 입력 인자에 특별한 전역 요건이 있는 NCCL 집합 통신도 있습니다(예: 모든 입력 크기의 합이 모든 출력 크기의 합과 같아야 함). PyTorch는 전달된 인자가 유효한지 검증할 수 없으므로, 인자가 이런 NCCL 가정을 위반하면 NCCL 라이브러리는 위반이 발생한 집합 통신에서 멈추고 결국 NCCL 워치독 타임아웃을 일으킵니다.
> Usually, a NCCL collective takes in one input tensor and emits one output tensor. In most cases, these input/output tensors' data types and shapes must be the same across all ranks in the process group. There are also some NCCL collectives like [all\_to\_all\_single](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single), [broadcast](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast) and [gather](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.gather) that have special global requirements on the input arguments across the PG (e.g., sum of all input sizes must be the same as sum of all output sizes). PyTorch is not able to verify whether the passed-in arguments are valid or not, so when the arguments violate these NCCL assumptions, the NCCL library will hang on the violating collectives and eventually cause NCCL watchdog timeout.

Meta 플릿에서 관측한 흔한 사례 하나는 [all\_to\_all\_single](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single) 입니다. 구현이 P2P(점대점) 방식이라, 호출자가 입력/출력 텐서 크기 분할(size splits)을 넘겨 송신/수신 토폴로지를 정의해야 합니다. 분할이 유효하지 않은 경우 — 예를 들어 랭크 X가 랭크 Y로부터 받기를 기대하는 데이터가 랭크 Y가 랭크 X로 보내는 데이터보다 많은 경우 — 랭크 X의 `ncclRecv` 는 영원히 블로킹되고 `all_to_all` 집합 통신은 절대 끝나지 않습니다.
> One common case we have observed in Meta's fleet is on [all\_to\_all\_single](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single). Due to the P2P nature of its implementation, it requires the caller to define the send/recv topology by passing in the input/output tensor size splits. In the case of an invalid split – e.g., rank X is expecting more data from rank Y than that rank Y is sending to rank X, the `ncclRecv` on rank X will block forever and never finish the all\_to\_all collective.

### 범주 4: 네트워크 또는 하드웨어 문제 / Category 4: Network or hardware issue

마지막으로, 관측한 타임아웃의 20~30%는 일시적이거나 지속적인 네트워크 또는 하드웨어 문제에서 비롯됩니다.
> Last but not least, between 20-30% of the timeouts we have observed are caused by either transient or persistent network or hardware issues.

일시적인 네트워크 문제(예: 링크나 포트 플랩(flap))가 이런 장애의 대부분을 차지합니다. 일시적인 네트워크 문제는 비동기화 *없이* 나타날 수 있는 드문 경우 중 하나로, 모든 집합 통신이 시작됐지만 어느 것도 완료되지 않은 상태입니다. 다만 랭크가 비대칭적으로 동작하는 집합 통신(예: `all_reduce` 나 `broadcast`)에서는, 일부 랭크는 집합 통신을 완료했지만 나머지는 아직 실행 중인 모습이 더 흔하게 보입니다.
> Transient network issues (e.g., link or port flaps) constitute the bulk of such failures. Transient network issues are one of the few cases that can manifest *without* a desync, where all collectives have started but none have completed. However, for collectives where ranks behave asymmetrically (e.g., all\_reduce or broadcast), more commonly, we see some ranks having completed the collective while others are still executing them.

결함 있는 GPU 하드웨어는 GPU 멈춤의 흔한 원인이며 범주 2의 타임아웃과 똑같은 형태로 나타납니다. 다만 범주 2와 달리, 결함 있는 GPU는 서로 무관한 여러 작업에서 장애를 유발합니다. 진단하려면 특정 GPU에서 반복되는 장애 패턴(다른 CUDA 오류 포함)을 찾거나 하드웨어 신호(예: XID)를 살펴봐야 합니다.
> Faulty GPU hardware is a common cause for GPU hang and manifests the same as Category 2 timeouts. However, unlike Category 2, a faulty GPU will induce failures across multiple, unrelated jobs. Diagnosis requires looking for a repeated pattern of failures for a given GPU (including other CUDA errors) or at hardware signals (e.g., XIDs).

## PyTorch의 진단 해법: [Flight Recorder](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html) / PyTorch's diagnostic solution: [Flight Recorder](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html)

사용자가 NCCL 워치독 타임아웃 오류를 디버깅하는 데 실질적으로 도움을 주기 위해, PyTorch는 c10d 계층 안에 [Flight Recorder](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html)(FR)를 구현했습니다. 타임아웃이 발생하면 FR은 핵심 로깅 정보를 저장소에 자동으로 덤프(dump)해, 사용자가 타임아웃 이후 분석으로 상세히 조사할 수 있게 합니다.
> To meaningfully help users debug the NCCL watchdog timeout error, PyTorch implements [Flight Recorder](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html) (FR) inside the c10d layer. When a timeout occurs, FR automatically dumps critical logging information to storage, enabling users to perform post-timeout analysis for detailed investigation.

### Flight Recorder란? / What is Flight Recorder?

FR은 랭크마다 하나씩 있는 CPU 측 링 버퍼(ring buffer)이며, 모든 프로세스 그룹이 전역으로 공유합니다. FR은 집합 통신 실행 요청과 관련해 다음과 같은 필수 메타데이터를 기록합니다:

- **종류(Type):** NCCL 집합 통신 종류(예: `all_reduce`, `all_to_all` 등)
- **상태(State):** 집합 통신은 4개 상태를 거칩니다 — 스케줄링되지 않음(not scheduled, 이른바 누락(missing)) → 스케줄링됨(scheduled, CPU에서) → 시작됨(started, GPU에서) → 완료됨(completed, GPU에서)
- **입력/출력 dtype:** GPU 연산 커널과 비슷하게 NCCL 집합 통신도 텐서를 입력 매개변수로 받고 텐서를 출력할 수도 있습니다. dtype(데이터 타입)은 입력/출력 텐서의 dtype을 가리킵니다
- **입력/출력 크기:** NCCL 집합 통신의 입력/출력 텐서 크기
- **집합 통신 호출 스택:** *NCCL 집합 통신이 스케줄링된* CPU 측 호출 스택(Python과 C++ 호출 스택 모두 기록 가능)

> FR is a per-rank, CPU-side ring buffer, shared globally across all process groups. FR records the following essential metadata related to collective launches:
> - **Type:** NCCL collective type (e.g. all\_reduce, all\_to\_all, etc.)
> - **State:** Collectives go through 4 states: not scheduled (a.k.a. missing) → scheduled (from CPU) → started (on GPU) → completed (on GPU)
> - **Input/output dtype:** Similar to GPU compute kernels, NCCL collectives take in tensors as input parameters and can also output a tensor. The dtype (data type) refers to the input/output tensor dtype
> - **Input/output size:** The input/output tensor size for the NCCL collectives
> - **Collective call stacks:** The CPU-side call stack *from which the NCCL collective was scheduled* (both Python and C++ call stacks can be recorded)

집합 통신에는 해당 랭크가 참여하는 각 프로세스 그룹 안에서 단조 증가하는 시퀀스 ID로 색인도 붙습니다. 이 메타데이터는 이후 랭크 간에 집합 통신의 사용과 순서를 검증할 때 결정적입니다.
> Collectives are also indexed by a sequence ID that is monotonically increasing within each process group the rank participates in. This metadata is crucial for later cross-rank verification of collective usage and ordering.

Python API를 통해 링 버퍼에서 Flight Recorder 데이터를 실시간으로 가져와 스트리밍 텔레메트리 분석에 쓸 수도 있습니다. 파이프 파일(환경 변수 `TORCH_NCCL_DEBUG_INFO_PIPE_FILE` 로 설정)에 쓰거나 HTTP 요청을 보내 FR 기록을 저장소로 덤프하도록 직접 트리거할 수도 있습니다. 이번 글에서 가장 중요한 것은, `TORCH_NCCL_DUMP_ON_TIMEOUT` 이 설정된 경우 NCCL 워치독 타임아웃이 감지되면 PyTorch가 FR 기록을 저장소(기본값은 로컬 파일 시스템이지만 확장 가능)로 즉시 덤프하도록 트리거한다는 점입니다.
> A Python API allows users to retrieve Flight Recorder data from the ring buffer in real-time for streaming telemetry analysis. Users can also manually trigger a dump of FR records to storage by writing to a pipe file (configured using environment variable `TORCH_NCCL_DEBUG_INFO_PIPE_FILE` or through HTTP request. Most importantly for this post, in the case of `TORCH_NCCL_DUMP_ON_TIMEOUT` is set, when a NCCL watchdog timeout is detected, PyTorch triggers the immediate dump of FR records to storage (local filesystem by default, but it is extensible).

FR 덤프가 성공하는 데 핵심은 *모든* 랭크가, 심지어 멈춰 있는 랭크까지도 자기 기록을 덤프하게 만드는 것입니다. 과거에는 랭크가 CUDA 스레드와 워치독 스레드 양쪽에서 멈출 수 있어 타임아웃 기록이 일부만 남곤 했습니다. 이를 해결하기 위해 PyTorch는 `TCPStore`(TCP/IP 기반 키-값 저장소)를 활용한 별도의 TCP/IP 채널을 도입해 타임아웃 신호를 모든 랭크에 브로드캐스트합니다. 전용 모니터 스레드가 `TCPStore` 를 폴링하다가 신호를 받으면 FR 덤프를 트리거합니다. 자세한 내용은 그림 7을 참고하세요.
> The key to a successful FR dump is ensuring that *all* ranks, even the one that is hanging, dump their records. Historically, ranks could hang on both the CUDA and watchdog threads, leading to only a partial set of timeout records. To solve this, PyTorch introduces a side TCP/IP channel, leveraging `TCPStore` (a TCP/IP-based key-value store), to broadcast timeout signals to all ranks. A dedicated monitor thread polls the `TCPStore` and triggers the FR dump upon signal receipt. See Fig. 7 for details.

![Flight Recorder가 트레이스를 덤프하는 방식 / How Flight Recorder dumps traces](/assets/blog/2026-03-25-flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/unnamed-10.png){:style="width:100%"}

*그림 7: Flight Recorder가 트레이스를 덤프하는 방식을 보여주는 시퀀스 다이어그램 / Figure 7: Sequence diagram showing how Flight Recorder dumps traces*

FR 덤프는 각 프로세스 그룹 단위로 트리거됩니다. 복잡한 N차원 병렬화 환경에서는 사용자가 여러 프로세스 그룹을 관리하고, 각 그룹은 자기 워치독과 모니터 스레드를 가집니다. 경쟁 상태(race condition)를 줄이기 위해, 신호를 확인하고 덤프를 개시하는 역할은 기본 프로세스 그룹(전체 랭크 수(world size)와 같은 크기)의 모니터 스레드에만 지정됩니다. 나머지 모니터 스레드는 덤프가 완료될 시간을 충분히 주도록 잠깐(예: 1분) 잠들도록 지시받습니다. 타임아웃 중에는 시스템이 본래 취약하기 때문에 PyTorch Flight Recorder는 최선 노력(best-effort) 방식의 로컬 덤프에 의존합니다. 이 설계는 효과적이었고, Meta 플릿 내부에서 Flight Recorder 전체 덤프 성공률(즉, 종료 전에 모든 랭크가 덤프를 완료하는 비율)이 거의 100%에 이르렀습니다.
> The FR dump is triggered at the level of each process group. In complex N-D parallelism scenarios, users manage multiple process groups, each of which has its own watchdog and monitor thread. To mitigate race conditions, only the monitor thread of the default process group (sized to the world size) is designated to check for signals and initiate the dump. All other monitor threads are instructed to sleep briefly (e.g., for 1 minute) to allow sufficient time for the dump to complete. PyTorch Flight Recorder relies on a best-effort local dump because the system is inherently fragile during a timeout. This design has proven effective, yielding a near 100% Flight Recorder full dump rate (i.e., all ranks completing the dump before teardown) inside Meta's fleet.

Meta 플릿에서는 대부분의 학습 스택에서 모든 작업에 타임아웃 시 FR 덤프를 켜 두었습니다. 모든 작업에 FR을 켜는 데 드는 약간의 오버헤드보다, 실제로 NCCL 워치독 타임아웃이 발생했을 때 그 텔레메트리가 디버깅에서 갖는 가치가 더 큽니다. 작업이 종료되면 외부 오케스트레이션이 모든 랭크의 FR 덤프를 모아 오프라인 분석(아래에서 설명)을 위해 후처리합니다.
> Within Meta's fleet, FR dump on timeout is enabled for all jobs across most training stacks. The minor overhead from enabling FR for every job is outweighed by the value of the telemetry in debugging NCCL watchdog timeouts when they occur. Upon job termination, external orchestration aggregates the FR dumps from all ranks and post-processes them for offline analysis (described below).

*참고: NCCL 워치독 타임아웃에 대한 FR 덤프를 분석할 때는 실시간 분석보다 타임아웃 이후의 오프라인 분석이 더 낫습니다. 타임아웃 중에는 (신호 브로드캐스트를 제외하면) 랭크 간 협조가 제한적인 것과 같은 이유입니다 — 시스템이 이미 조각나 있고, 모든 랭크가 살아 있으리라 기대할 수 없습니다. 랭크 하나만 죽어도 랭크 간 협조는 결국 실패합니다. 게다가 스레드가 아무 판단 없이 오랫동안 기다리게 하면 값비싼 학습 자원을 낭비하게 됩니다. PyTorch 사용자를 위한 `desync-debug` 같은 도구를 써 온 경험에서는 대규모 모델 학습에서 성능 저하와 확장성 문제가 드러났습니다. 그래서 원본 FR 기록을 먼저 덤프하고 이후에 철저히 분석하는 아키텍처 결정을 내렸습니다.*
> *Note: When analyzing FR dumps for a NCCL watchdog timeout, offline post-timeout analysis is preferred over real-time analysis. This is for the same reason that there is limited coordination between ranks during the timeout (aside from the signal broadcast) – the system is already fragmented, and not all ranks can be expected to remain alive. If even a single rank were to die, inter-rank coordination would eventually fail. Furthermore, having threads blindly wait for an extended period of time would waste valuable training resources. Historical experience with tools like `desync-debug` for PyTorch users showed performance regressions and scaling issues in large-scale model training. Consequently, we made the architectural decision to first dump the raw FR records and perform a thorough analysis afterwards.*

### FR 덤프를 타임아웃 이후 분석에 활용하는 방법 / How to leverage FR dumps for post-timeout analysis?

FR 데이터를 NCCL 워치독 타임아웃 디버깅에 제대로 활용하려면, 먼저 모든 랭크의 집합 통신 기록을 스케줄링 순서와 집합 통신 메타데이터(시퀀스 ID, 집합 통신 종류 등)를 기준으로 정렬(align)하고, 각 프로세스 그룹(PG) 안에서 집합 통신 기록을 한데 모아야 합니다. 그러면 특정 PG 안에서 실행된 하나의 집합 통신에 대응하는 각 기록 묶음을 살펴보며 불일치를 찾아낼 수 있습니다 — 즉 어떤 랭크가 집합 통신을 스케줄링하지 못했는지(누락된 랭크), 또는 같은 PG의 다른 랭크와 어긋났는지를 알아낼 수 있습니다.
> To effectively use FR data for debugging NCCL watchdog timeouts, it is first necessary to align the collective records from all ranks by their scheduling order and collective metadata (sequence ID, collective type, etc.) and to aggregate collective records within each process group (PG). Each group of records, which correspond to a single collective being executed within a particular PG, can then be examined to identify mismatches – i.e., which rank(s) failed to schedule the collective (i.e., missing ranks) or diverged from the others in the same PG.

이런 집합 통신 불일치를 보면, 위 심층 분석에서 설명한 범주 중 어느 것이 NCCL 워치독 타임아웃의 근본 원인이었는지 짚어낼 수 있습니다:

- 범주 1: CPU 측 연산의 멈춤/지연은 보통 *누락된 랭크* 나 *집합 통신 상태 불일치* 로 나타나고, 랭크 간 CPU 실행 분기는 *누락된 랭크* 또는 *집합 통신 종류·호출 스택, 때로는 dtype 의 불일치* 로 나타날 수 있습니다.
- 범주 2: GPU 멈춤은 CPU-GPU 동기화가 NCCL 집합 통신이 스케줄링되기 전에 일어나는지 후에 일어나는지에 따라 대개 *누락된 랭크* 나 *집합 통신 상태 불일치* 로 나타납니다.
- 범주 3: 잘못 설정된 집합 통신 인자는 보통 *집합 통신 dtype 또는 크기 불일치* 로 나타납니다.
- 범주 4: 결함 있는 하드웨어는 보통 GPU 멈춤과 같은 형태(*누락된 랭크 또는 집합 통신 상태 불일치*)로 나타나지만, 네트워크 문제는 불일치 *없이* (집합 통신이 모두 `started` 상태로) 나타나거나 *집합 통신 상태 불일치* 로 나타날 수 있습니다. 이런 경우는 일시적이거나 겉으로 잘 드러나지 않는 회색 장애(gray failure)일 수 있어서, 같은 네트워크/하드웨어 구성 요소가 관여할 때 반복되는 장애 패턴을 찾아 진단하는 것이 보통입니다.

> These collective mismatches allow one to pinpoint which of the categories described in the deep dive above was the root cause of the NCCL watchdog timeout:
> - Category 1: Stuckness/slowness on CPU-side operation usually manifests as a *missing rank* or *collective state mismatch*, whereas cross-rank CPU execution divergence may manifest as a *missing rank or mismatch in collective type, call stack, and occasionally dtype*.
> - Category 2: GPU hang most often manifests as a *missing rank* or *collective state mismatch*, depending on whether CPU-GPU synchronization occurs before or after the NCCL collective is scheduled.
> - Category 3: Misconfigured collective arguments usually manifest as a *collective dtype or size mismatch.*
> - Category 4: Faulty hardware usually manifests the same as GPU hang (*missing rank or collective state mismatch*), but network issues can manifest *without* a mismatch (with collectives all in the `started` state) or as a *collective state mismatch*. Typically, these must be diagnosed by looking for repeat patterns of failure when the same network/hardware components are involved, as they may be transient or gray failures.

PyTorch는 모든 랭크의 FR 덤프가 주어졌을 때 위에서 말한 정렬과 집계를 수행하는 [fr\_trace 진단 도구](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html#analyzing-flight-recorder-dumps)를 제공합니다. `fr_trace` 는 모든 NCCL 집합 통신 불일치를 나열하면서 해당 집합 통신에 참여한 랭크, 집합 통신 메타데이터, 그리고 참여 랭크 중 대표 하나의 집합 통신 호출 스택을 출력합니다.
> PyTorch provides the [fr\_trace diagnostic tool](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html#analyzing-flight-recorder-dumps) to perform the abovementioned alignment and aggregation given the FR dumps from all ranks. fr\_trace enumerates all NCCL collective mismatches, outputting the participating ranks for those collectives, collective metadata, and the collective call stack for one representative participating rank.

Meta 내부에서는 랭크와 프로세스 그룹 전반의 분산 집합 통신 활동을 시각화하는 도구를 개발해 `fr_trace` 를 보강했습니다. 그 목업(mockup)이 그림 8에 있습니다. 이 시각화는 `fr_trace` 가 만들어 낸 *정렬된* 집합 통신 기록을 바탕으로 동작하며, 후처리 과정에서 그 기록을 (집합 통신 메타데이터와 함께) 테이블 형태의 저장소에 씁니다.
> Within Meta, we have augmented fr\_trace by developing a visualization of the distributed collective activity across ranks and process groups, a mockup of which is shown in Figure 8. The visualization is powered by the *aligned* collective records generated by fr\_trace, which we write to a tabular store (including collective metadata) during postprocessing.

X축을 따라 이어지는 열은 FR이 기록한, PG와 무관한 전역 집합 통신 스케줄링 순서를 나타내고, Y축의 각 행은 전역 랭크와 프로세스 그룹의 서로 다른 조합에 대응합니다(여러 PG에 참여하는 랭크는 여러 번 나타날 수 있습니다). 각 셀은 개별 집합 통신에 해당하며, {집합 통신 종류, 호출 스택} 조합이 다르면 색이 달라지도록 색으로 구분됩니다. 셀 하나 또는 여러 셀을 선택하면 그 호출 스택들의 아이시클 차트(icicle chart)가 표시됩니다(그림 9).
> Successive columns along the X-axis represent the global, PG-agnostic collective scheduling order recorded by FR, and each row along the Y-axis corresponds to a distinct combination of global rank and process group (ranks can appear multiple times if participating in multiple PGs). Individual cells correspond to individual collectives and are color coded such that distinct combinations of {collective type, call stack} have different colors. Selecting a cell or group of cells loads an icicle chart of their call stacks (Figure 9).

*오픈 소스(OSS) 라이브러리로 이 후처리와 시각화를 구현해 보인 개념 증명(proof-of-concept) PyTorch PR을 [여기](https://github.com/pytorch/pytorch/pull/166095)에 공개했습니다.*
> *We have published a proof-of-concept PyTorch PR [here](https://github.com/pytorch/pytorch/pull/166095) to demonstrate the post-processing and visualization using OSS libraries.*

![타임아웃 디버깅 시 NCCL 집합 통신을 살펴보는 데 쓰는 시각화 목업 / Mockup of the visualization used within Meta to inspect NCCL collectives](/assets/blog/2026-03-25-flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/unnamed-11.png){:style="width:100%"}

*그림 8: 타임아웃을 디버깅할 때 NCCL 집합 통신을 살펴보기 위해 Meta 내부에서 사용하는 시각화의 목업 / Figure 8: Mockup of the visualization used within Meta to inspect NCCL collectives when debugging timeouts*

![여러 셀을 선택했을 때 나타나는 호출 스택 아이시클 뷰 목업 / Mockup of the call stack icicle view produced when multiple cells are selected](/assets/blog/2026-03-25-flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/unnamed-12.png){:style="width:100%"}

*그림 9: 여러 셀을 선택했을 때 만들어지는 호출 스택 아이시클 뷰의 목업 / Figure 9: Mockup of the call stack icicle view produced when multiple cells are selected*

이 시각화는 NCCL 워치독 타임아웃 디버깅에서 대단히 강력했습니다. 랭크 내부와 랭크 간의 집합 통신 활동 이력을 빠르게 훑어볼 수 있고, 색 구분 덕분에 어긋난 집합 통신을 눈으로 쉽게 찾아낼 수 있습니다. 어긋난 집합 통신의 호출 스택을 빠르게 비교할 수 있다는 점은 CPU 측 분기의 출처를 짚어내는 데 특히 유용합니다.
> This visualization has proven incredibly powerful in debugging NCCL watchdog timeouts. It facilitates quick scanning of the history of collective activity within and across ranks, and the color coding makes it easy to visually identify mismatched collectives. The ability to quickly compare collective call stacks for mismatched collectives is particularly useful for pinpointing the source of CPU-side divergence.

랭크를 PG별로 묶는 것은 N차원 병렬화(예: FSDP)를 쓰는 작업을 디버깅할 때 특히 도움이 됩니다 — 각 PG *안* 의 집합 통신 활동을 보는 것과 특정 랭크가 참여하는 *모든* PG를 아울러 보는 것 사이를 시각화에서 쉽게 전환할 수 있습니다. PG 간 집합 통신 스케줄링 경쟁 상태를 디버깅하는 데도 효과적이었습니다.
> Grouping ranks by PG is particularly helpful when debugging jobs that use N-D parallelism (e.g. FSDP) – the visualization makes it easy to switch between examining the collective activity *within* each PG and *across* all PGs a given rank is participating in. It has proven effective for debugging cases of cross-PG collective scheduling race conditions.

*참고: 효과적인 디버깅을 위해서는 FR을 CPU 메인 스레드 호출 스택에 대한 비슷한 분산 시각화와 함께 써야 합니다. 타임아웃 시점에 CPU가 무엇을 하고 있었는지(CPU 측 연산? CPU 측 배리어? CPU-GPU 동기화 지점? 예외 처리?) 파악해야 그 타임아웃이 어느 범주에 속하는지 좁혀 나갈 수 있기 때문입니다. PyTorch는 현재 이를 위한 진단 도구를 제공하지 않지만, 기반 텔레메트리는 py-spy 같은 오픈 소스 도구로 수집할 수 있습니다.*
> *Note: For effective debugging, FR needs to be coupled with a similar distributed visualization of CPU main thread call stacks, as understanding what the CPU was doing at timeout (CPU-side operation? CPU-side barrier? CPU-GPU synchronization point? exception handling?) is necessary to triangulate which of the categories the timeout belongs to. While PyTorch does not currently provide any diagnostic tools for this, the underlying telemetry could be collected using OSS tools like py-spy.*

## Meta 워크로드 기반 사례 연구 / Case studies based on Meta workloads

### 사례 1: CPU 실행 분기 / Case one: CPU execution divergence

분산 학습으로 추천 시스템을 확장할 때 흔히 부딪히는 어려움은 랭크 간 CPU 실행 분기를 관리하는 것입니다. 이 문제는 여러 랭크의 결과를 집합 통신 기본 연산(primitive)으로 집계하고 기록해야 하는 지표(metric) 계산 과정에서 드러나는 것을 봤습니다. 정상적으로는 정확하고 동기화된 지표 집계를 보장하기 위해 모든 랭크가 이 NCCL 집합 통신 호출에 참여해야 합니다. 하지만 구현이 충분히 견고하지 않으면, 조건 분기 로직이나 조기 종료(early exit), 또는 다른 코드 경로 분기 때문에 일부 랭크가 의도치 않게 이 호출을 건너뛸 수 있습니다.
> A common challenge when scaling recommendation systems with distributed training is managing cross-rank CPU execution divergence. We have seen this issue surface during metric computation, where results from multiple ranks must be aggregated using collective communication primitives and logged. Normally, every rank should participate in these NCCL collective calls to ensure accurate and synchronized metric aggregation. However, if the implementation is not sufficiently robust, some ranks may inadvertently skip these calls – either due to conditional logic, early exits, or other code path divergences.

이렇게 되면 집합 통신 연산에 참여하는 랭크 구성이 불완전해집니다. 참여한 랭크는 빠져나간 랭크의 입력을 무한정 기다리게 되고, 학습이 멈춘 뒤 결국 NCCL 워치독 타임아웃이 발생합니다. 아래 그림 10이 이 현상을 보여줍니다. 마지막 열이 코드 경로의 분기를 나타내는데, 일부 랭크는 지표 집계 집합 통신을 실행하고(노란색) 나머지는 지표 집계에 참여하지 않고 다음 NCCL 연산으로 넘어갑니다(검은색).
> When this happens, the membership of ranks in the collective operation becomes incomplete. Participating ranks may end up waiting indefinitely for input from ranks that opted out, leading to stalled training and eventually NCCL watchdog timeout. Figure 10 below illustrates this phenomenon: the last column shows the divergence in code paths, with some ranks executing the metric aggregation collective (yellow) while others proceed to the next NCCL operation without participating in metric aggregation (black).

![CPU 실행 분기에 대한 집합 통신 시각화 예 / Example collective visualization for CPU execution divergence](/assets/blog/2026-03-25-flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/unnamed-13.png){:style="width:100%"}

*그림 10: CPU 실행 분기에 대한 집합 통신 시각화 예 / Figure 10: Example collective visualization for CPU execution divergence*

이런 유형의 분기가 실제로 발생한 한 사례에서는, 어긋난 집합 통신과 CPU 호출 스택을 분석해 근본 원인을 학습 코드의 잘못된 조건 분기 로직으로 특정했습니다. 데이터와 관련된 특정 조건을 만족한 랭크가 지표 계산을 건너뛰게 만드는 로직이었습니다.
> In one instance of this type of divergence, through analysis of the divergent collective and CPU call stacks, we isolated the root cause to faulty conditional logic in the training code causing metrics computation to be skipped by ranks that met certain data-related conditions.

### 사례 2: 잘못 설정된 NCCL 집합 통신 입력 / Case two: Misconfigured NCCL collective inputs

GPU 간 통신 멈춤은 대규모 분산 학습, 특히 추천 시스템(RecSys)에서 잘 알려진 어려움입니다. `all_to_all` 집합 통신 연산은 RecSys 워크로드에서 널리 쓰이는데, 샤딩된(sharded) 임베딩 테이블에서는 입력과 풀링된(pooled) 출력 모두를 모든 랭크에 걸쳐 모으고 재분배해야 하기 때문입니다. 이를 통해 각 랭크가 자기 배치에 필요한 임베딩을 온전히 받아 효율적인 병렬 처리가 가능해집니다. `all_to_all` 을 쓸 때 결정적인 세부 사항은 입력과 출력 분할(split)을 지정하는 것입니다. 이 분할을 명시적으로 주지 않으면, PyTorch c10d [all\_to\_all\_single](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single) API는 입력 텐서의 길이가 모든 랭크에서 같다고 가정합니다. 이 요건이 있어야 하위 구현이 요소를 균등하게 재분배해 동기화를 유지하고 교착 상태를 막을 수 있습니다.
> GPU-to-GPU communication hangs are a well-known challenge in large-scale distributed training, especially in recommendation systems (RecSys). The all\_to\_all collective operation is widely used in RecSys workloads, as sharded embedding tables require both the inputs and the pooled outputs to be gathered and redistributed across all ranks. This ensures that each rank receives a complete set of embeddings for its batch, enabling efficient parallel processing. A critical detail in using all\_to\_all is the specification of input and output splits. If these splits are not explicitly provided, the PyTorch c10d [all\_to\_all\_single](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single) API expects that the input tensors have the same length across all ranks. This requirement allows the underlying implementation to evenly redistribute elements, maintaining synchronization and preventing deadlocks.

내부 RecSys 사례 하나에서는 GPU가 서로 다른 두 `all_to_all` 호출 — 하나는 균등 분할, 다른 하나는 불균등 분할 — 에 멈춰 있는 상황을 만났습니다. 처음에는 모든 랭크가 같은 `all_to_all` 집합 통신을 실행하는 것처럼 보여 디버깅 과정에서 혼란이 있었습니다. 처음 세운 가설은 왜 서로 다른 `all_to_all` 변형이 실행됐는지에 초점을 맞췄지만, 이 방향으로는 근본 원인이 드러나지 않았습니다. 돌파구는 FR이 제공한 `all_to_all` 집합 통신의 *호출 스택* 을 살펴본 순간에 나왔습니다 — 어긋난 `all_to_all` 변형들이 서로 다른, 연속된 두 집합 통신 연산에서 비롯됐다는 것을 발견했습니다. 잘못 설정된 입력 크기 때문에 일부 랭크는 첫 번째 `all_to_all` 을 예정보다 일찍 완료하고 다음으로 넘어갔고, 나머지는 처음 집합 통신에서 계속 기다리며 멈춰 있었습니다. 이렇게 어긋난 상태에서 랭크들이 서로 다른 집합 통신 연산에서 상대를 기다리게 되어 통신 멈춤이 발생했습니다.
> In one internal RecSys scenario, we encountered a situation where GPUs became stuck in two different all\_to\_all calls: one with even splits and another with uneven splits. At first glance, all ranks seemed to be executing the same all\_to\_all collective, which led to confusion during debugging. Our initial hypothesis focused on why different all\_to\_all variants were triggered, but this line of inquiry did not reveal the root cause. The breakthrough came when we examined the *call stacks* provided by FR for the all\_to\_all collectives – we discovered that the mismatched all\_to\_all variants originated from two different, consecutive collective operations. Due to misconfigured input sizes, some ranks prematurely completed the first all\_to\_all and moved on to the next, while others remained stuck waiting in the initial collective. This misalignment caused a communication hang, as ranks were waiting for each other in different collective operations.

## 향후 과제 / Future work

Flight Recorder의 향후 개발과 통합을 위해 다음 영역에 투자할 계획입니다:

- **TorchComm과의 통합:** 최근 발표된 [TorchComm](https://github.com/meta-pytorch/torchcomms) 라이브러리에 Flight Recorder를 통합할 계획입니다.
- **호스트 측 최적화 지원:** 가속기가 발전하고 CUDA 그래프 같은 호스트 측 최적화가 보편화되면서, Flight Recorder의 설계가 이런 기법들과 조합 가능(composable)한지 검증하고 확장해야 합니다.
- **추가 백엔드 온보딩:** Flight Recorder는 이미 범용으로 설계되어 있으므로, 그 범위를 NCCL을 넘어 MTIA와 Gloo 같은 다른 백엔드까지 확장할 계획입니다.

> We plan to invest in the following areas for Flight Recorder's future development and integration:
> - **Integration with TorchComm:** We plan to integrate Flight Recorder into the recently announced [TorchComm](https://github.com/meta-pytorch/torchcomms) library.
> - **Support for Host-Side Optimizations:** As accelerators evolve and host-side optimizations like CUDA graphs become common, we must validate and extend the Flight Recorder's design to ensure composability with these techniques.
> - **Onboarding Additional Backends:** Since Flight Recorder is already designed to be generic, we plan to expand its scope beyond NCCL to include other backends, such as MTIA and Gloo.

## 감사의 말 / Acknowledgements

이 작업에 여러 기여와 피드백, 지원과 조언을 준 Meta의 다음 협력자들에게 감사드립니다: Tristan Rice, Will Constable, Zachary DeVito, Shuqiang Zhang, Chirag Pandya, Iris Zhang, Yue Dong, Ke Wen, Chao Chen, Chien-Chin Huang, Zack Cao, Atul Jangra, David Lai, Hang Qi, Jayesh Seshadri, Shai Duvdevani, Haibo Chen, Shyam Sundar Chandrasekaran, Karthik Kambatla.
> We would like to thank the following collaborators at Meta for their various contributions, feedback, support, and guidance for this work: Tristan Rice, Will Constable, Zachary DeVito, Shuqiang Zhang, Chirag Pandya, Iris Zhang, Yue Dong, Ke Wen, Chao Chen, Chien-Chin Huang, Zack Cao, Atul Jangra, David Lai, Hang Qi, Jayesh Seshadri, Shai Duvdevani, Haibo Chen, Shyam Sundar Chandrasekaran, Karthik Kambatla.
