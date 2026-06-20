---
layout: blog_detail
title: "AutoSP 소개: 컴파일러 기반 자동 시퀀스 병렬화"
author: Ahan Gupta¹, Zhihao Wang¹, Neel Dani¹, Masahiro Tanaka², Olatunji Ruwase³, Minjia Zhang¹
category: ["pytorch.org", "translation"]
org_title: "Introducing AutoSP"
org_link: https://pytorch.org/blog/introducing-autosp/
---

대규모 언어 모델(Large-Language-Models, LLMs)은 점점 더 긴 컨텍스트를 다루는 작업을 위해 학습되고 있으며, 토큰 수가 100k를 넘어서기도 합니다. 이런 토큰 수에서는 ZeRO/FSDP와 같은 기존 학습 기법으로 디바이스 수를 늘리더라도 메모리 부족(OOM, out-of-memory) 문제가 발생하기 시작합니다. 이러한 문제를 회피하기 위해 **시퀀스 병렬화(Sequence Parallelism, SP)** — 입력 토큰을 여러 디바이스에 분할하여 GPU 수를 늘릴수록 더 긴 컨텍스트 학습을 가능하게 하는 기법 — 가 널리 사용되는 병렬 학습 기법입니다.
> Increasingly, Large-Language-Models (LLMs) are being trained for extremely long-context tasks, where token counts can exceed 100k+. At these token counts, out-of-memory (OOM) issues start to surface, even when scaling device counts using conventional training techniques such as ZeRO/FSDP. To circumvent these issues, sequence parallelism (SP): partitioning the input tokens across devices to enable long-context training with increasing GPU counts, is a commonly used parallel training technique.

그러나 SP 구현은 매우 까다로운 작업으로 잘 알려져 있으며, DeepSpeed나 HuggingFace 같은 기존 라이브러리에 침습적인 코드 수정을 요구합니다. 이러한 코드 수정에는 입력 토큰 컨텍스트(및 중간 활성화)의 분할, 집합 통신(communication collectives) 삽입, 통신과 계산의 중첩이 포함되며, 이 모든 작업을 순전파와 역전파 모두에 대해 수행해야 합니다. 그 결과, 긴 컨텍스트 기능을 실험하려는 연구자들은 이러한 기능을 활성화하기 위한 시스템 스택 엔지니어링에 상당한 노력을 들이게 되고, 하드웨어 벤더가 달라질 때마다 이 작업을 반복해야 합니다.
> However, implementing SP is notoriously difficult, requiring invasive code changes to existing libraries such as DeepSpeed or HuggingFace. These code changes often involve partitioning input token contexts (and intermediate activations), inserting communication collectives, and overlapping communication with computation, all of which must be done for both the forward and backwards pass. This results in researchers who want to experiment with long context capabilities spending significant effort on engineering the system's stack to enable such capability, repeating this effort for different hardware vendors.

이러한 복잡성을 피하기 위해 [AutoSP](https://openreview.net/pdf?id=0fgsHvmBBI)를 소개합니다. AutoSP는 작성하기 쉬운 학습 코드를 자동으로 멀티 GPU 시퀀스 병렬 코드로 변환하는 완전 자동화된 컴파일러 기반 솔루션으로, 기존 병렬 전략(예: ZeRO)과 조합하면서도 GPU를 효율적으로 활용해 더 긴 입력 컨텍스트에서 학습할 수 있게 합니다. 이를 통해 개발자가 긴 컨텍스트 학습을 위해 학습 파이프라인을 반복적으로 수정해야 하는 번거로움이 사라집니다. 이제 사용자는 AutoSP를 임포트(import)하고 AutoSP 백엔드를 사용해 임의의 모델을 컴파일하는 것만으로 누구나 긴 컨텍스트 학습의 힘을 누릴 수 있습니다. 또한 이 기술을 컴파일러에 내장함으로써 접근 방식이 성능 이식성(performance-portable)을 갖추게 되어, 다양한 하드웨어에서도 높은 성능의 SP를 구현할 수 있습니다.
> To avoid this complexity, we introduce [AutoSP](https://openreview.net/pdf?id=0fgsHvmBBI): a fully automated compiler-based solution that automatically converts easy-to-write training code to multi-GPU sequence parallel code that efficiently uses GPUs to train on longer input contexts while composing with existing parallel strategies (such as ZeRO). This avoids the cumbersome need for developers to repeatedly modify training pipelines for long-context training. Users can now simply import AutoSP and compile arbitrary models using the AutoSP backend, giving the power of long-context training to anyone. Moreover, by embedding this technology into the compiler, our approach is performance-portable: highly performant SP can be realised on diverse hardware.

이번 글은 다음과 같이 구성됩니다. (1) AutoSP와 모델 연구자가 이를 어떻게 활용해 긴 컨텍스트 학습을 가능하게 할 수 있는지, (2) AutoSP의 핵심 설계 결정, (3) AutoSP의 주요 결과로 사용 편의성과 영향력 입증, (4) AutoSP의 한계와 할 수 없는 일.
> We structure this post as follows: (1) AutoSP and how model scientists can use it to enable long-context training, (2) Key design decisions of AutoSP, (3) key AutoSP results, demonstrating its ease-of-use and impact, (4) some limitations and things AutoSP cannot do.

## AutoSP 사용법 / AutoSP Usage

AutoSP의 핵심 설계 철학은 여러 GPU를 프로그래밍하는 데 따르는 복잡성 대부분을 사용자로부터 추상화하여 단순화하는 것입니다. 이를 위해 AutoSP를 [DeepCompile](https://arxiv.org/pdf/2504.09983) 위에 구현하였습니다. DeepCompile은 DeepSpeed 내부의 컴파일러 생태계로, 심층 신경망(deep neural network) 학습을 위한 다양한 최적화를 프로그래밍 방식으로 적용할 수 있게 합니다. 이를 통해 DeepSpeed를 사용하는 누구나 거의 부담 없이 시퀀스 병렬화를 자동으로 활성화할 수 있습니다. 예제를 살펴보겠습니다.
> A key design philosophy of AutoSP is simplicity in abstracting most of the complexity in programming multiple GPUs from users. To do this, we implement AutoSP within [DeepCompile](https://arxiv.org/pdf/2504.09983): a compiler ecosystem within DeepSpeed to programmatically enable diverse optimisations for deep neural network training. With this, any user who uses DeepSpeed can automatically enable Sequence Parallelism with almost zero hassle. We take a look at an example next.

```python
# deepspeed config를 인스턴스화합니다.
# 2개의 DP rank와 4개의 SP rank를 가진 8개의 GPU를 가정합니다.

config = {
    "train_micro_batch_size_per_gpu": 1,
    "train_batch_size": 2,
    "steps_per_print": 1,
    "optimiser": {
        "type": "Adam",
        "params": {
            "lr": 1e-4
        }
    },
    "zero_optimization": {
        "stage": 1, # AutoSP는 ZeRO 0/1과 상호운용됩니다.
    },
    # deepcompile을 켜고 AutoSP 패스를 활성화하도록 설정합니다.
    "compile": {
        "deepcompile": True,
        "passes": ["autosp"]
    },
    "sequence_parallel_size": 4,
    "gradient_clipping": 1.0,
}

# 모델로 deepspeed를 초기화합니다.
model, _, _ = deepspeed.initialize(config=config,model=model)

# 모델을 컴파일하고 AutoSP 패스를 자동 적용합니다.
model.compile(compile_kwargs={"dynamic": True})

for idx, batch in enumerate(train_loader):
    # deepspeed/compile/passes/sp_compile 내에서 노출하는 커스텀 함수입니다.
    inputs, labels, positions, mask = prepare_autosp_inputs(batch)

    loss = model(
        input_ids=inputs,
        labels=labels,
        position_ids=positions,
        attention_mask=mask
    )

    ... # 역전파, 옵티마이저 스텝 등...
```

위 예제에서 볼 수 있듯이, 사용자는 단일 디바이스에서 실행되는 기존 학습 코드를 가져와 다음 작업을 수행합니다. (1) AutoSP 내부 프로그램 분석에 사용되도록, DeepSpeed에서 노출하는 `prepare_autosp_inputs` 유틸리티 함수(옮긴이 주: 원문의 `prepare_auto_sp_inputs` / `prepare_autosp_input` 표기는 오타이며, 실제 DeepSpeed에 노출된 함수명은 `prepare_autosp_inputs` 입니다)를 사용해 입력 토큰, 어텐션 마스크, 위치 ID(position id)에 가벼운 태깅을 수행합니다. (2) DeepSpeed config를 조정해 DeepCompile을 켜고, `passes` 플래그를 `autosp`로 지정합니다. 나머지는 모델 컴파일 시 호출되는 AutoSP 컴파일러 패스를 통해 처리되며, 다른 긴 컨텍스트 학습 최적화와 함께 시퀀스 병렬화를 자동으로 활성화합니다. 또한 AutoSP는 기본적으로 ZeRO stage 1과 자동 조합되며, DeepSpeed에서 ZeRO-1 플래그를 AutoSP 플래그와 함께 설정하기만 하면 두 전략을 결합할 수 있습니다.
> As seen in the example above, users take existing training code that runs on a single device and do the following: (1) use the `prepare_autosp_input` utility function (exposed in DeepSpeed) for lightweight tagging of input tokens, attention masks and position ids for use in program analysis within AutoSP. (2) Adjust the DeepSpeed config to turn DeepCompile on, specifying the "passes" flag to "autosp". The rest is handled through the AutoSP compiler passes, called when compiling the model, which automatically enable sequence-parallelism alongside other long-context training optimisations. AutoSP additionally automatically composes with ZeRO stage 1 out of the box, simply set the ZeRO-1 flag in DeepSpeed alongside the AutoSP flags to combine both strategies.

## AutoSP 컴파일러 패스 / AutoSP Compiler Passes

AutoSP가 사용자 코드를 변환하여 더 긴 컨텍스트 학습을 가능하게 하기 때문에, 투명성을 위해 AutoSP의 핵심 설계 포인트와 코드 변환, 그리고 그것이 사용자에게 미치는 결과를 간략히 살펴봅니다.
> Since AutoSP transforms user code to enable longer-context training, we briefly cover the key design points of AutoSP and code transformations, as well as its consequences to users for transparency.

**시퀀스 병렬화 코드 변환(Sequence Parallelism Code Transformations).** AutoSP는 단일 GPU 코드를 멀티 GPU 시퀀스 병렬(SP) 코드로 자동 변환합니다. AutoSP가 변환하는 구체적인 SP 전략은 [DeepSpeed-Ulysses](https://dl.acm.org/doi/10.1145/3662158.3662806) 입니다. 다른 전략(예: [RingAttention](https://arxiv.org/pdf/2310.01889))보다 DeepSpeed-Ulysses에 집중하는 이유는, NVLink 네트워크 토폴로지나 팻 트리(fat-tree) 네트워크에서 GPU 수가 증가해도 통신 오버헤드가 일정하게 유지되기 때문입니다. 다만 DeepSpeed-Ulysses는 SP 크기를 모델의 헤드 수(7-8B 모델에서는 32)까지만 확장할 수 있다는 제약이 있습니다.
> Sequence Parallelism Code Transformations. AutoSP automatically converts single-GPU code to multi-GPU sequence parallel (SP) code. The specific SP strategy AutoSP converts code into is [DeepSpeed-Ulysses](https://dl.acm.org/doi/10.1145/3662158.3662806). We specifically focus on DeepSpeed-Ulysses over other strategies (e.g. [RingAttention](https://arxiv.org/pdf/2310.01889)) as its communication overhead stays constant with increasing GPU counts on NVLink network topologies or fat-tree networks. However, DeepSpeed-Ulysses only enables scaling the SP-size to the number of heads in a model (32 in 7-8B models).

**긴 컨텍스트 학습을 위한 활성화 체크포인팅(Activation Checkpointing for longer-context training).** AutoSP는 추가로 긴 컨텍스트 모델링에 맞춰 설계된 커스텀 활성화 체크포인팅(AC) 전략을 적용합니다. AC는 계산 비용이 저렴한 연산자의 중간 활성화를 해제하고, 역전파에서 관련 변화도(gradient)를 계산할 때 필요에 따라 다시 계산하는 방식입니다. PyTorch-2.0은 최대 흐름-최소 절단(max-flow min-cut) 기반의 자동화된 [AC 정식화](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467)를 도입했지만, 긴 컨텍스트 모델링에는 지나치게 보수적이라는 점을 확인했습니다. 이에 따라 긴 컨텍스트 학습을 겨냥한 새로운 AC 전략인 **시퀀스 인식 AC(Sequence-aware AC, SAC)** 를 도입하여, 긴 컨텍스트에서 나타나는 고유한 FLOP 동역학(dynamics)을 활용합니다. 이 기능이 활성화되면(AutoSP의 기본 설정) 학습 처리량이 근소하게 감소하지만, 이 기능 없이는 더 긴 컨텍스트에서의 학습이 불가능합니다. 따라서 사용자는 OOM이 발생하는 설정에 한해 이 패스를 선택적으로 켤 수 있습니다.
> Activation Checkpointing for longer-context training. AutoSP additionally applies a custom activation-checkpointing (AC) strategy curated for long-context modelling. AC releases intermediate activations of cheap-to-compute operators, recomputing them in the backwards pass as required to compute relevant gradients. PyTorch-2.0 introduces an automated max-flow min-cut based [AC formulation](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467), but we find this to be overly conservative for long-context modelling. We accordingly introduce a novel AC strategy targeted for long-context training: Sequence-aware AC (SAC), which exploits unique long-context FLOP dynamics. When triggered on (the default setting in AutoSP), this marginally reduces training throughput. However, without it, training on longer contexts is infeasible, so the user can selectively choose to turn this pass on only for configurations that OOM.

## 실제 모델에서의 AutoSP 평가 / Evaluating AutoSP on Real Models

AutoSP의 실효성을 입증하기 위해, NVIDIA GPU에서 다양한 크기의 모델로 성능을 평가하여 사용 편의성이 런타임 성능 손실로 거의 이어지지 않음을 보입니다. 8장의 A100-80GB SXM GPU를 갖춘 단일 노드에서 다양한 Llama 3.1 모델을 벤치마크합니다. PyTorch 2.7과 CUDA 12.8을 사용하며, AutoSP를 [RingFlashAttention](https://github.com/zhuzilin/ring-flash-attention), DeepSpeed-Ulysses, ZeRO-3의 torch-컴파일된 수작업 베이스라인과 비교합니다. 주요 결과를 아래 그림에 요약합니다.
> To demonstrate AutoSP's viability, we evaluate its performance on models of varying sizes on NVIDIA GPUs to show that its ease of use comes at little to no cost to runtime performance. We benchmark different Llama 3.1 models on an 8 A100-80Gb SXM node. We use PyTorch 2.7 with CUDA 12.8, comparing AutoSP to torch-compiled hand-written baselines of: [RingFlashAttention](https://github.com/zhuzilin/ring-flash-attention), DeepSpeed-Ulysses, and ZeRO-3. We summarise key results in the figure below:

![AutoSP 벤치마크 결과 / AutoSP benchmark results](/assets/blog/2026-04-29-introducing-autosp/unnamed-32.png){:style="width:100%"}

AutoSP는 동일한 자원으로 학습 가능한 최대 시퀀스 길이를 늘릴 수 있을 뿐 아니라(왼쪽 그림 — 클수록 좋음), 이러한 이득이 런타임 성능에 거의 비용을 발생시키지 않습니다(오른쪽 그림 — 작을수록 좋음).
> Not only can AutoSP increase the maximum trainable sequence length given the same resources (left figure – higher is better), but also these benefits come at little cost to runtime performance (right figure – lower is better).

## 한계 / Limitations

AutoSP에는 두 가지 핵심 한계가 있습니다. 첫째, 사용자는 트랜스포머를 단일 컴파일 가능한 아티팩트로 강제 컴파일해야 합니다. 때로 PyTorch 사용자는 여러 함수를 개별적으로 컴파일하여 하나의 모델로 이어 붙이기도 하는데, AutoSP에서는 이를 허용하지 않습니다. 입력 시퀀스를 올바르게 분할(shard)하고 그래프 전체에 정보를 전파하려면 모델 전체를 컴파일하고 관찰해야 하기 때문입니다. 둘째, 컴파일 가능한 아티팩트 내의 그래프 단절(graph break)을 일체 허용하지 않습니다. 그래프 단절은 정보의 분석과 전파를 복잡하게 만들기 때문에, AutoSP를 그래프 단절에 견디도록 확장하는 작업은 향후 연구 과제로 남겨둡니다.
> There are two key limitations of AutoSP. First, we require that the user forcefully compile a transformer as a single compilable artifact. Occasionally, PyTorch users may compile many functions individually and stitch them together into one model. This is disallowed in AutoSP as we need to compile and see the entire model to correctly shard input sequences and propagate this information throughout the entire graph. Second, we disallow any graph breaks in compilable artifacts. This complicates analysis and propagation of information, and we leave extending AutoSP to be graph-break resilient to future research.

## 결론 / Conclusion

AutoSP를 사용하면 임의의 트랜스포머 학습 코드를 손쉽게 확장하여 시퀀스 병렬화를 적용할 수 있으며, 긴 컨텍스트 학습 강화를 위한 커스텀 AC 전략도 함께 사용할 수 있습니다. DeepSpeed와의 통합을 통해 사용자는 설정 파일만 변경하는 것만으로 기존 DeepSpeed 학습 코드를 활용하여 더 긴 컨텍스트에서 학습할 수 있습니다. 실제 모델 워크로드(예: Llama 3.1 8B)에서 직접 사용해 볼 수 있는 종단 간(end-to-end) 예제를 [이곳](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks/autosp)에 준비해 두었습니다. 직접 시도해보고 긴 컨텍스트 학습이 얼마나 쉬워졌는지 확인해 보세요.
> AutoSP enables users to easily extend arbitrary transformer training code to enable Sequence Parallelism, with a custom AC strategy for enhanced long-context training. Integration with DeepSpeed allows users to easily use existing DeepSpeed training code to train on longer contexts by simply changing a config file. We have prepared end-to-end examples for users to play around with on real model workloads (e.g. Llama 3.1 8B) [here](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks/autosp). Give it a try to see how easy long context training has become.
