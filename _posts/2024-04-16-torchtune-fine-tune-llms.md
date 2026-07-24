---
layout: blog_detail
title: "torchtune: PyTorch를 사용한 쉬운 LLM 파인튜닝"
image: /assets/blog/2024-04-16-torchtune-fine-tune-llms/hero.png
org_title: "torchtune: Easily fine-tune LLMs using PyTorch"
category: ["pytorch.org", "translation"]
org_link: https://pytorch.org/blog/torchtune-fine-tune-llms/
---

대규모 언어 모델(LLM)을 손쉽게 파인튜닝(미세조정)할 수 있는 PyTorch 네이티브 라이브러리인 torchtune의 알파 릴리즈를 발표하게 되어 기쁩니다.
> We’re pleased to announce the alpha release of torchtune, a PyTorch-native library for easily fine-tuning large language models.

torchtune은 파이토치(PyTorch)의 설계 원칙을 충실히 따랐으며, 다양한 소비자급 및 전문가용 GPU에서 인기있는 LLM들을 파인튜닝할 수 있도록 모듈식 블록 구성과 확장이 쉬운 학습 예시(training recipe)들을 제공합니다.
> Staying true to PyTorch's design principles, torchtune provides composable and modular building blocks along with easy-to-extend training recipes to fine-tune popular LLMs on a variety of consumer-grade and professional GPUs.

torchtune은 다음과 같은 시작부터 끝까지의 파인튜닝 워크플로우를 전반을 지원합니다:
> torchtune supports the full fine-tuning workflow from start to finish, including

* 데이터셋 및 모델 체크포인트 다운로드 및 준비.
* 다양한 모델 아키텍처와 매개변수 효율적 미세 조정(PEFT) 기술 등을 지원하는 빌딩 블록 구성으로 학습 과정의 커스터마이징.
* 학습 과정에서의 진행 상황 및 지표(metric)을 기록하여 인사이트 확보.
* 파인튜닝 후 모델 양자화(quantization).
* 파인튜닝된 모델을 인기 있는 벤치마크들로 평가.
* 파인튜닝된 모델 테스트를 위한 로컬 추론 실행.
* (파인튜닝한 모델의) 체크포인트와 주로 사용되는 주요 프로덕션 추론 시스템과 호환성.
> * Downloading and preparing datasets and model checkpoints.
> * Customizing the training with composable building blocks that support different model architectures, parameter-efficient fine-tuning (PEFT) techniques, and more.
> * Logging progress and metrics to gain insight into the training process.
> * Quantizing the model post-tuning.
> * Evaluating the fine-tuned model on popular benchmarks.
> * Running local inference for testing fine-tuned models.
> * Checkpoint compatibility with popular production inference systems.

시작하려면 [코드](https://www.github.com/pytorch/torchtune)나다양한 [튜토리얼](https://pytorch.org/torchtune/main/)을 살펴보세요!
> To get started, jump right into the [code](https://www.github.com/pytorch/torchtune) or walk through our many [tutorials](https://pytorch.org/torchtune/main/)!


## 왜 torchtune을 사용해야 하나요? / Why torchtune?

지난 한 해 동안 개방형 LLM(Open LLM)에 대한 관심이 폭발적으로 증가했습니다. 최신 모델(SotA, State-of-the-Art)들을 특정한 사용 사례에 맞춰 파인튜닝하는 것이 중요한 기술로 부상했으며, 이러한 적응은 데이터셋 및 모델 선택부터 양자화(Quantization), 평가(Evaluation) 및 추론(Inference)까지 광범위한 사용자 정의를 필요로 하기도 합니다. 게다가 이러한 모델의 크기는 메모리가 제한된 소비자급 GPU에서 파인튜닝을 시도할 때 상당한 어려움을 야기합니다.
> Over the past year there has been an explosion of interest in open LLMs. Fine-tuning these state of the art models has emerged as a critical technique for adapting them to specific use cases. This adaptation can require extensive customization from dataset and model selection all the way through to quantization, evaluation and inference. Moreover, the size of these models poses a significant challenge when trying to fine-tune them on consumer-level GPUs with limited memory.

기존 솔루션들은 사용자 정의(customization)나 최적화(optimization)에 필요한 부분들을 추상화된 계층 뒤에 숨겨놓아, 이러한 기능들을 추가하기 어렵게 만듭니다. 서로 다른 구성 요소가 어떻게 상호 작용하며 새로운 기능을 추가하려면 어떤 부분을 업데이트해야 하는지가 분명하지 않습니다. torchtune은 개발자들에게 완전한 제어와 가시성을 제공하여 특정한 요구 사항과 제약 조건에 맞춰 LLM을 쉽게 적응하고 제어할 수 있도록 지원합니다.
> Existing solutions make it hard to add these customizations or optimizations by hiding the necessary pieces behind layers of abstractions. It’s unclear how different components interact with each other and which of these need to be updated to add new functionality. torchtune empowers developers to adapt LLMs to their specific needs and constraints with full control and visibility.


## torchtune의 설계 / torchtune’s Design

torchtune은 다음과 같은 원칙을 바탕으로 설계되었습니다:
> torchtune was built with the following principles in mind

* **손쉬운 확장** - 새로운 기법들이 끊임없이 등장하고 있으며, 모든 파인튜닝 사례는 서로 다릅니다. torchtune의 학습 예시(recipe)는 쉽게 조합(composable)할 수 있는 구성요소들과 변경 가능한(hackable) 학습 루프로 설계되어 있어, 파인튜닝을 어렵게 하는 추상화를 최소화하였습니다. 각 [학습 예시(recipe)들](https://github.com/pytorch/torchtune/tree/main/recipes)은 별도의 학습기(trainer)나 프레임워크 없이 독립적으로 구성되어 있으며, 600줄 미만의 코드로 쉽게 읽을 수 있도록 설계되었습니다!
* **파인튜닝의 민주화** - 모든 사용자들이 지식 수준과 상관없이 torchtune을 사용할 수 있도록 하였습니다. 설정(config)을 복제하고 변경하거나 코드를 직접 바꿔보세요! 또한, 데이터센터에 있는 고성능의 GPU가 필요하지 않습니다. 메모리 효율적인 학습 예시들은 24GB 게이밍 GPU를 단지 하나만 장착한 기기에서 테스트되었습니다.
* **오픈소스 LLM 생태계와의 상호 운용성** - 오픈소스 LLM 생태계는 엄청나게 번창하고 있으며, torchtune은 이를 활용하여 다양한 제품군과 상호 운용할 수 있도록 지원합니다. 이러한 유연성을 통해 사용자가 모델을 어떻게 학습하고, 파인튜닝된 모델을 사용할지를 확실하게 제어할 수 있습니다.
> * **Easy extensibility** - New techniques emerge all the time and everyone’s fine-tuning use case is different. torchtune’s recipes are designed around easily composable components and hackable training loops, with minimal abstraction getting in the way of fine-tuning your fine-tuning. Each [recipe](https://github.com/pytorch/torchtune/tree/main/recipes) is self-contained - no trainers or frameworks, and is designed to be easy to read - less than 600 lines of code!
> * **Democratize fine-tuning** - Users, regardless of their level of expertise, should be able to use torchtune. Clone and modify configs, or get your hands dirty with some code! You also don’t need beefy data center GPUs. Our memory efficient recipes have been tested on machines with a single 24GB gaming GPU.
> * **Interoperability with the OSS LLM ecosystem** - The open source LLM ecosystem is absolutely thriving, and torchtune takes advantage of this to provide interoperability with a wide range of offerings. This flexibility puts you firmly in control of how you train and use your fine-tuned models.

내년에는 개방형 LLM(Open LLM)이 더 많은 언어(다국어)와 더 많은 모달리티(멀티모달), 그리고 더 많은 작업들을 지원하며 강력해질 것입니다. 이러한 모델의 복잡성이 증가함에 따라, 우리는 제공되는 기능이나 학습 실행 시의 성능뿐만 아니라 라이브러리를 '어떻게' 설계할지에 대해서도 마찬가지로 주의를 기울여야 합니다. 커뮤니티가 현재의 혁신 속도를 유지하기 위해서는 유연성(flexibility)이 핵심 요소가 될 것이며, 다양한 사용 사례를 지원하기 위해 수많은 라이브러리와 도구들이 서로 잘 동작해야 할 것입니다. torchtune은 처음부터 이러한 미래를 염두에 두고 설계되었습니다.
> Over the next year, open LLMs will become even more powerful, with support for more languages (multilingual), more modalities (multimodal) and more tasks. As the complexity of these models increases, we need to pay the same attention to “how” we design our libraries as we do to the features provided or performance of a training run. Flexibility will be key to ensuring the community can maintain the current pace of innovation, and many libraries/tools will need to play well with each other to power the full spectrum of use cases. torchtune is built from the ground up with this future in mind.

진정한 파이토치 정신(True PyTorch Spirit)에 따라, torchtune은 LLM 작업에 주로 사용되는 도구들과의 통합을 제공하여 쉽게 시작할 수 있도록 합니다.
> In the true PyTorch spirit, torchtune makes it easy to get started by providing integrations with some of the most popular tools for working with LLMs.


* **[허깅페이스 허브](https://huggingface.co/docs/hub/en/index)** - 허깅페이스는 파인튜닝을 위한 방대한 오픈소스 모델과 데이터셋을 제공합니다. torchtune은 `tune download` CLI 명령을 통해 쉽게 시작할 수 있도록 허깅페이스 허브를 통합하여 파인튜닝을 바로 시작할 수 있도록 지원합니다.
* **[PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)** - PyTorch FSDP를 사용하여 학습을 확장할 수 있습니다. 일반적으로 NVIDIA의 3090/4090과 같은 소비자급 GPU를 장착한 기기들을 많이 사용하고 있습니다. torchtune은 FSDP 기반의 분산 학습 예시를 제공하여 이러한 설정을 활용할 수 있도록 지원합니다.
* **[Weights & Biases](https://wandb.ai/site)** - torchtune은 Weights & Biases의 AI 플랫폼을 사용하여 학습 중 지표(metric)들과 모델의 체크포인트를 기록합니다. 파인튜닝 실행 중, 설정과 메트릭 및 모델 등을 한 곳에서 한꺼번에 추적할 수 있습니다!
* **[EleutherAI의 언어모델 평가도구](https://github.com/EleutherAI/lm-evaluation-harness)** - 파인튜닝을 통해 원하는 결과를 얻을 수 있는지 여부를 이해하기 위해서는 파인튜닝된 모델의 평가(Evaluation)가 중요합니다. torchtune은 EleutherAI의 언어모델(LM) 평가도구(Evaluation Harness)를 활용하여 일반적으로 많이 사용하는 LLM 벤치마크 모음에 쉽게 접근할 수 있는 간단한 평가 학습 예시를 제공합니다. 평가 과정의 중요성을 고려하여, 앞으로 몇 달 동안 EleutherAI와 밀접하게 협력하여 더 깊고 더 "네이티브(native)"하게 통합할 예정입니다.
* **[ExecuTorch](https://pytorch.org/executorch-overview)** - torchtune으로 파인튜닝된 모델은 ExecuTorch로 [쉽게 내보내기(export)](https://github.com/pytorch/executorch/tree/main/examples/models/llama2#optional-finetuning)할 수 있어, 다양한 모바일 및 엣지 장치(Edge Device)에서 효율적인 추론을 실행할 수 있습니다.
* **[torchao](https://github.com/pytorch-labs/ao)** - torchao의 양자화 API(Quantization API)를 활용한 간단한 [학습 후 양자화 예시(Post-training recipe)](https://github.com/pytorch/torchtune/blob/main/recipes/quantize.py)를 통해 파인튜닝된 모델을 4비트 또는 8비트로 쉽게 양자화할 수 있습니다.
> * **[Hugging Face Hub](https://huggingface.co/docs/hub/en/index)** - Hugging Face provides an expansive repository of open source models and datasets for fine-tuning. torchtune seamlessly integrates through the `tune download` CLI command so you can get started right away with fine-tuning your first model.
> * **[PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)** - Scale your training using PyTorch FSDP. It is very common for people to invest in machines with multiple consumer level cards like the 3090/4090 by NVidia. torchtune allows you to take advantage of these setups by providing distributed recipes powered by FSDP.
> * **[Weights & Biases](https://wandb.ai/site)** - torchtune uses the Weights & Biases AI platform to log metrics and model checkpoints during training. Track your configs, metrics and models from your fine-tuning runs all in one place!
> * **[EleutherAI’s LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)** - Evaluating fine-tuned models is critical to understanding whether fine-tuning is giving you the results you need. torchtune includes a simple evaluation recipe powered by EleutherAI’s LM Evaluation Harness to provide easy access to a comprehensive suite of standard LLM benchmarks. Given the importance of evaluation, we will be working with EleutherAI very closely in the next few months to build an even deeper and more “native” integration.
> * **[ExecuTorch](https://pytorch.org/executorch-overview)** - Models fine-tuned with torchtune can be [easily exported](https://github.com/pytorch/executorch/tree/main/examples/models/llama2#optional-finetuning) to ExecuTorch, enabling efficient inference to be run on a wide variety of mobile and edge devices.
> * **[torchao](https://github.com/pytorch-labs/ao)** - Easily and efficiently quantize your fine-tuned models into 4-bit or 8-bit using a simple [post-training recipe](https://github.com/pytorch/torchtune/blob/main/recipes/quantize.py) powered by the quantization APIs from torchao.


## 다음 단게는 무엇인가요? / What’s Next?

이것은 시작에 불과하며 활기차고 에너지가 넘치는 커뮤니티에 이번 알파 버전을 제공하게 되어 매우 기쁩니다. 앞으로 몇 주 동안, 더 많은 모델과 기능 및 파인튜닝 기법들을 추가할 예정입니다. 피드백이나 의견, 기능 요청은 GitHub 저장소의 이슈(issue)나 [Discord 채널](https://discord.com/invite/4Xsdn8Rr9Q)로 보내주시기 바랍니다. 언제나 그렇듯, 이 멋진 커뮤니티로부터의 모든 기여를 환영합니다. 즐거운 파인튜닝하세요!
> This is just the beginning and we’re really excited to put this alpha version in front of a vibrant and energetic community. In the coming weeks, we’ll continue to augment the library with more models, features and fine-tuning techniques. We’d love to hear any feedback, comments or feature requests in the form of GitHub issues on our repository, or on our [Discord channel](https://discord.com/invite/4Xsdn8Rr9Q). As always, we’d love any contributions from this awesome community. Happy Tuning!
