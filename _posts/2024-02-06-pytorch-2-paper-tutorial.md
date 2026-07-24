---
layout: blog_detail
title: "PyTorch 2 논문 및 튜토리얼 @ ASPLOS 2024"
image: /assets/blog/2024-02-06-pytorch-2-paper-tutorial/hero.png
category: ["pytorch.org", "translation"]
org_title: "PyTorch 2 paper and tutorial @ ASPLOS 2024"
org_link: https://pytorch.org/blog/pytorch-2-paper-tutorial/
---

2024년 4월 27일부터 5월 1일까지 미국 캘리포니아주 샌디에이고에서 열릴 예정인 ACM 국제 컨퍼런스 ASPLOS(Architectural Support for Programming Languages and Operating Systems)에서 PyTorch 2에 대한 논문이 선정되어 발표하게 되었다는 소식을 전할 수 있어 기쁘게 생각합니다.
> The PyTorch team is excited to share that our paper on PyTorch 2 has been accepted for presentation at the ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), scheduled to take place from April 27 to May 1, 2024, in San Diego, CA, USA.

이 논문에서는 torch.compile의 구현에 대해 자세히 살펴보며, 특히 이를 위한 주요 기술들인 TorchDynamo(그래프 캡처), TorchInductor(백엔드 컴파일러) 및 Dynamic Shape 지원 등을 중점적으로 다루고 있습니다.
> The paper delves into the implementation of torch.compile and highlights the key technologies driving it, including TorchDynamo (graph capture), TorchInductor (backend compiler), and Dynamic Shape support.

ASPLOS 컨퍼런스 기간 중인 4월 27일(토)에 시스템 연구자들을 위해 PyTorch 2의 내부 동작 방식과 이를 활용하고 구축할 수 있는 방법에 초점을 맞춘 튜토리얼을 진행할 예정입니다. 행사 일정에 맞춰 세부적인 내용이 확정되는대로 공유드리도록 하겠습니다. 많은 참여 기대합니다!
> During the ASPLOS conference, we'll be conducting a tutorial on Saturday, April 27, focusing on the inner workings of PyTorch 2 and how systems researchers can leverage and build upon it. Stay tuned for more details as the event approaches – we look forward to your participation!

논문의 미리보기는 아래 첨부하였습니다:
> A preview of the paper is attached below:

제목:  **PyTorch 2: 동적 Python 바이트코드 변환과 그래프 컴파일을 통한 더 빠른 머신 러닝** [**논문 전문 PDF**](/assets/pytorch2-2.pdf)
> Title: **PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation.** [**Full Paper PDF**](/assets/pytorch2-2.pdf)

### 논문 초록 (Abstract)

이 논문에서는 인기있는 파이토치(PyTorch) 머신러닝 프레임워크의 두 가지 확장 기능인 TorchDynamo와 TorchInductor를 소개합니다. 이 두 확장 기능은 PyTorch 2에서 발표된 torch.compile 기능을 구현하기 위한 것입니다. TorchDynamo는 Python 수준의 JIT(Just-in-Time) 컴파일러로, Python의 유연성을 희생하지 않으면서 PyTorch 프로그램에서 그래프 컴파일(graph compilation)을 가능하게 합니다. 이를 위해 TorchDynamo는 Python 바이트코드(bytecode)를 실행 전 동적으로 수정하고 PyTorch 연산 시퀀스를 FX 그래프로 추출한 다음, 확장 가능한 다양한 백엔드 중 하나를 사용하여 JIT 컴파일을 수행합니다. TorchInductor는 TorchDynamo의 기본 컴파일 백엔드로, PyTorch 프로그램을 OpenAI의 Triton 및 CPU용 C++ 코드로 변환(translate)합니다. 실험 결과, TorchDynamo는 최소한의 오버헤드만으로 이전의 접근 방식보다 더 견고(robust)하게 그래프를 캡쳐할 수 있으며, TorchInductor는 MVIDIA A100 GPU에서 180개 이상의 실제 사용 모델(180+ real-world models)에 대해서 학습 시 1.14배와 추론 시 2.27배의 평균적 속도 향상(기하 평균, geometric mean)을 보이는 것으로 나타났습니다. 이러한 확장 기능들은 PyTorch와 같은 Eager 방식의 프레임워크에서 컴파일러를 통해 최적화를 적용하는 새로운 방법을 제공합니다.
> This paper introduces two extensions to the popular PyTorch machine learning framework, TorchDynamo and TorchInductor, which implement the torch.compile feature released in PyTorch 2. TorchDynamo is a Python-level just-in-time (JIT) compiler that enables graph compilation in PyTorch programs without sacrificing the flexibility of Python. It achieves this by dynamically modifying Python bytecode before execution and extracting sequences of PyTorch operations into an FX graph, which is then JIT compiled using one of many extensible backends. TorchInductor is the default compiler backend for TorchDynamo, which translates PyTorch programs into OpenAI's Triton for GPUs and C++ for CPUs. Results show that TorchDynamo is able to capture graphs more robustly than prior approaches while adding minimal overhead, and TorchInductor is able to provide a 2.27x inference and 1.41x training geometric mean speedup on an NVIDIA A100 GPU across 180+ real-world models, which outperforms six other compilers. These extensions provide a new way to apply optimizations through compilers in eager mode frameworks like PyTorch.


### 논문 저자 (Authors)

Jason Ansel (Meta); Edward Yang (Meta); Horace He (Meta); Natalia Gimelshein (OpenAI); Animesh Jain (Meta); Michael Voznesensky (Meta); Bin Bao (Meta); Peter Bell (Quansight); David Berard (Meta); Evgeni Burovski Quansight; Geeta Chauhan (Meta); Anjali Chourdia (Meta); Will Constable (Meta); Alban Desmaison (Meta); Zachary DeVito (Meta); Elias Ellison (Meta); Will Feng (Meta); Jiong Gong (Intel); Michael Gschwind (Meta); Brian Hirsh (Meta); Sherlock Huang (Meta); Kshiteej Kalambarkar (Quansight); Laurent Kirsch (Meta); Michael Lazos (Meta); Mario Lezcano (Quansight); Yanbo Liang (Meta); Jason Liang (Meta); Yinghai Lu (Meta); CK Luk (Meta); Bert Maher (Meta); Yunjie Pan (University of Michigan); Christian Puhrsch (Meta); Matthias Reso (Meta); Mark Saroufim (Meta); Marcos Yukio Siraichi (Quansight); Helen Suk (Meta); Michael Suo (Meta); Phil Tillet (OpenAI); Eikan Wang (Intel); Xiaodong Wang (Meta); William Wen (Meta); Shunting Zhang (Meta); Xu Zhao (Meta); Keren Zhou (OpenAI & George Mason University); Richard Zou (Meta); Ajit Mathews (Meta); Gregory Chanan (Meta); Peng Wu (Meta); Soumith Chintala (Meta)

### ASPLOS'24 - Full Day Tutorial Schedule

ASPLOS'24의 파이토치 2(PyTorch 2) 튜토리얼은 4월 27일, 토요일에 진행됩니다. 자세한 일정은 [여기](https://github.com/pytorch/workshops/tree/master/ASPLOS_2024)를 참고해 주세요.
> Full schedule for the ASPLOS'24 PyTorch 2 Tutoral on Saturday, April 27th is available [here](https://github.com/pytorch/workshops/tree/master/ASPLOS_2024)
