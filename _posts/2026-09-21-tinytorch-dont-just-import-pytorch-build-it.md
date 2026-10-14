---
layout: blog_detail
title: "TinyTorch: PyTorch를 import만 하지 말고, 직접 만들어 보세요"
author: Vijay Janapa Reddi (Harvard University and ETH Zurich), Andrea Mattia Garavagno (ETH Zurich)
authors:
  - Vijay Janapa Reddi
  - Andrea Mattia Garavagno
ext_author: Junghwan Park (박정환)
category: ["pytorch.org", "translation"]
date: 2026-09-21 12:00:00
image: /assets/blog/2026-09-21-tinytorch-dont-just-import-pytorch-build-it/hero.png
org_title: "TinyTorch: Don’t Just Import PyTorch. Build It."
org_link: https://pytorch.org/blog/tinytorch-dont-just-import-pytorch-build-it/
---

![TinyTorch: PyTorch를 import만 하지 말고, 직접 만들어 보세요 대표 이미지 / TinyTorch: Don’t Just Import PyTorch. Build It.](/assets/blog/2026-09-21-tinytorch-dont-just-import-pytorch-build-it/hero.png){:style="width:100%"}

*텐서부터 트랜스포머까지, 직접 작성하는 프레임워크*
> *A framework you write yourself, tensors through transformers*

## 요약 / TL;DR

성숙한 시스템 프로젝트는 결국 모두 교육용 버전이 필요해집니다. TinyTorch는 텐서부터 트랜스포머까지 동작하는 ML 프레임워크를 PyTorch 자체의 API를 사용해 순수 Python으로 처음부터 직접 만들어 보는 무료 오픈소스 커리큘럼(curriculum)입니다. 모듈은 20개입니다. RAM 4GB에 GPU가 없는 노트북에서도 실행됩니다. [tinytorch.ai](https://tinytorch.ai/)에서 볼 수 있습니다.
> Every mature systems project eventually needs a teaching version. TinyTorch is a free, open-source curriculum where you build a working ML framework from scratch, tensors through transformers, in pure Python, using PyTorch’s own API. Twenty modules. Runs on a laptop with 4 GB of RAM and no GPU. It is at [tinytorch.ai](https://tinytorch.ai/).

이 글의 나머지 부분에서는 왜 이런 것이 있어야 했다고 생각하는지, 그리고 6년 동안 운영하며 배운 것 중 오픈 커리큘럼 작업을 하는 다른 사람들에게 쓸모 있을 만한 것을 이야기합니다.
> The rest of this post is why we think it needed to exist, and what six years of running it taught us that might be useful to anyone else doing open curriculum work.

## 모든 시스템 분야는 결국 교육용 버전을 만듭니다 / Every Systems Field Eventually Builds Its Teaching Version

Unix가 머릿속에 다 담기에는 너무 커지자, Andrew Tanenbaum은 MINIX를 만들었습니다. 학생이 실제로 끝까지 해낼 수 있을 만큼 작았습니다. MINIX는 한 세대의 시스템 엔지니어에게 큰 영향을 주었고, 잘 알려진 것처럼 Linux에도 영감을 주었습니다.
> Unix got too big to hold in your head, so Andrew Tanenbaum wrote MINIX. Small enough that a student could actually finish it. It went on to shape a generation of systems engineers and famously inspired Linux.

컴파일러도 같은 길을 걸었습니다. LLVM과 GCC는 수십 년에 걸친 훌륭한 엔지니어링의 결과물이지만, 처음 읽는 교재로는 거의 읽을 수 없는 수준이라 강의에서는 대신 Tiger 컴파일러를 가르칩니다. MIT가 xv6를 x86에서 RISC-V로 다시 작성한 것도 같은 이유였습니다. 역사적으로 쌓인 복잡성을 걷어내고 깔끔한 추상화를 드러내기 위해서였습니다. 이들보다 앞서서는 Nachos와 Pintos, 그리고 SICP가 있었습니다.
> Compilers went the same way. LLVM and GCC are decades of excellent engineering and close to unreadable as a first text, so courses teach the Tiger compiler instead. MIT rewrote xv6 from x86 to RISC-V for the same reason, stripping out historical complexity to expose clean abstractions. Before all of them, Nachos and Pintos and SICP.

이 중 어느 것도 프로덕션 시스템을 대체하려던 것이 아닙니다. 이들은 프로덕션 시스템이 감출 수밖에 없는 것을 가르쳤습니다.
> None of these were trying to replace the production system. They taught what production systems have to hide.

이제 PyTorch가 그 지점에 와 있고, 이는 칭찬입니다. PyTorch 내부 구조에 대해서는 정말 좋은 글이 있고, 대부분은 그 내부를 직접 작성한 사람들이 썼습니다. 다만 그 글들은 거의 모두 독자가 이미 프레임워크 엔지니어처럼 생각한다고 가정합니다. 빠져 있던 것은 그보다 한 단계 아래의 입문 과정입니다. 직접 만들어 보는 것이면서, 대학이 학점 과목으로 실제 채택할 수 있는 채점 인프라를 갖추고, 독학하는 사람도 GPU 예산 없이 끝까지 해낼 수 있는 것입니다.
> PyTorch is at that point now, which is a compliment. There is genuinely good writing on its internals, most of it by the people who wrote them, and nearly all of it assumes you already think like a framework engineer. What has been missing is the rung below that. Something you build yourself, with grading infrastructure a university can actually adopt for credit, that a self-learner can get through without a GPU budget.

## 학생만이 아니라 PyTorch에도 중요한 이유 / Why This Matters to PyTorch, Not Only to Students

이 주장에는 교육자만 관심을 가질 버전이 있습니다. 이 글은 그 버전이 아닙니다.
> There is a version of this argument that only educators care about. This is not that version.

모든 프레임워크는 그것을 안쪽에서부터 이해하고 따져 볼 수 있는 소수의 사람들 위에서 돌아갑니다. 텐서 캐싱에서 메모리 누수를 찾아내는 사람, 변화도 체크포인팅(gradient checkpointing)이 재계산 비용을 감수할 만한 때를 아는 사람, 느린 학습 실행을 보고 프로파일러(profiler)를 열기도 전에 병목을 짚어내는 사람들입니다.
> Every framework runs on a small population of people who can reason about it from the inside. The ones who spot a memory leak in tensor caching, who know when gradient checkpointing is worth the recompute, who can look at a slow training run and name the bottleneck before they open a profiler.

이런 사람들은 저절로 늘어나지 않습니다. 지금은 대부분의 사람이 우연히 PyTorch 내부에 도달합니다. 무언가가 크게 고장 나서 어쩔 수 없이 들어가게 되는 것입니다. 이들은 마감에 쫓기며, 바깥에서 안쪽으로, 버그가 요구하는 순서대로 코드베이스를 배웁니다. 시스템 엔지니어를 온보딩해 본 사람이라면 그 결과를 압니다. 군데군데 구멍이 난 역량입니다.
> That population does not grow by itself. Right now most people get to PyTorch’s internals by accident, because something broke badly enough to force the trip. They learn the codebase under deadline pressure, from the outside in, in whatever order the bug happened to demand. Anyone who has onboarded a systems engineer knows the result, which is competence with holes in it.

직접 만들어 보면 도달하는 경로가 바뀌고, 그 변화는 영구적입니다. Autograd를 한 번 구현하고 나면 계산 그래프(computational graph)가 계속 눈에 보입니다. 자신의 메모리 할당을 한 번 프로파일링하고 나면 그 비용을 잊을 수 없습니다.
> Building the thing yourself changes the arrival path, and it changes it permanently. Once you have implemented autograd you cannot unsee the computational graph. Once you have profiled your own memory allocation you cannot unknow the cost.

![같은 선형 계층을 두 가지 방식으로 작성한 비교 / The same linear layer, two ways](/assets/blog/2026-09-21-tinytorch-dont-just-import-pytorch-build-it/Figure1.png){:style="width:100%"}
*그림 1: 같은 선형 계층(linear layer)을 두 가지 방식으로 작성한 모습. 왼쪽은 동작하지 않게 되는 순간까지만 잘 동작하는 프레임워크 호출입니다. 오른쪽은 직접 작성해서, 무언가 고장 났을 때 열어 볼 수 있는 버전입니다. 차이는 문법이 아니라, 추상화가 벽인가 문인가에 있습니다. / Figure 1: The same linear layer, two ways. On the left, the framework call that works right up until it does not. On the right, the version you wrote and can open when something breaks. The difference is not syntax, it is whether the abstraction is a wall or a door.*

`backward()`를 직접 작성해 보고, 모멘텀(momentum)과 분산(variance) 버퍼를 직접 할당하며 Adam의 메모리 사용량이 세 배로 늘어나는 것을 지켜본 학생은, 이미 머릿속에 모델을 갖춘 상태로 PyTorch의 실제 Autograd를 만납니다. 이 학생은 프로덕션 코드를 자신이 이해하는 것의 더 정교한 버전으로 읽습니다. 그만큼 온보딩 비용이 적게 드는 엔지니어입니다.
> A student who has written `backward()` themselves, who allocated the momentum and variance buffers and watched Adam’s memory footprint triple, shows up to PyTorch’s real autograd with the mental model already loaded. They read the production code as a more sophisticated version of something they understand. That is a cheaper engineer to onboard.

예상하지 못했던 부분은 기업들도 이것을 원한다는 점입니다. 여러 팀이 TinyTorch를 2~3주짜리 집중 신입 온보딩 과정으로, 한 분기에 걸친 사내 교육으로, 그리고 특정 디버깅 워크숍으로 사용했습니다. 디버깅 워크숍에서는 누군가가 Autograd를 다루는 모듈 06이나 어텐션을 다루는 모듈 12를 풀어 봅니다. 그 하위 시스템에서 계속 막히기 때문입니다. 어딘가에 수강 등록을 할 필요도 없습니다.
> The part we did not anticipate is that companies want this too. Teams have used TinyTorch for new-hire onboarding as a two to three week intensive, as internal training spread across a quarter, and as targeted debugging workshops where somebody works through Module 06 on autograd or Module 12 on attention because that is the subsystem they keep losing to. You do not have to be enrolled anywhere.

## 무엇을 만들었나 / What We Built

4개 단계(tier)로 나뉜 20개 모듈을 `tito`라는 CLI로 진행하며, 어려운 부분을 비워 둔 Jupyter 노트북 형태로 제공해 직접 채워 넣게 합니다. 필요한 것은 Python과 NumPy에 익숙한 정도입니다. GPU, 클라우드 계정, ML 시스템에 대한 사전 배경지식은 필요하지 않습니다.
> Twenty modules in four tiers, driven by a CLI called `tito`, delivered as Jupyter notebooks with the hard parts cut out for you to fill in. You need Python and to be comfortable with NumPy. You do not need a GPU, a cloud account, or any prior ML systems background.

![TinyTorch의 4개 단계 / The four tiers of TinyTorch](/assets/blog/2026-09-21-tinytorch-dont-just-import-pytorch-build-it/Figure2.png){:style="width:100%"}
*그림 2: 4개 단계. 각 단계는 아래 단계에 의존하므로, 최적화할 대상인 학습 루프를 만들지 않고는 최적화 단계로 건너뛸 수 없습니다. 기초(Foundation) 단계는 반 학기 단위 과목에 맞고, 20개 모듈 전체는 4학점 과목에 맞으며, 독학하는 사람은 몇 주간 집중해서 끝내기도 하고 몇 달에 걸쳐 여유 있게 진행하기도 합니다. / Figure 2: The four tiers. Each one depends on the tier below it, so you cannot skip ahead to optimization without having built the training loop you are optimizing. Foundation fits a half-semester module, all twenty fit a four-credit course, and self-paced learners take anywhere from a few intense weeks to several unhurried months.*

교육적인 효과의 대부분은 세 가지 설계 결정에서 나옵니다.
> Three design decisions do most of the pedagogical work.

**첫날부터 시스템을 다룹니다.** 모듈 01은 행렬 곱셈보다 `memory_footprint()` 메서드를 먼저 제공합니다. ImageNet 이미지 32장짜리 배치 하나가 19MB를 차지한다는 것을 어딘가에서 읽어서가 아니라 직접 계산해서 알게 됩니다. 나중에는 Adam이 SGD보다 옵티마이저 메모리를 대략 3배 더 필요로 한다는 것을 알게 되는데, 그 버퍼를 직접 할당하고 측정해 보았기 때문입니다.
> **Systems from day one.** Module 01 ships a `memory_footprint()` method before it ships matrix multiplication. You learn that one batch of 32 ImageNet images costs 19 MB by computing it, not by reading it somewhere. Later you find out Adam needs roughly 3× the optimizer memory of SGD, because you allocated those buffers yourself and then measured them.

**점진적 공개(progressive disclosure).** `Tensor` 클래스는 모듈 05까지 깔끔하게 유지되며, 데이터 레이아웃과 산술 연산을 어지럽히는 변화도 관련 장치가 없습니다. 그러다 모듈 06에서 `enable_autograd()`를 구현하는데, 이 함수는 이미 이해하고 있는 클래스에 `requires_grad`, `.grad`, `.backward()`를 붙여 넣습니다. 기존 코드는 그대로 동작합니다.
> **Progressive disclosure.** The `Tensor` class stays clean through Module 05, with no gradient machinery cluttering up data layout and arithmetic. Then in Module 06 you implement `enable_autograd()`, which bolts `requires_grad`, `.grad`, and `.backward()` onto the class you already understand. Your old code keeps working.

이 부분은 오래 고민했습니다. 세련된 방법은 상속입니다. 실제로 내놓은 것은 런타임 몽키 패칭(monkey-patching)인데, 이 글을 읽는 누군가는 불쾌하게 여길 것입니다. 그래도 이 방식을 택한 이유는 20개 모듈 전체에서 `Tensor` 클래스를 두 개가 아니라 하나로 유지할 수 있고, 텐서가 눈에 보이게 새로운 능력을 얻는 순간이 학생들이 기억하는 장면이 된다는 것을 알게 되었기 때문입니다. 공교롭게도 이는 PyTorch 0.4와도 닮았습니다. PyTorch 0.4는 대략 같은 이유로 `Variable`을 `Tensor`에 통합했습니다.
> We went back and forth on this one. The tasteful way to do it is inheritance. What we shipped is runtime monkey-patching, which is going to offend somebody reading this. It won because it keeps one `Tensor` class across all twenty modules instead of two, and because the moment your tensors visibly grow new powers turns out to be the thing students remember. It also happens to mirror PyTorch 0.4, which merged `Variable` into `Tensor` for roughly the same reason.

![모듈 06에서 만나는 행렬 곱셈의 변화도 구현 과제 / The gradient of matrix multiplication in Module 06](/assets/blog/2026-09-21-tinytorch-dont-just-import-pytorch-build-it/Figure3.png){:style="width:100%"}
*그림 3: 학습자가 모듈 06에서 만나는 행렬 곱셈의 변화도. 독스트링(docstring)이 수학적 규칙을 알려 주고 번호가 매겨진 접근 방법이 이를 단계로 나누어 주지만, 구현은 직접 해야 합니다. 20개 모듈이 모두 이런 모습입니다. / Figure 3: The gradient of matrix multiplication, as a learner meets it in Module 06. The docstring gives you the mathematical rule and the numbered approach breaks it into steps, but the implementation is yours. All twenty modules look like this.*

**만들어서 검증합니다.** 6개의 역사적 마일스톤(milestone)이 구현이 제대로 동작함을 증명합니다. 1958년 Rosenblatt의 퍼셉트론(Perceptron), 1969년 XOR 위기, 1986년 역전파(backpropagation)의 부활, 직접 만든 네트워크가 CIFAR-10에서 75%를 넘어야 하는 1998년 CNN의 돌파구, 2017년 트랜스포머, 그리고 마지막으로 MLPerf 방식의 벤치마킹입니다. 역사는 동기를 주기 위해 있습니다. 과제 성능은 단위 테스트보다 훨씬 속이기 어렵기 때문에 있습니다. 네트워크가 학습하니까, 여러분의 Autograd가 올바른 것입니다.
> **Build to validate.** Six historical milestones prove your implementation works. Rosenblatt’s Perceptron in 1958, the XOR crisis in 1969, the backpropagation revival in 1986, the CNN breakthrough in 1998 where your network has to clear 75% on CIFAR-10, the transformer in 2017, and finally MLPerf-style benchmarking. The history is there for motivation. The task performance is there because it is a much harder thing to fake than a unit test. Your autograd is correct because your network learns.

![마일스톤 사다리 / The milestone ladder](/assets/blog/2026-09-21-tinytorch-dont-just-import-pytorch-build-it/Figure4.png){:style="width:100%"}
*그림 4: 마일스톤 사다리. 마일스톤은 그 아래 모듈들이 동작하는 구현을 만들어 낼 때만 열리므로, 이 연대표는 진도 추적기이자 정확성 증명 역할을 함께 합니다. 학생들은 직접 작성한 코드만으로 67년의 ML 역사를 재현합니다. / Figure 4: The milestone ladder. A milestone unlocks only when the modules beneath it produce a working implementation, so the timeline doubles as a progress tracker and a correctness proof. Students recreate 67 years of ML history running nothing but code they wrote.*

TinyTorch는 전반에 걸쳐 의도적으로 PyTorch의 API를 그대로 따릅니다. API가 지식을 옮겨 주는 수단이고, 이 수단은 양방향으로 작동합니다. 여기서 `loss.backward()`를 만들어 본 사람은 나중에 PyTorch의 구현을 열었을 때 그 형태를 알아봅니다. PyTorch 개발자는 오후 한나절이면 TinyTorch의 어텐션 모듈을 읽고 그래프가 어떻게 구성되는지 볼 수 있습니다.
> Throughout, TinyTorch mirrors PyTorch’s API deliberately. The API is the transfer mechanism and it carries both ways. Somebody who builds `loss.backward()` here can open PyTorch’s version afterward and recognize the shape of it. A PyTorch developer can read our attention module in an afternoon and see how the graph gets constructed.

![TinyTorch 학습 루프와 PyTorch 학습 루프 비교 / A TinyTorch training loop next to the equivalent PyTorch loop](/assets/blog/2026-09-21-tinytorch-dont-just-import-pytorch-build-it/Figure5.png){:style="width:100%"}
*그림 5: TinyTorch 학습 루프와 그에 해당하는 PyTorch 루프를 나란히 놓은 모습. import 문만 다르고 나머지는 거의 같은데, 바로 그것이 핵심입니다. 여기서 배운 것 중 나중에 잊어야 할 것은 없습니다. / Figure 5: A TinyTorch training loop next to the equivalent PyTorch loop. The imports differ and almost nothing else does, which is the entire point. Nothing you learn here has to be unlearned later.*

TinyTorch가 아닌 것도 두 가지 있습니다. 닮은 점은 API 표면에서 끝나므로, 디스패처(dispatcher)도, C++이나 CUDA 계층도, JIT도, 분산 처리도 없습니다. 그리고 느립니다. 순수 Python은 PyTorch보다 대략 100배에서 10,000배 느린데, 이 점이 중요한 것으로 드러났기 때문에 뒤에서 다시 이야기합니다.
> Two things it is not. The resemblance stops at the API surface, so there is no dispatcher, no C++ or CUDA layer, no JIT, nothing distributed. And it is slow. Pure Python runs somewhere between 100 and 10,000 times slower than PyTorch, which we will come back to, because it turned out to matter.

## 어디에서 시작되었나 / Where It Came From

TinyTorch는 프레임워크로 시작하지 않았습니다. 문제를 안고 있던 강의로 시작했습니다.
> TinyTorch did not start as a framework. It started as a course with a problem.

CS 249r은 2020년 Harvard에서 TinyML을 다루는 대학원 세미나로 개설되었는데, 지정할 교재가 없었습니다. 그래서 강의 노트가 교재가 되었습니다. 이 책을 오픈 저장소로 공개하자 학생과 교육자들이 예제를 고치고 새로운 장을 제안하기 시작했고, 사람들이 대규모 학습에 대해 계속 물어 온 덕분에 2024년에는 TinyML이라는 출발점을 넘어설 만큼 커졌습니다. 결국 책은 두 권으로 나뉘었습니다.
> CS 249r launched at Harvard in 2020, a graduate seminar on TinyML, and there was no textbook to assign. So the course notes became one. We put the book up as an open repository, students and educators started fixing examples and proposing chapters, and by 2024 it had outgrown its TinyML origins because people kept asking about training at scale. It split into two volumes.

그 과정 어딘가에서 교재만으로는 부족하다는 것이 분명해졌습니다. Autograd에 대해 읽는 것과 Autograd를 구현하는 것은 서로 다른 종류의 지식을 만들어 내고, 그중 하나만 프로덕션 버그를 만났을 때 살아남습니다. 그래서 프로젝트는 여러 갈래로 뻗어 나갔습니다. TinyTorch는 직접 만드는 갈래입니다. 지금은 그 주변에 트레이드오프(trade-off)를 직접 건드려 보는 Marimo 실습, 성능 모델링을 위한 MLSys·im, Arduino와 Raspberry Pi용 하드웨어 키트, 면접 준비를 위한 StaffML, 그리고 슬라이드와 강의 계획서를 갖춘 강사 허브가 있습니다.
> Somewhere in there it became obvious a textbook was not enough. Reading about autograd and implementing autograd produce different kinds of knowledge, and only one of them survives contact with a production bug. So the project grew limbs. TinyTorch is the build limb. Around it now sit Marimo labs for poking at trade-offs, MLSys·im for performance modeling, hardware kits for Arduino and Raspberry Pi, StaffML for interview prep, and an instructor hub with slides and syllabi.

![TinyTorch 커뮤니티 지도 / The TinyTorch community map](/assets/blog/2026-09-21-tinytorch-dont-just-import-pytorch-build-it/Picture6.png){:style="width:100%"}
*그림 6: TinyTorch 커뮤니티 지도. 2025년 12월 출시 이후 92개 기관에서 682명이 참여했습니다. 흥미로운 것은 분포입니다. GPU를 쓰기 가장 어려운 곳에서 가장 많이 사용되는데, 접근성 하한선(accessibility floor)을 둔 이유가 바로 이것이었습니다. / Figure 6: The TinyTorch community map, 682 members across 92 institutions since it launched in December 2025. The clustering is the interesting part. Uptake is heaviest where GPU access is hardest, which is what the accessibility floor was for.*

그리고 적어 두기에 조금 민망한 부분이 있습니다.
> Then there is the part that is slightly embarrassing to write down.

5년 동안 이 프로젝트는 느리게 자랐습니다. 꾸준했고, 입소문으로 퍼졌고, 1년에 스타(star)가 수백 개씩 늘었습니다. 눈앞의 학생들에게 필요하니까 계속하게 되는 종류의 일이었습니다. 2025년 8월에 저장소의 스타는 약 2,000개였습니다.
> For five years this was a slow project. Steady, word of mouth, a few hundred stars a year, the kind of thing you keep doing because the students in front of you need it. In August 2025 the repository had about 2,000 stars.

10월에, 한 번도 만난 적 없는 누군가가 X에 이 프로젝트를 올렸습니다.
> In October, somebody we had never met posted it on X.

2주 뒤 스타는 5,000개가 되었습니다. 12월에는 10,000개를 넘었고, 지금은 27,000개가 넘으며, 기여자는 95명 이상, 이 커리큘럼으로 강의를 운영하는 대학은 50곳 이상입니다. 이것을 저희가 설계해서 이뤄냈다고 말하고 싶지만, 그렇지 않습니다. 영향력 있는 한 사람이 이 작업이 공유할 가치가 있다고 판단했고, 5년간 조용히 쌓인 것이 약 11개월 만에 터져 나왔습니다.
> Two weeks later we were at 5,000. By December we had passed 10,000, and today it is above 27,000, with 95 or more contributors and courses running at 50 or more universities. We would like to tell you we engineered that. We did not. One person with reach decided the work was worth sharing, and five years of quiet accumulation went off in about eleven months.

![Machine Learning Systems 저장소의 스타 수 추이 / Stars on the Machine Learning Systems repository](/assets/blog/2026-09-21-tinytorch-dont-just-import-pytorch-build-it/Figure7.png){:style="width:100%"}
*그림 7: Machine Learning Systems 저장소의 스타 수. 모두가 눈여겨보는 것은 절벽처럼 치솟은 부분입니다. 하지만 실제 작업은 평평한 부분에서 이루어졌습니다. / Figure 7: Stars on the Machine Learning Systems repository. The cliff is the part everyone looks at. The flat part is where the work happened.*

TinyTorch는 Harvard에서 [Machine Learning Systems](https://mlsysbook.ai/) 프로젝트의 일부로 만들어졌고, 지금은 Andrea가 ETH Zurich에서 유지보수하고 있습니다. 저희 두 사람 모두 현재 ETH Zurich에 있습니다. 커리큘럼을 두 번째 기관에 넘기는 일은, 그것을 자기 강의실 너머의 누군가를 위해 썼는지 확인하는 첫 번째 정직한 시험입니다.
> TinyTorch was built at Harvard as part of the [Machine Learning Systems](https://mlsysbook.ai/) project, and Andrea now maintains it from ETH Zurich, where both of us are currently based. Handing a curriculum to a second institution is the first honest test of whether you wrote it for anyone beyond your own classroom.

## 다른 프로젝트에도 통한 교훈 / What Transferred to Other Projects

이 가이드라인은 다른 기관이 적용할 수 있는 교훈을 요청하는데, 마땅히 요청할 만한 것입니다. 네 가지를 소개합니다.
> These guidelines ask for lessons other institutions can apply, which is the right thing to ask for. Four of them.

**프로덕션 API와 정확히 맞추세요.** 저희가 내린 결정 중 가장 효과가 컸던 것은 독자적인 문법을 만들지 않기로 한 것입니다. 학습자가 교육용 API와 실제 API 사이를 번역하는 데 쓰는 시간은 학습자에게 아무것도 남기지 않고, 여러분은 그때마다 학습자 일부를 잃게 됩니다. API 호환성은 있으면 좋은 정도로 여기지 말고, 반드시 지켜야 할 요구사항으로 다루세요.
> **Match the production API exactly.** The highest-leverage decision we made was refusing to invent our own syntax. Every hour a learner spends translating between your teaching API and the real one is an hour that buys them nothing and costs you a fraction of them. Treat API compatibility as a hard requirement, not a nice-to-have.

**기능 목록보다 하드웨어 하한선을 먼저 정하세요.** TinyTorch는 2GHz 듀얼 코어 CPU와 4GB RAM에서, 학습 중 네트워크 연결 없이 실행됩니다. 작은 오프라인 데이터셋 두 개(흑백 숫자 이미지 약 1,000개와 대화형 질문-답변 쌍 350개, 합쳐서 50MB 미만)를 함께 제공하기 때문입니다. 그 대가로 GPU 지원을 포기했습니다. 대신 Chromebook을 쓰는 교실, 5년 된 노트북, 네트워크 정책 때문에 다른 방식은 막혔을 대학들을 얻었습니다. 누구를 배제할 각오가 되어 있는지 시작하기 전에 정하세요. 이 결정은 나중에 다시 바꾸려면 비용이 커집니다.
> **Decide your hardware floor before your feature list.** TinyTorch runs on a dual-core 2 GHz CPU with 4 GB of RAM and no network during training, because we ship two tiny offline datasets (about 1,000 grayscale digits and 350 conversational question-answer pairs, under 50 MB together). That cost us GPU support. It bought us Chromebook classrooms, five-year-old laptops, and universities whose network policies would have blocked anything else. Work out who you are willing to exclude before you start, because that call gets expensive to revisit.

저희가 양해를 구했던 느린 속도는 프로젝트에서 가장 좋은 우연이 되었습니다. PyTorch가 10밀리초에 처리하는 배치를 학생의 `Conv2d`가 97초 걸려 처리하면, 벡터화(vectorization)가 필요한 이유는 읽어서 아는 것이 아니라 직접 겪은 일이 됩니다.
> The slowness we apologized for turned out to be the best accident in the project. When a student’s `Conv2d` takes 97 seconds on a batch that PyTorch clears in 10 milliseconds, the argument for vectorization stops being something they read and starts being something that happened to them.

**진짜 병목은 강사용 인프라입니다.** 좋은 콘텐츠라고 채택되지는 않습니다. 채점할 수 있는 콘텐츠가 채택됩니다. 저희는 잠긴 테스트 셀과 배점이 있는 NBGrader 자동 채점, 설정과 채점 기준표(rubric)와 학생들이 실제로 겪는 오류를 다루는 `INSTRUCTOR.md`, 마일스톤 검증 스크립트, 그리고 세 가지 통합 모델을 제공했습니다. 세 가지 통합 모델 덕분에 학과는 20개 모듈을 4학점 과목으로, 8개 모듈을 반 학기 단위로, 또는 최적화 단계만 떼어 엣지(edge) 세미나로 운영할 수 있습니다. 학습자 경로만큼 강사 경로에도 예산을 들이세요. 커리큘럼 위원회 앞에서 그 선택을 변호해야 하는 사람은 강사이고, 강사가 변호하지 못하면 교육의 질은 아무 의미가 없습니다.
> **Instructor infrastructure is the actual bottleneck.** Good content does not get adopted. Gradeable content gets adopted. We shipped NBGrader autograding with locked test cells and point allocations, an `INSTRUCTOR.md` covering setup and rubrics and the errors students actually hit, milestone validation scripts, and three integration models so a department can run twenty modules as a four-credit course, eight as a half-semester unit, or the optimization tier alone as an edge seminar. Budget as much for the instructor path as the learner path. The instructor is the one who has to defend the choice to a curriculum committee, and none of the teaching quality matters if they cannot.

**필요해지기 전에 승계 계획을 세우세요.** 학계의 오픈소스는 책임 연구자(PI)가 관심사를 바꾸면 사라집니다. 저희는 이 문제를 공개적으로 다루고 있습니다. 2027년까지의 유지보수 약속, 풀 리퀘스트 2주 내 리뷰 목표, 그리고 2026년부터 2027년에 걸쳐 교육자 자문 위원회를 만들고 누가 이어받을지 문서로 남기는 거버넌스 전환이 그것입니다. 지속 가능성은 더 열심히 일하는 문제가 아닙니다. 자신을 대체 가능하게 만드는 문제이고, 거기에는 날짜가 필요합니다.
> **Plan succession before you need it.** Academic open source dies when the PI changes focus. We are handling that in the open, with a maintenance commitment through 2027, a two-week pull request review target, and a governance transition across 2026 and 2027 that sets up an educator advisory board and writes down who takes over. Sustainability is not a working-harder problem. It is a making-yourself-replaceable problem, and it needs a date on it.

## 아직 모르는 것 / What We Still Do Not Know

TinyTorch는 프리뷰 단계이며 2026년 가을 학기 강의에 쓸 수 있는 수준을 목표로 하고 있습니다. 깔끔한 성공담보다 한계가 여러분에게 더 쓸모 있습니다.
> TinyTorch is in preview and aimed at classroom readiness for Fall 2026. The limits are worth more to you than a clean success story.

학습 성과는 아직 측정하지 않았습니다. 설계는 구성주의(constructionism), 인지적 도제(cognitive apprenticeship), 생산적 실패(productive failure), 문턱 개념(threshold concepts), 그리고 시스템 교육에서 직접 만들어 보는 방식이 효과적이라는 50년간의 근거에 기대고 있습니다. 이는 강력한 사전 근거(prior)입니다. 하지만 결과는 아닙니다. TinyTorch로 배운 학생이 기존 강의를 들은 학생보다 프로덕션 시스템을 더 잘 디버깅한다는 통제된 데이터는 없으며, 이를 얻으려면 직접 강의를 운영해 보고 어떤 일이 있었는지 알려 줄 강사들이 필요합니다.
> We have not measured learning outcomes. The design leans on constructionism, cognitive apprenticeship, productive failure, threshold concepts, and five decades of evidence that build-it-yourself works in systems education. That is a strong prior. It is not a result. We do not have controlled data showing TinyTorch students debug production systems better than students who took a conventional course, and getting it needs instructors willing to run the thing and tell us what happened.

범위도 목표보다 좁습니다. 단일 노드, CPU 전용이므로 메모리와 연산은 잘 가르치지만 GPU 커널, 분산 학습, 변화도 동기화에 대해서는 아무것도 가르치지 않습니다. 병렬 데이터 로딩과 GPU 메모리 관리는 역량 영역으로 정리까지 해 두고서 일부러 비워 둔 유일한 영역입니다. 제대로 다루면 4GB 하한선이 깨지기 때문입니다. 이는 실제로 비어 있는 부분이고, 로드맵이 아니라 사람이 필요합니다.
> The scope is also narrower than the ambition. Single-node, CPU-only, so it teaches memory and compute well and teaches nothing about GPU kernels, distributed training, or gradient synchronization. Parallel data loading and GPU memory management are the one competency area we mapped out and then deliberately left empty, because doing them properly breaks the 4 GB floor. Those are real holes and they need people, not a roadmap.

## 도움을 줄 수 있는 부분 / Where You Can Help

일단 써 보고 싶다면, 설치는 한 줄이면 되고 모든 것이 로컬에서 실행됩니다.
> If you would rather just try it, installation is one line and everything runs locally.

```sh
curl -sSL mlsysbook.ai/tinytorch/install.sh | bash
```

참여할 수 있는 다섯 가지 방법을, 대략 영향이 큰 순서대로 소개합니다.

1. **실제 PyTorch 의미론(semantics)에 비추어 모듈을 리뷰해 주세요.** PyTorch 내부를 다루는 분이라면, TinyTorch의 Autograd나 옵티마이저 상태 처리, KV 캐시가 올바른 머릿속 모델을 가르치는지 한 시간만 확인해 주셔도 엄청난 가치가 있습니다. TinyTorch가 가르치는 것은 한 기수의 학생들이 PyTorch에 대해 믿는 내용이 되며, 미묘하게 틀린 것을 가르치는 것은 아예 가르치지 않는 것보다 나쁩니다. 해당 모듈과 바꾸고 싶은 점을 적어 이슈를 열거나, 노트북에 대한 풀 리퀘스트를 보내 주세요. 천 명이 잘못 배운 뒤보다 지금 듣는 편이 훨씬 낫습니다.
2. **한 단계를 시범 운영하고 무엇이 망가졌는지 알려 주세요.** 2026년 가을 학기 강의 계획서는 이미 대부분 확정되었으므로, 현실적으로는 2027년 봄이나 2027년 가을 학기가 될 것이며, 이번 가을에 자료를 미리 따라가 보는 것이 결정하는 데 좋은 방법입니다. 기초 단계는 학부 시스템 과목의 한 단위에 맞고, 20개 모듈 전체는 한 학기에 맞으며, 최적화 단계는 TinyML이나 엣지 세미나용으로 단독 운영할 수 있습니다. 성공담보다 실패 보고를 더 원합니다.
3. **저희가 만들 수 없는 모듈을 작성해 주세요.** 분산 학습, GPU 가속, 병렬 데이터 로딩입니다. 이 모듈들은 이미 그 내용을 가르치고 있는 사람이 필요합니다.
4. **현지화해 주세요.** 데이터셋은 설계상 작고 오프라인이므로, 대화형 데이터셋을 다른 언어로 번역하는 일은 주말 동안 할 수 있으면서도 파급력이 큰 프로젝트입니다.
5. **채택하고 알려 주세요.** 커뮤니티 지도에 여러분의 기관을 올리면, 다음 학과에 이것이 실험이 아니라 실제로 쓸 수 있는 선택지라는 것을 알려 줄 수 있습니다.

> Five openings, roughly in order of how much each would move things.
> 1. **Review a module against real PyTorch semantics.** If you work on PyTorch internals, an hour checking whether our autograd or our optimizer state handling or our KV cache teaches the right mental model is worth an enormous amount. Whatever TinyTorch teaches becomes what a cohort of students believes about PyTorch, and teaching something subtly wrong is worse than not teaching it. Open an issue naming the module and what you would change, or send a pull request against the notebook. We would much rather hear it now than after a thousand people have learned it.
> 2. **Pilot a tier and tell us what broke.** Fall 2026 syllabi are mostly locked by now, so realistically that is Spring 2027 or Fall 2027, and shadowing the material this fall is a good way to decide. Foundation fits an undergraduate systems module, all twenty fit a semester, the Optimization tier stands alone for a TinyML or edge seminar. We want the failure reports more than the success stories.
> 3. **Write the modules we cannot.** Distributed training, GPU acceleration, parallel data loading. These need somebody who already teaches the material.
> 4. **Localize it.** The datasets are small and offline by design, which makes translating the conversational one into another language a weekend project with real reach.
> 5. **Adopt it and say so.** Putting your institution on the community map is what tells the next department this is a real option and not an experiment.

모든 것은 [Machine Learning Systems 저장소](https://github.com/harvard-edge/cs249r_book)에 있습니다. 코드는 MIT, 커리큘럼은 CC BY-SA 4.0 라이선스이므로 여러분의 기관에 맞게 포크(fork)하고 수정하는 것은 명시적으로 허용되며, 고친 내용을 업스트림(upstream)에 반영해 주시면 감사하겠습니다.
> Everything lives in the [Machine Learning Systems repository](https://github.com/harvard-edge/cs249r_book). Code is MIT, curriculum is CC BY-SA 4.0, so forking and adapting for your own institution is explicitly fine, and upstreaming what you fix is appreciated.

이 모든 것은 무료이며 앞으로도 무료입니다. 이 글의 주장에 공감했고 가장 손쉽게 행동하고 싶다면, 저장소에 스타를 눌러 주세요. 그 숫자는 저희에게 점수판이 아닙니다. 학과장이 오픈 커리큘럼을 기반으로 강의를 개설해도 안전한지 판단할 때 보는 숫자이고, 후원자가 이런 종류의 작업이 지원받을 가치가 있는지 판단할 때 보는 숫자입니다. 여러분에게는 아무 비용도 들지 않으며, 이 목록의 다른 어떤 것보다 큰 도움이 됩니다.
> All of it is free and it stays free. If the argument here landed and you want the cheapest way to act on it, star the repository. That number is not a scoreboard for us. It is what a department chair looks at when deciding whether an open curriculum is safe to build a course on, and what a funder looks at when deciding whether this kind of work deserves support. It costs you nothing and it helps more than anything else on this list.

## PyTorch Academic OSPO 워킹 그룹에 참여하세요 / Join the PyTorch Academic OSPO Working Group

TinyTorch는 대학이 열린 협업을 통해 PyTorch 생태계를 발전시키는 한 가지 사례입니다. 이 글도 PyTorch Academic OSPO 워킹 그룹 덕분에 나올 수 있었는데, 이 워킹 그룹은 열정적인 학습자 무리가 실제 거버넌스를 갖춘 조직으로 바뀌도록 도와 왔습니다. 위의 교훈 중 몇 가지는 저희 코드가 아니라 그 대화에서 나왔습니다.
> TinyTorch is one example of universities pushing the PyTorch ecosystem forward through open collaboration, and this post exists because of the PyTorch Academic OSPO Working Group, which has been helping us turn an enthusiastic pile of learners into something with actual governance. Several of the lessons above came out of those conversations rather than out of our own code.

학계 프로젝트를 공유하거나, 모범 사례를 함께 만들거나, PyTorch와 학계가 만나는 지점에서 일하는 사람들과 교류하는 데 관심이 있다면 [PyTorch Academic OSPO 워킹 그룹](https://github.com/pytorch-fdn/wg-ospo-and-academic-outreach)에 참여해 보세요. 연구자, 교육자, 학생, 오픈소스 실무자 모두를 환영합니다.
> If you are interested in sharing academic projects, developing best practices, or connecting with people working where PyTorch meets academia, consider joining the [PyTorch Academic OSPO Working Group](https://github.com/pytorch-fdn/wg-ospo-and-academic-outreach). It welcomes researchers, educators, students, and open-source practitioners.

![PyTorch Foundation OSPO 및 학계 아웃리치 로고 / PyTorch Foundation OSPO and Academic Outreach logo](/assets/blog/2026-09-21-tinytorch-dont-just-import-pytorch-build-it/ospoOutreach_horizontalColor.png){:style="width:100%"}
