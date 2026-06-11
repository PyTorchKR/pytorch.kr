---
layout: blog_detail
title: "vLLM과 PyTorch가 함께 만든 aarch64에서의 더 나은 개발자 경험"
author: Kaichao You (Inferact)
category: ["pytorch.org", "translation"]
org_title: "vLLM and PyTorch Work Together to Improve the Developer Experience on aarch64"
org_link: https://pytorch.org/blog/vllm-and-pytorch-work-together-to-improve-the-developer-experience-on-aarch64/
---

*TL;DR: PyTorch 2.11부터는 aarch64 Linux에서도 PyPI를 통해 CUDA가 활성화된 PyTorch 휠(wheel)을 곧바로 설치할 수 있게 되었습니다. 이로써 NVIDIA GH200, GB200, GB300 등의 시스템에 배포할 때 그동안 발목을 잡아왔던 별도의 패키지 인덱스나 우회 방법(workaround)이 더 이상 필요하지 않습니다. 이 글에서는 Kaichao You(Inferact)가 이번 패키징 변경이 vLLM 사용자의 설치 경험을 어떻게 개선하는지 설명하고, PyTorch Foundation을 통한 vLLM과 PyTorch의 협업이 어떻게 이번 수정 사항을 프로덕션에 안착시켰는지 짚어봅니다.*
> *TLDR: PyTorch 2.11 makes it possible to install CUDA-enabled PyTorch wheels on aarch64 Linux directly from PyPI, eliminating the need for custom package indexes and workarounds that previously complicated deployment on systems such as NVIDIA GH200, GB200, and GB300. In this post, Kaichao You (Inferact) explains how this packaging change improves the installation experience for vLLM users and highlights how collaboration between vLLM and PyTorch through PyTorch Foundation helped bring the fix to production.*

![GB200 / GB300 / GH200에서의 PyTorch aarch64 설치 / PyTorch aarch64 install on GB200 / GB300 / GH200](/assets/blog/2026-05-19-vllm-and-pytorch-improve-aarch64-experience/unnamed-7.png){:style="width:100%"}

*GB200 / GB300 / GH200에서의 사용 경험을 한결 편하게 만들어 줄, 2년 동안 기다려온 수정 사항입니다.*
> *A fix, two years in the making, that makes life much easier on GB200 / GB300 / GH200.*


## 해커톤에서 처음 마주친 문제 / An issue I first hit at a hackathon

사실 이 이야기는 2024년 10월로 거슬러 올라갑니다.
> This story actually starts back in October 2024.

저는 당시 CUDA MODE(현재의 GPU MODE) IRL 해커톤에 참가해서 GH200 머신에 vLLM을 띄워보려고 하던 중이었습니다. 5분이면 끝났어야 할 작업이었지만, 표면적으로는 멀쩡해 보이는 `pip install` 명령어를 하루 종일 들여다보게 되었습니다. 휠은 해석되었고 의존성도 충족되었으며, 설치도 별다른 오류 없이 끝났지만, 런타임에 `torch.cuda.is_available()`는 끝까지 `False`를 반환했습니다.
> I was at the CUDA MODE (now GPU MODE) IRL hackathon, trying to get vLLM running on a GH200 box. It should have been a five-minute job. Instead, I spent a frustrating chunk of the day staring at a `pip install` that, on the surface, looked perfectly fine — wheels were resolved, dependencies were satisfied, the install completed without errors — but at runtime `torch.cuda.is_available()` stubbornly returned `False`.

원인을 파고들어 보니 어이없을 정도로 단순한 문제였습니다. `aarch64` Linux에서는 `pip install torch`가 PyPI에서 **CPU 전용** 휠을 받아오고 있었던 것입니다. 기본 PyPI 인덱스에는 `aarch64`용 GPU 휠 자체가 올라가 있지 않았고, CUDA가 포함된 빌드를 받으려면 pip에 PyTorch 다운로드 인덱스를 명시적으로 지정해줘야 했습니다.
> The reason, once I dug in, was almost comically mundane: on `aarch64` Linux, `pip install torch` was pulling the **CPU-only** wheel from PyPI. There simply was no GPU wheel for `aarch64` published to the default PyPI index. To get a CUDA-enabled build, you had to explicitly point pip at the PyTorch download index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

이 자체만 두고 보면 그저 약간 성가신 정도였습니다. 진짜 문제는 이 동작이 전이 의존성(transitive dependency)과 만났을 때 벌어졌습니다. PyPI는 패키지가 자신의 의존성을 위해 별도의 인덱스를 지정하는 것을 허용하지 않습니다. 따라서 vLLM의 의존성 트리에 있는 어떤 패키지가 `torch==<특정 버전>`을 요구하고 그 버전이 일치하지 않는 순간, pip는 곧장 기본 PyPI 인덱스로 돌아가 CPU 휠을 찾아내고, 방금 공들여 설치한 GPU 빌드를 **조용히 제거**한 뒤 CPU 빌드로 갈아치웠습니다. 모델이 GPU를 찾지 못한다고 항변하기 전까지는 모든 게 정상으로 보였습니다.
> That, by itself, would be only mildly annoying. The real damage came from how this interacted with transitive dependencies. PyPI does not let a package specify a custom index for its dependencies. So if any package in vLLM's dependency tree declared a requirement of `torch==<some_version>`, and that version doesn't match, pip would happily go back to the default PyPI index, find the CPU wheel, **silently uninstall** the GPU build I had just carefully installed, and replace it with the CPU one. You'd think everything was fine until your model refused to find a GPU.

GH200 — 그리고 이후의 GB200 / GB300 — 에서 vLLM을 띄워보려던 모든 사람에게, 이 문제는 한 줄짜리 설치 명령을 `--index-url` 플래그와 버전 고정(pin), 설치 후 점검 절차가 뒤엉킨 미로로 바꿔놓았습니다.
> For anyone trying to bring up vLLM on GH200 — and later on GB200 / GB300 — this turned a one-line install into a maze of `--index-url` flags, pinned versions, and post-install sanity checks.


## 그동안 vLLM이 들고 다닌 우회책들 / The workarounds vLLM carried in the meantime

업스트림에서 제대로 된 수정이 이루어지기 전까지, vLLM은 aarch64 사용자들이 막히지 않도록 자체적인 우회책을 들고 다녀야 했습니다.
> While we waited for a proper fix upstream, vLLM had to ship its own workarounds so that aarch64 users were not stuck.

첫 번째 우회책은 2024년 9월 [vllm-project/vllm#8713](https://github.com/vllm-project/vllm/pull/8713)에서 추가된 [use_existing_torch.py](https://github.com/vllm-project/vllm/blob/main/use_existing_torch.py)였습니다. PR 제목 자체에 *"enable existing pytorch (for GH200, aarch64, nightly)"* 라고 명시되어 있을 만큼 의도가 분명한 패치였죠. 흐름은 이름 그대로입니다. 사용자가 알맞은 `torch` 빌드(PyTorch 인덱스에서 받은 빌드든, nightly든, 직접 만든 커스텀 빌드든)를 먼저 설치한 뒤 `python use_existing_torch.py`를 실행하면, vLLM의 `requirements/*.txt`, `requirements/*.in`, `pyproject.toml` 안에 있는 모든 `torch`/`torchvision`/`torchaudio` 요구사항을 제거합니다. 이렇게 핀(pin)을 없애면, 뒤이은 vLLM 설치 과정에서 pip가 "친절하게도" 기본 PyPI 인덱스로 돌아가 CUDA가 활성화된 `torch`를 CPU 휠로 조용히 갈아치우는 사고를 더 이상 일으킬 수 없게 됩니다. 설치 시점에 자기 자신의 의존성 파일을 다시 쓴다는 점에서 보기 좋은 방식은 아니었지만, 그래도 1년이 넘는 시간 동안 GH200 사용자들의 발목을 풀어준 우회책이었습니다.
> The first one was [use_existing_torch.py](https://github.com/vllm-project/vllm/blob/main/use_existing_torch.py), added in [vllm-project/vllm#8713](https://github.com/vllm-project/vllm/pull/8713) back in September 2024 — explicitly framed in the PR title as *"enable existing pytorch (for GH200, aarch64, nightly)"*. The flow is exactly what the name suggests: you install the right `torch` build yourself (from the PyTorch index, or a nightly, or a custom build), then run `python use_existing_torch.py`, which strips every `torch`/`torchvision`/`torchaudio` requirement out of vLLM's `requirements/*.txt`, `requirements/*.in`, and `pyproject.toml`. With those pins gone, the subsequent vLLM install can no longer trigger pip to "helpfully" reach back into the default PyPI index and silently swap your CUDA-enabled `torch` for the CPU wheel. It is ugly — we are literally rewriting our own dependency files at install time — but it kept GH200 users unblocked for over a year.

이후 `uv`가 성숙하면서 좀 더 깔끔한 방법이 생겼습니다. [vllm-project/vllm#24303](https://github.com/vllm-project/vllm/pull/24303)에서 `pyproject.toml`에 다음 설정을 추가했죠.
> Later, as `uv` matured, we got a cleaner option. In [vllm-project/vllm#24303](https://github.com/vllm-project/vllm/pull/24303) we added the following to `pyproject.toml`:

```toml
[tool.uv]
no-build-isolation-package = ["torch"]
```

이 설정은 `uv`에게 `torch`를 격리된 환경에서 빌드하지 말라고 알려주는 것입니다. 실제 동작 관점에서 보면, uv가 별도로 `torch`를 해석하고 다시 설치하려 하지 않고, 현재 환경에 이미 설치된 torch를 재사용하게 만드는 효과를 가집니다. 적절한 인덱스에서 먼저 `torch`를 설치해두는 흐름과 결합하면, 파일을 다시 쓰는 트릭에 비해 훨씬 자연스러운 경로가 만들어집니다. `pyproject.toml`에 한 줄짜리 설정만 추가하면, `uv pip install vllm` (또는 `uv sync`) 명령이 aarch64에서 미리 설치된 CUDA 지원 `torch`를 그대로 존중하게 됩니다.
> This tells `uv` not to build `torch` in an isolated environment — which in practice means uv will reuse the torch already present in the current environment instead of trying to resolve and reinstall its own copy. Combined with installing `torch` first from the right index, this gave us a much more ergonomic path than the file-rewriting trick: a single config line in `pyproject.toml`, and `uv pip install vllm` (or a `uv sync`) would respect the pre-installed CUDA-enabled `torch` on aarch64.

vLLM의 우회책은, 패키징 표준의 빈틈을 커뮤니티가 임기응변으로 메워온 사례라고 볼 수 있습니다. [Wheel Variants](https://developer.nvidia.com/blog/streamline-cuda-accelerated-python-install-and-packaging-workflows-with-wheel-variants/)는 NVIDIA와 Astral이 이런 임기응변이 더 이상 필요 없도록 그 빈틈을 표준 차원에서 메우려는 시도입니다.
> The vLLM workaround is the community improvising around a gap in the packaging standard. [Wheel Variants](https://developer.nvidia.com/blog/streamline-cuda-accelerated-python-install-and-packaging-workflows-with-wheel-variants/) is NVIDIA and Astral formalizing the fix so the improvisation is no longer needed.


## 해커톤의 골칫거리에서 TAC 안건으로 / From a hackathon headache to a TAC agenda item

시간을 2025년으로 옮겨봅시다. vLLM이 PyTorch Foundation에 합류했고, 저는 기술 자문 위원회(TAC, Technical Advisory Committee)에서 vLLM을 대표하는 인원 중 한 명이 되었습니다. aarch64 휠 문제는 제 일에서도, 다른 Grace Hopper / Grace Blackwell 시스템 위의 vLLM 사용자들로부터도 계속 거론되었습니다. 2025년 8월에 저는 이 문제를 공식적으로 추적하기 위해 [pytorch/pytorch#160162](https://github.com/pytorch/pytorch/issues/160162)를 등록했고, 올해 초인 2026년 1월의 TAC 회의에서 vLLM 사용자들을 대신해 이 문제를 직접 안건으로 올렸습니다.
> Fast forward to 2025. vLLM joined the PyTorch Foundation, and I became one of its representatives on the Technical Advisory Committee (TAC). The aarch64 wheel situation kept coming up — both in my own work and from other vLLM users on Grace Hopper and Grace Blackwell systems. In August 2025, I filed [pytorch/pytorch#160162](https://github.com/pytorch/pytorch/issues/160162) to track the problem formally, and earlier this year, in a January 2026 TAC meeting, I raised it directly on behalf of vLLM users.

요청 사항은 단순했습니다. aarch64용 GPU 휠을 기본 PyPI 인덱스에 게시해서, GB200 계열 머신에서도 x86에서처럼 `pip install torch`만으로 "그냥 동작"하게 만들어 달라는 것이었죠. 이 휠들은 x86에서 이미 쓰고 있는 방식대로 NCCL이나 cuBLAS 같은 라이브러리에 동적으로 링크(dynamic linking)될 것이므로, 용량이 크게 부풀어 오를 일은 없습니다. 대용량 휠 업로드는 사용자 입장에서는 다운로드가 부담스럽고 PyPI 메인테이너에게는 호스팅 비용이 커지기 때문에, PyPI 메인테이너 측에서도 제한을 두고 강하게 자제를 권하고 있습니다.
> The ask was straightforward: publish aarch64 GPU wheels to the default PyPI index so that `pip install torch` "just works" on GB200-class machines, the same way it does on x86. Those wheels would dynamically link to libraries like NCCL and cuBLAS — the same approach already used on x86 — so they don't balloon in size. Such large binary sizes are both hard to download for users and expensive to host by the PyPi project maintainers. Hence it is limited and heavily discouraged by the PyPi maintainer.

NVIDIA 엔지니어링 팀은 CUDA SBSA 휠을 PyPI에 게시할 것을 요청했고, 거기에 (동적으로) 링크되는 작은 휠 방식을 주도해 나갔습니다.
> The Nvidia engineering team requested that the CUDA SBSA wheels be published to PyPI, and then drove the small wheel approach that links against them.

이런 종류의 프로젝트 간(cross-project) 인프라 수준 이슈야말로 PyTorch Foundation이 잘 조율할 수 있는 영역입니다. vLLM과 PyTorch 모두 Foundation 산하 프로젝트이고, 각 프로젝트가 따로따로 우회책을 쌓아 가는 대신, 생태계 차원의 마찰을 공유된 자리에서 같이 꺼내놓고 이야기할 수 있다는 점이 실제로 큰 차이를 만들었습니다.
> This is exactly the kind of cross-project, infrastructure-level issue that the PyTorch Foundation is well-positioned to coordinate. vLLM and PyTorch are both Foundation projects, and having a shared forum to surface ecosystem friction — rather than each project working around it independently — turned out to make a real difference.


## 마침내 수정된 문제 / The fix has landed

2026년 4월의 또 다른 TAC 회의에서, 저는 이 문제가 해결되었다는 소식을 들었습니다. **PyTorch 2.11.0**부터는 aarch64 Linux에서도 `pip install torch`의 기본 동작이 CPU 전용 휠이 아닌 CUDA가 활성화된 휠을 받도록 바뀐 것입니다. NVIDIA의 Piotr Bialecki가 2.11.0 릴리스에 해당 변경이 반영되었음을 확인해 주었습니다.
> In April 2026, in another TAC meeting, I learned the issue is resolved: starting with **PyTorch 2.11.0**, the default `pip install torch` on aarch64 Linux now pulls a CUDA-enabled wheel rather than the CPU-only one. Piotr Bialecki from NVIDIA confirmed the change is live in the 2.11.0 release.

GB200에서 직접 확인해 봤는데, 결과는 우리가 바라던 바로 그 모습 — 가장 좋은 의미에서 "지루한 결과" — 그대로였습니다.
> I verified it on a GB200, and the difference is exactly what you'd want — boring, in the best possible way:

```bash
$ uv run --no-project --python 3.12 --with 'torch==2.11.0' -- python -c "import torch; print(torch.cuda.is_available())"
True

$ uv run --no-project --python 3.12 --with 'torch==2.10.0' -- python -c "import torch; print(torch.cuda.is_available())"
False
```

버전 하나 올렸을 뿐인데, 그동안 쌓여 있던 우회책 더미가 통째로 사라집니다. 요구사항 파일을 타고 번지던 커스텀 인덱스 URL도, 멀쩡히 설치된 GPU 빌드를 조용히 갈아치우던 CPU 휠 대체도, "왜 내 GB200이 GPU를 못 찾지?"라며 새로 합류한 사용자가 시간을 쏟아붓는 디버깅 세션도 더 이상 없습니다.
> One version bump, and the entire workaround stack disappears. No more custom index URLs propagating through requirements files. No more silent CPU-wheel substitutions clobbering a working install. No more "why is my GB200 not finding the GPU" debugging sessions for new users.

vLLM 입장에서 보면, GB200 / GB300에서의 설치가 비로소 매끄러워졌다는 뜻입니다. Grace Blackwell 시스템을 들고 새로 들어오는 사용자가 표준 설치 안내를 그대로 따라하기만 해도 첫 시도에 동작합니다. 갓 등장한 플랫폼 위에서 추론을 띄워보려는 단계에서 이런 매끄러움은 생각보다 큰 차이를 만듭니다.
> For vLLM specifically, this means installation on GB200 / GB300 is now genuinely smooth. New users showing up with a Grace Blackwell system can follow the standard install instructions and have things work the first time — which, when you're trying to get inference up and running on a brand-new platform, matters a lot.

물론 vLLM의 우회책들 — `use_existing_torch.py`와 `[tool.uv] no-build-isolation-package = ["torch"]` 설정 — 은 그대로 유지됩니다. 커스텀 PyTorch 빌드(나이틀리, 패치를 얹은 포크, 혹은 소스 빌드된 PyTorch와 짝을 이루는 vLLM 소스 빌드)를 쓰면서 vLLM 설치 과정이 그 `torch`에 손대지 않도록 만들고 싶은 고급 사용자들에게는 여전히 유용하기 때문입니다. 바뀌는 것은 *기본* 경로입니다. aarch64를 쓰는 평범한 사용자라면 위와 같은 사정을 더 이상 알 필요가 없어집니다. 그냥 `pip install`만 하면 끝이고, 우회책은 모두에게 부과되던 세금이 아니라 조용히 자리잡은 고급 사용자용 도구로 남게 됩니다.
> The workarounds in vLLM — both `use_existing_torch.py` and the `[tool.uv] no-build-isolation-package = ["torch"]` setting — will stay. They are still useful for advanced users who run a custom PyTorch build (a nightly, a patched fork, or a from-source build paired with a vLLM source build) and need vLLM's install to leave that `torch` strictly alone. What changes is the *default* path: ordinary users on aarch64 no longer have to know any of this exists. They can `pip install` and get on with their work, and the workarounds quietly become an advanced-user tool rather than a tax on everyone.


## 왜 이 이야기를 굳이 글로 쓰는가 / Why this is worth writing about

큰 그림에서 보면 자그마한 변화입니다. 새 기능이 아니라 패키징을 한 번 손본 정도이니까요. 그래도 한 번쯤 짚고 넘어갈 만한 이유가 두 가지 있다고 생각합니다.
> It's a small change in the grand scheme of things — a packaging tweak, not a new feature. But I think it's worth taking a moment to appreciate, for a couple of reasons.

첫째, 이 사례는 vLLM과 PyTorch가 PyTorch Foundation이라는 우산 아래에서 생산적으로 협업한 구체적인 예입니다. TAC는 단순한 거버넌스 의례가 아니라, 다운스트림 프로젝트의 고충이 실제로 그것을 해결할 수 있는 사람들 앞에 도달하는 자리이자, 프로젝트 간 조율이 우연이 아닌 기본 동작으로 이루어지는 자리입니다. 이번 이슈는 해커톤에서 터미널을 두들겨대며 욕설을 삼키던 개발자에서 출발해, TAC 토론을 거치고, GitHub 이슈로 추적되고, 마침내 릴리스에 반영되기까지의 전 과정을 완주했습니다. 그 길을 짧게 만들어 준 것이 바로 Foundation이었습니다.
> First, it's a concrete example of vLLM and PyTorch collaborating productively under the PyTorch Foundation umbrella. The TAC isn't just a governance ritual; it's a venue where pain points from downstream projects can land in front of the people who can actually fix them, and where coordination across projects happens by default rather than by accident. This issue traveled the full path — from a developer cursing at a terminal during a hackathon, to a TAC discussion, to a tracked GitHub issue, to a release — and the Foundation is what made that path short.

둘째, 개발자 경험은 누적됩니다. 누군가가 `--index-url` 플래그와 씨름하지 않아도 되는 매 시간은, 그 사람이 vLLM과 PyTorch 위에 무언가를 실제로 만들어 보는 시간으로 바뀝니다. aarch64 GPU 시스템은 앞으로 더 흔해지기만 할 텐데, 그렇다면 이런 문제는 사용자 한 명 한 명이 직접 마주치고 우회하도록 내버려두는 것보다, 지금 단계에서 — 지루하지만 중요한 인프라 레이어에서 — 한 번에 해결해두는 편이 훨씬 낫습니다.
> Second, developer experience compounds. Every hour someone doesn't spend wrestling with `--index-url` flags is an hour they spend actually building things on top of vLLM and PyTorch. aarch64 GPU systems are only going to get more common, and it's much better to fix this now, in the boring infrastructure layer, than to leave each user to discover and work around it on their own.

uv 쪽 우회책(build isolation passthrough)은 더 큰 흐름인 [WheelNext](https://wheelnext.dev/proposals/pepxxx_build_isolation_passthrough/)의 일부이기도 합니다. AI 시대에 가속기에 묶이는 의존성을 Python 패키징이 어떻게 다뤄야 할지 다시 생각해보자는, 매우 환영할 만한 움직임이죠.
> The uv-side workaround (build isolation passthrough) is part of the broader [WheelNext effort](https://wheelnext.dev/proposals/pepxxx_build_isolation_passthrough/) — a very welcome push to rethink how Python packaging handles accelerator-bound dependencies in the AI era.

마지막으로, 이번 일을 가능하게 만든 분들에게 감사의 말씀을 전합니다:

* PyTorch 코어 팀의 Alban Desmaison, Nikita Shulga, Andrey Talman — 처음 제기된 요청을 받아 직접 들고 굴려 주신 분들
* NVIDIA PyTorch 팀 — aarch64 빌드 작업을 주도하고, 2.11.0에 수정이 들어갔음을 확인해 주었으며, 이 과정에서 Piotr Bialecki가 NVIDIA와 업스트림을 오가는 든든한 연락 창구 역할을 해주었습니다.
* 휠을 빌드하고 게시한 PyTorch 릴리스 엔지니어링 팀
* PyTorch, NVIDIA, Arm 전반에 걸쳐 툴체인과 CI 인프라, 패키징을 떠받쳐 준 수많은 엔지니어 — 이번 결과는 여러분의 작업 위에 서 있습니다.
* 더불어 TAC에서 이런 대화의 자리를 계속 열어 두어주신 모든 분들께도 감사드립니다.

> A big shoutout to the people who made this happen:
> Alban Desmaison, Nikita Shulga, and Andrey Talman from the PyTorch core team, who picked up the original ask and helped move it through; The NVIDIA PyTorch team, who drove the aarch64 build work and confirmed the fix had landed in 2.11.0 with Piotr Bialecki supporting the effort and acting as the steady point of contact across NVIDIA and upstream on these issues; the PyTorch release engineering team for getting the wheels built and published; and the many engineers behind the scenes — across PyTorch, NVIDIA, and Arm — whose work on toolchains, CI infrastructure, and packaging made this possible. Thanks also to everyone in the TAC for keeping the door open for these kinds of conversations.

앞으로 나아갑시다.
> Onwards.
