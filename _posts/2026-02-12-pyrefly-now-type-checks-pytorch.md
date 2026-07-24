---
layout: blog_detail
title: "Pyrefly, PyTorch 타입 체크 지원"
image: /assets/blog/2026-02-12-pyrefly-now-type-checks-pytorch/hero.png
author: PyTorch and Pyrefly Teams at Meta
category: ["pytorch.org", "translation"]
org_title: "Pyrefly Now Type Checks PyTorch"
org_link: https://pytorch.org/blog/pyrefly-now-type-checks-pytorch/
---

PyTorch가 이제 [핵심 저장소](https://github.com/pytorch/pytorch)뿐 아니라 Helion, TorchTitan, Ignite 등 PyTorch 생태계의 여러 프로젝트에서 타입 체크를 수행하기 위해 Pyrefly를 활용하게 되었음을 공유합니다. PyTorch처럼 규모가 큰 프로젝트에서는 동적 코드에서 놓치기 쉬운 버그를 방지하고 일관성을 유지하기 위해 타입과 타입 체크를 활용하는 것이 매우 중요한 일입니다. Pyrefly로의 전환은 이러한 개발 워크플로우에 정말 필요했던 개선을 가져다주며, 번개같이 빠른 속도, 표준을 준수하는 타입 체크, 그리고 현대적인 IDE 경험을 제공합니다. Pyrefly 덕분에 메인테이너와 컨트리뷰터는 버그를 더 일찍 발견하고, 로컬과 CI 실행 간에 일관성 있는 결과를 얻으며, 고급 타이핑 기능을 활용할 수 있습니다. 이 글에서는 Pyrefly로 넘어간 이유와 전환 이후 PyTorch가 얻은 개선점을 소개합니다.
> We're excited to share that PyTorch now leverages Pyrefly to power type checking across [our core repository](https://github.com/pytorch/pytorch), along with a number of projects in the PyTorch ecosystem: Helion, TorchTitan and Ignite. For a project the size of PyTorch, leveraging typing and type checking has long been essential for ensuring consistency and preventing common bugs that often go unnoticed in dynamic code. Migrating to Pyrefly brings a much needed upgrade to these development workflows, with lightning-fast, standards-compliant type checking and a modern IDE experience. With Pyrefly, our maintainers and contributors can catch bugs earlier, benefit from consistent results between local and CI runs, and take advantage of advanced typing features. In this blog post, we'll share why we made this transition and highlight the improvements PyTorch has already experienced since adopting Pyrefly.


## 왜 Pyrefly인가? / Why Switch to Pyrefly?

PyTorch의 향후 개발을 위해서는 빠르고, 사용하기 쉽고, 환경 간 결과가 일관성이 있으며, 잘 관리되는 타입 체커가 필요했습니다. 이러한 기준을 충족한 것이 바로 Pyrefly였습니다.
> To support the future development of PyTorch, we wanted a type checker that is fast, easy to use, consistent across developer environments, and actively maintained. These factors ultimately influenced the decision to move forward with Pyrefly.

### 속도와 정확성의 균형 / Balancing Speed with Accuracy

최근 벤치마크 결과에서 MyPy로 PyTorch를 타입 체크하는 데 50.6초가 걸렸던 반면, Pyrefly(v44.1)는 단 5.5초가 걸렸습니다. 기존 PyTorch 도구 대비 속도 향상이 매우 크면서도, 견고한 타입 안정성을 유지합니다. 우리는 빠른 결과를 제공할 뿐 아니라, 컨트리뷰터가 버그를 일찍 발견하고 타입 커버리지의 빈틈을 알아내는 데도 도움이 되는 방법을 원했습니다. Pyrefly는 타입 안정성의 품질을 떨어뜨리지 않으면서도 개발 속도를 따라갈 만큼 충분히 빨라서, 적합한 균형점을 제공하는 것으로 보입니다.
> In a recent round of benchmarking type checking Pytorch took 50.6 seconds using MyPy, whereas Pyrefly (v44.1) took only 5.5 seconds. This is a significant speed improvement over Pytorch's existing tooling while still maintaining robust type safety. We wanted an alternative that not only delivered fast results, but would also help our contributors catch bugs early and identify gaps in our type coverage. Pyrefly appears to strike the right balance for us, being fast enough to keep up with our development speed without compromising on the quality of type safety.

물론, 이는 시작일 뿐입니다. Pyrefly는 앞으로도 더 빨라질 수 있으며, 도구가 발전함에 따라 속도 향상의 이점을 훨씬 더 많이 누릴 수 있을 것으로 기대합니다. 우리는 Pyrefly의 지속적인 개발을 주시할 것이며, 향후 제공될 성능 향상을 적극적으로 통합할 예정입니다.
> That said, we see this as just the beginning; there is still room for Pyrefly to become even faster, and we expect to benefit from even greater speed gains as the tool continues to evolve. We'll be closely following Pyrefly's ongoing development and look forward to integrating future performance enhancements as they become available.

### 간소화된 설정 / Simplified Configuration

이전에 MyPy 기반으로 작업할 때에는, 컨트리뷰터가 코드베이스 전반의 커버리지와 strictness 수준을 관리하기 위해 여러 설정 파일을 다뤄야 했습니다. 이로 인해 어떤 파일이 어떤 규칙으로 체크되는지 명확히 파악하기 어려웠습니다. Pyrefly로 전환하면서 이러한 문제들이 해결되었습니다. Pyrefly 팀의 직접적인 지원을 통해 PyTorch는 이제 단일 통합 Pyrefly 설정과 필요한 억제(suppression)만을 사용하도록 전환되었으며, 이를 통해 메인테이너가 파일별 타입 체크 방식을 훨씬 더 쉽게 이해할 수 있게 되었습니다.
> Previously, our reliance on MyPy required contributors to juggle multiple configuration files to manage coverage and strictness levels across the codebase. This made it difficult to determine exactly which files were being checked and under what specific rules. Transitioning to Pyrefly has helped address these challenges. With direct support from the Pyrefly team, PyTorch has now transitioned to use a single unified Pyrefly configuration and required suppressions, making it much easier for our maintainers to understand which files are being typechecked and how.

### 개발 환경 간의 일관성 / Consistency across Development Environments

이전에는 IDE, 로컬 CLI, CI 환경에서 서로 다른 타입 체크 엔진을 사용하여 개발자들이 종종 불일치를 경험하는 경우가 있었습니다. 예를 들어 PyTorch CI 작업에서는 MyPy를 사용하지만, IDE에 따라 다른 타입 체커가 선호되는 경우가 있었습니다. 혹은 개발자들이 로컬 CLI에서 CI와 다른 strictness 수준의 MyPy 구성을 사용하기도 했습니다. 이러한 불일치는 로컬에서는 통과한 코드가 CI에서는 실패하는 등 예측 불가능한 피드백 루프와 고통스러운 개발 경험으로 이어졌습니다. 고품질의 IDE 경험과 견고한 CLI/CI 기능을 함께 제공하는 Pyrefly를 적용함으로써, PyTorch 개발자들은 이제 모든 개발 환경에서 일관된 결과를 경험할 수 있습니다.
> Previously, developers often encountered discrepancies between their IDE, local CLI, and the CI environment because different type-checking engines were being used at each stage. MyPy might be used in PyTorch CI jobs, but when it comes to IDEs, other type checkers were preferred that behaved slightly differently. Or developers would have a different MyPy strictness mode enabled for their CLI runs that differed from what was used in CI. These inconsistencies led to unpredictable feedback loops and a frustrating experience where code that passed their local type checking run would fail in CI. By adopting Pyrefly, which provides a high-quality IDE experience alongside robust CLI and CI functionality, PyTorch developers can now benefit from consistent results across all their development environments.

| | 이전 / Before | 현재 / After |
|------|-------|-------|
| CI | MyPy (full project run) | Pyrefly |
| CLI | MyPy (only on select files) | Pyrefly |
| IDE | Pyright OR other | Pyrefly |

### 활발한 유지 보수 & 빠른 개발 속도 / Active Maintenance and Rapid Development

전환의 또 다른 주요 이유는 Pyrefly가 활발히 유지되고 있으며 빠르게 발전하고 있고, 지속적인 성능 개선의 여지가 크기 때문입니다. 우리는 사용자 피드백에 대한 빠른 대응과 매주 월요일마다 새로운 마이너 버전을 배포하는 신속한 개발 주기를 높이 평가하고 있습니다. 버그가 보고된 다음 주에 바로 수정되는 경우가 대부분일 정도로, 문제 해결과 기능 제공이 매우 신속하게 이루어집니다. 최근 [Pyrefly 블로그 게시물](https://pyrefly.org/blog/)에서 소개된 예로, 성능 병목이 발견된 뒤 신속히 해결되어 PyTorch 코드베이스 전체의 IDE 반응성이 18배 향상된 사례가 있습니다.
> Another major reason for migrating is that Pyrefly is actively maintained and evolving quickly, with significant room for continual performance improvements. We've appreciated the responsiveness to user feedback and the rapid development cycles, which include new minor releases every Monday. It's not uncommon for a bug to be reported and resolved in time for the very next release, ensuring that issues are addressed and new features are delivered promptly. An example of this is described in a recent [Pyrefly blog post](https://pyrefly.org/blog/), where a performance bottleneck was identified and promptly resolved, resulting in an 18x speed up in IDE responsiveness across the PyTorch codebase.

이번 마이그레이션 과정에서, 그리고 앞으로 Pyrefly를 계속 사용하는 동안 우리의 최우선 순위는 타입 안정성과 개발자 경험의 퇴행을 방지하는 것입니다. Pyrefly 팀과의 지속적인 소통은 엣지 케이스를 신속하게 해결하고 컨트리뷰터가 원활하게 작업할 수 있도록 하는 데 매우 중요한 부분이었습니다.
> Throughout this migration, and as we continue using Pyrefly, our priority is to avoid regressions in type safety or developer experience. Maintaining a regular line of communication with the Pyrefly team has been essential for quickly addressing edge cases and enabling a smooth transition for our contributors.


## PyTorch 컨트리뷰터가 체감한 추가적인 이점 / Additional Benefits for PyTorch Contributors

PyTorch 컨트리뷰터와 메인테이너는 Pyrefly로 전환한 이후 이미 의미 있는 개선을 경험하고 있습니다. 초기에 전환을 선택했던 이유 외에도 다음과 같은 추가적인 이점이 있습니다.
> PyTorch contributors and maintainers have already experienced meaningful improvements since moving to Pyrefly. Beyond the initial motivations for the transition, other benefits include the following:

### 코드 품질 향상 / Improved Code Quality

Pyrefly 도입은 이미 PyTorch 코드베이스의 많은 버그를 발견하고 해결하는 데 기여했습니다. 이를 가능하게 한 요인 중 하나는 Pyrefly가 PyTorch 전반에서 일관된 모드로 동작하기 때문입니다. 아래 예시 코드처럼 **MyPy가 strict 모드가 아니라면 타입이 지정되지 않은 함수의 본문은 검사하지 않아** 오류가 발견되지 않을 수 있습니다. 반면 Pyrefly는 코드베이스 전체에서 하나의 일관된 모드로 동작하며, 이러한 오류를 잡아낼 수 있습니다.
> The rollout of Pyrefly has already led to the discovery and resolution of numerous bugs in the PyTorch codebase. One factor that helped achieve this was due to the fact that Pyrefly runs in a consistent mode across Pytorch. Take the code example below: **unless MyPy is in strict mode, it doesn't type check the bodies of untyped functions**, meaning errors like this would possibly go unnoticed. Pyrefly, on the other hand, runs in one consistent mode across the codebase and is able to catch these types of errors.

```python
def foo():
    return 1 + "" # pyrefly error
```

### 매끄러운 IDE 경험 / Seamless IDE Experience

Pyrefly는 [주요 IDE](https://pyrefly.org/en/docs/IDE/)와 자연스럽게 통합되어 실시간 타입 피드백, hover 문서, 즉각 진단을 제공하며, 로컬과 CI의 결과가 일치하게 합니다. 덕분에 PyTorch 컨트리뷰터는 다양한 IDE를 사용하더라도 코드를 작성하면서 타입 오류를 즉시 발견할 수 있고, 결과의 일관성에 대한 확신을 가질 수 있습니다. 이를 통해 문맥 전환에 대한 피로도가 줄고 높은 코드 품질을 유지하기 쉬워졌습니다. VSCode 사용자는 [여기](https://marketplace.visualstudio.com/items?itemName=meta.pyrefly)에서 IDE 확장을 다운로드할 수 있으며, 한 번 활성화하면 PyTorch 프로젝트의 설정 파일을 자동으로 찾습니다.
> Pyrefly integrates natively with [many major IDEs](https://pyrefly.org/en/docs/IDE/), bringing real-time type feedback, hover documentation, and instant diagnostics directly into the editor that match your local and CI results. Now PyTorch contributors using a diverse range of IDEs can spot type errors as they code and be confident their results are consistent, reducing context-switching and making it easier to maintain high code quality. VSCode users can download our IDE extension [here](https://marketplace.visualstudio.com/items?itemName=meta.pyrefly). Once enabled, it will automatically find the configuration file in the PyTorch project.

### 고급 타이핑 기능 / Advanced Typing Capabilities

Pyrefly는 복잡한 타이핑 패턴에 대한 견고한 지원과 Python 타입 규격의 엄격한 준수를 포함한 고급 타이핑 기능을 PyTorch에 제공합니다. 이는 성능과 안정적인 개발 경험을 유지하면서도, 컨트리뷰터가 더 안전하고 표현력 있는 코드를 작성할 수 있도록 합니다.
> Pyrefly brings advanced typing features to PyTorch, including robust support for complex typing patterns and strict adherence to Python typing specifications. This empowers contributors to write safer and more expressive code, while maintaining performance and a smooth developer experience.

Pyrefly의 추론 능력은 명시적 타입 주석이 없는 코드에서도 타입 오류를 감지할 수 있게 해줍니다. 즉, 레거시 코드나 실험적 모듈, 빠르게 변하는 프로토타입도 대규모 주석 작업 없이 더 높은 타입 안정성의 이점을 누릴 수 있습니다. 또한 더 명시적인 타입 주석이 필요한 코드 영역을 식별하므로 코드베이스의 타입 커버리지를 높이려는 목표를 달성하는 데 도움이 됩니다. 현재 PyTorch에서는 기본적으로 반환 타입 추론이 활성화되어 있지 않지만, 이 기능을 근시일 내에 사용할 수 있도록 주석을 추가하고 타입 문제를 해결하는 등 적극적으로 작업하고 있습니다.
> Pyrefly's inference capabilities can also enable developers to detect type errors even in code that lacks explicit type annotations. This means that legacy code, experimental modules, and fast-moving prototypes can benefit from increased type safety, without requiring a massive upfront investment in annotation. It can also help identify areas of code that could benefit from more explicit type annotations, helping us move forward with our goals of increasing type coverage in the codebase. Currently, return type inference is not enabled by default in PyTorch, but we are actively working to add annotations and fix type issues in order to un-gate this feature in the near future.

```python
def foo():
    return 1
foo() + "hello" # mypy: no error, # pyrefly: error [unsupported-operation]
```

## Pyrefly 시작하기 / Get Started with Pyrefly

[PyTorch](https://github.com/pytorch/pytorch) 컨트리뷰터는 IDE에 확장을 설치하여 Pyrefly를 사용할 수 있으며, lintrunner를 사용해 로컬 타입 체크를 빠르게 시작할 수 있습니다.
> Contributors to [PyTorch](https://github.com/pytorch/pytorch) can get started using Pyrefly by installing the extension in their editors, and can start using it for local type checking quickly and easily using lintrunner:

```sh
lintrunner init
lintrunner
```

[Helion](https://github.com/pytorch/helion) 컨트리뷰터는 IDE 확장을 설치한 다음, 저장소의 `lint.sh` 파일을 실행해 로컬 타입 체크를 수행할 수 있습니다.
> Contributors to [Helion](https://github.com/pytorch/helion) can also get started by installing the IDE extension and can do a local type check by running the repository's `lint.sh` file:

```sh
 ./lint.sh install && ./lint.sh
```

Pyrefly는 코드베이스 전체의 일관성을 보장하기 위해 CI의 lint 작업에도 통합되어 있습니다. 이를 통해 로컬 개발 중 적용된 동일한 규칙이 모든 PR에도 적용됩니다. PR을 열면 "Checks" 탭에서 lint 작업을 선택해 Pyrefly 결과를 확인할 수 있습니다.
> Pyrefly is also integrated into our CI suite under the lint job to ensure consistency across the codebase. This ensures that the same rules applied during local development are enforced on every PR. When you open a pull request, you can find the Pyrefly results by navigating to the "Checks" tab and selecting the lint job.

PyTorch 컨트리뷰터가 아니더라도 자신의 프로젝트에서 Pyrefly를 사용하고 싶다면, [여기](https://marketplace.visualstudio.com/items?itemName=meta.pyrefly)에서 VSCode 확장을 사용하거나 Pyrefly [공식 문서](https://pyrefly.org/en/docs/installation/)를 참고할 수 있습니다.
> If you're not a PyTorch contributor but still want to check out Pyrefly on your own project, you can get the VSCode extension [here](https://marketplace.visualstudio.com/items?itemName=meta.pyrefly) or check out the Pyrefly [documentation](https://pyrefly.org/en/docs/installation/).


## 추후 계획 / Future Work

Pyrefly로의 전환은 PyTorch 프로젝트에 실질적이고 의미 있는 진전을 가져왔습니다. 개발자들은 이미 더 빠르고 일관된 타입 체크의 이점을 누리고 있으며, 초기 도입 과정에서 상당한 수의 버그가 발견되고 해결되었습니다. 이 전환은 워크플로우를 간소화했으며, 코드 품질과 개발자 경험의 지속적인 향상을 위한 기반을 마련했습니다.
> Switching to Pyrefly marks a practical and meaningful advancement for the PyTorch project. Developers are already seeing the benefits of faster and more consistent type checking, and the initial rollout has helped uncover and resolve a substantial number of bugs. This transition has streamlined workflows and laid the foundation for ongoing improvements in both code quality and developer experience.

앞으로 Pyrefly의 개선에 따라 성능 향상이 지속되기를 기대합니다. 또한 우리는 코드베이스 전반의 타이핑을 개선하기 위해 Pyrefly 팀과 협력하게 된 점을 매우 기대하고 있습니다. 가장 널리 사용되는 AI/ML 라이브러리 중 하나에서 타입 주석을 강화하는 것은 메인테이너와 더 넓은 커뮤니티가 프로덕션 환경에서 PyTorch를 더욱 잘 활용할 수 있도록 합니다. Pyrefly를 통한 새롭고 빠른 타입 체커의 도입은 그 여정의 첫 단계일 뿐입니다.
> Looking ahead, we hope to continue seeing performance improvements from Pyrefly as the tool matures. We're also excited to partner with the Pyrefly team to further improve typing across the codebase. Strengthening type annotations in one of the most widely used AI/ML libraries will enable maintainers and the broader community to more confidently leverage PyTorch in production environments. Deploying a newer, faster type checker with Pyrefly is only the first step of that journey.

항상 그렇듯, 커뮤니티의 피드백은 매우 소중합니다. 타입 체크 워크플로우를 계속 개선해 나감에 따라, PyTorch 컨트리뷰터와 사용자 여러분이 사용 경험과 문제를 공유하며 개선점을 제안해주시기를 바랍니다. Pyrefly 팀에 질문이나 피드백이 있다면, [Discord](https://discord.com/invite/Cf7mFQtW7W)에서 이야기하거나 [GitHub 이슈를 만들어](https://github.com/facebook/pyrefly/issues) 버그를 알릴 수 있습니다.
> As always, community feedback is invaluable. We encourage PyTorch contributors and users to share their experiences, report issues, and suggest improvements as we continue refining the type checking workflow. If you have questions or wish to provide feedback to the Pyrefly team, you can do so in [Discord](https://discord.com/invite/Cf7mFQtW7W), or submit bug reports by [opening a GitHub issue](https://github.com/facebook/pyrefly/issues) in the Pyrefly repository.

마지막으로, 이번 전환 과정에서 피드백과 테스트를 제공해준 PyTorch 및 Pyrefly 팀, 그리고 커뮤니티 여러분께 진심으로 감사드립니다.
> Finally, we want to extend our sincere thanks to both the PyTorch and Pyrefly teams, as well as the community, for their feedback and testing throughout this transition.
