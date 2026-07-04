---
layout: blog_detail
title: "PyTorch의 테스트 인프라 이해하기"
author: Riya Punia, Red Hat
category: ["pytorch.org", "translation"]
org_title: "Understanding PyTorch's Test Infrastructure"
org_link: https://pytorch.org/blog/understanding-pytorchs-test-infrastructure/
---

![PyTorch의 테스트 인프라 이해하기 / Understanding PyTorch's Test Infrastructure](/assets/blog/2026-07-03-understanding-pytorchs-test-infrastructure/hero.png){:style="width:100%"}

### TL;DR

- PyTorch 테스트는 흔히 가져오기(import) 시점에 생성되므로, CI 실패 시 원본 템플릿과는 다른 디바이스(device)/dtype별 이름이 표시될 수 있습니다.
- 로컬 디버깅에는 대개 `pytest -k`와 `test/run_test.py`가 생성된 테스트 실패를 재현하는 가장 빠른 방법입니다.
- 디바이스에 무관한(device-generic) 테스트, OpInfo를 통한 연산자(operator) 메타데이터, 그리고 CI 샤딩(sharding)이 PyTorch 테스트에 기여하거나 디버깅할 때 이해해야 할 핵심 요소입니다.

> - PyTorch tests are often generated at import time, so CI failures may show device/dtype-specific names that differ from the source template.
> - For local debugging, pytest -k and test/run_test.py are usually the fastest ways to reproduce generated test failures.
> - Device-generic tests, operator metadata through OpInfos, and CI sharding are the key pieces to understand when contributing or debugging PyTorch tests.

PyTorch 테스트는 흔히 여러 디바이스와 dtype에 걸쳐 동적으로 생성되며, 이 때문에 CI의 테스트 이름이 소스 파일의 클래스·메서드 이름과 다르게 보일 수 있습니다. 이 글에서는 디바이스에 무관한 테스트와 OpInfo, `instantiate_device_type_tests()`, CI 샤딩이 어떻게 맞물려 동작하는지 설명하고, 기여자가 PyTorch 테스트를 더 효과적으로 실행하고 디버깅하는 방법을 다룹니다.
> PyTorch tests are often generated dynamically across devices and dtypes, which is why test names in CI may look different from the class and method names in the source file. This post explains how device-generic tests, OpInfos, instantiate_device_type_tests(), and CI sharding fit together, and how contributors can run and debug PyTorch tests more effectively.

### PyTorch 테스트가 다르게 느껴지는 이유 / Why PyTorch Testing Feels Different

PyTorch에 풀 리퀘스트(pull request)를 올려본 적이 있고, CI에서 `TestLinalgCUDA.test_matmul_cuda_float32`처럼 생성된 테스트가 실패하는 걸 보며 그 이름이 어디서 왔는지 궁금했던 적이 있거나 — 혹은 소스에 있는 이름으로 테스트를 실행했더니 "no tests collected"가 나온 적이 있다면, 이 글이 도움이 될 것입니다.
> If you have ever opened a pull request against PyTorch, watched a generated test like TestLinalgCUDA.test_matmul_cuda_float32 fail in CI, and wondered where that name came from – or tried running a test by its source name and got "no tests collected" – this guide is for you.

[PyTorch의 테스트 인프라](https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests)는 대규모로 동작하도록 만들어졌습니다. 사용된 데코레이터(decorator)와 [OpInfo](https://pytorch.org/blog/understanding-pytorchs-test-infrastructure/#section1)를 통해 제공되는 연산자 메타데이터에 따라, [하나의 테스트 메서드](https://github.com/pytorch/pytorch/blob/main/test/test_ops.py)가 여러 디바이스와 dtype, 연산자에 걸쳐 자동으로 확장될 수 있습니다. 이 덕분에 PyTorch는 수천 개의 조합을 수천 개의 손으로 작성한 테스트 없이도 검증할 수 있습니다. 하지만 이는 소스 파일에 작성한 테스트가 항상 CI가 실행하는 바로 그 테스트는 아니라는 뜻이기도 하며, 처음 마주치면 혼란스러울 수 있습니다.
> [PyTorch's test infrastructure](https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests) is built for scale. Depending on the decorators used and operator metadata provided through [OpInfos,](https://pytorch.org/blog/understanding-pytorchs-test-infrastructure/#section1) a [single test method](https://github.com/pytorch/pytorch/blob/main/test/test_ops.py) can expand across multiple devices, dtypes, and operators automatically. That is what lets PyTorch validate thousands of combinations without thousands of handwritten tests. But it also means the test you write in the source file is not always the exact test that CI runs, which can be confusing the first time you encounter it.

참고: 이 글에서 다루는 많은 헬퍼(helper)는 PyTorch의 내부 테스트 인프라인 `torch.testing._internal` 아래에 있습니다. 여러분 자신의 프로젝트를 테스트한다면, 대신 pytest나 [`torch.testing.assert_close`](https://docs.pytorch.org/docs/2.12/testing.html) 같은 공개 API를 사용하세요.
> Note: Many helpers discussed in this guide live under torch.testing._internal, which is PyTorch's internal test infrastructure. If you are testing your own project, use public APIs like pytest and [torch.testing.assert_close](https://docs.pytorch.org/docs/2.12/testing.html) instead.

### 이름 짓기의 미스터리: 왜 "No Tests Collected"가 뜰까? / The Naming Mystery: Why "No Tests Collected"?

PyTorch에 처음 기여하는 사람이 가장 먼저 헷갈리는 순간 중 하나는, 소스 파일에서 본 클래스·메서드 이름으로 테스트를 실행해보는 때입니다:
> One of the first confusing moments for new PyTorch contributors is trying to run a test by the class and method name they see in the source file:

```bash
pytest test/test_torch.py::TestTorch::test_matmul
```

많은 PyTorch 테스트 파일에서는 이렇게 실행하면 "no tests collected"가 나올 수 있습니다. 이는 대개 테스트가 없어서가 아니라, 소스 파일의 클래스가 테스트 러너가 보는 최종 클래스가 아니라 템플릿(template)이기 때문입니다.
> In many PyTorch test files, this may return "no tests collected." That is usually not because the test is missing. It is because the class in the source file is a template, not the final class that the test runner sees.

파일이 가져와지면(import), `instantiate_device_type_tests()`가 템플릿을 `TestTorchCPU`, `TestTorchCUDA`, `TestTorchMPS`처럼 구체적인 디바이스별 클래스로 확장합니다. 테스트가 dtype으로도 매개변수화(parameterize)되어 있다면, 생성된 메서드 이름에는 디바이스와 dtype도 포함될 수 있습니다. 예를 들어 `test_matmul_cuda_float32`처럼요. 이렇게 생성된 클래스는 원본 템플릿 클래스와 PyTorch의 디바이스별 테스트 베이스로부터 만들어지므로, PyTorch 내부 `TestCase`가 제공하는 공통 동작은 그대로 상속받습니다.
> When the file is imported, instantiate_device_type_tests() expands the template into concrete device-specific classes such as TestTorchCPU, TestTorchCUDA, or TestTorchMPS. If the test is also parameterized by dtype, the generated method name may include the device and dtype as well, for example test_matmul_cuda_float32. These generated classes are built from the original template class and PyTorch's device-specific test bases, so they still inherit the shared behavior provided by PyTorch's internal TestCase.

로컬에서 디버깅할 때는 원본 템플릿 클래스를 직접 지정하기보다, 생성된 테스트 이름 패턴으로 필터링하는 편이 대개 더 쉽습니다:
> For local debugging, it is usually easier to filter by the generated test name pattern instead of targeting the original template class directly:

```bash
pytest test/test_torch.py -k "test_matmul"
pytest test/test_torch.py -k "test_matmul_cuda_float32"
```

PyTorch가 가져오기 도중에 실행 가능한 테스트 이름을 생성한다는 것을 알고 나면, CI 실패를 소스 테스트로 훨씬 쉽게 되짚어갈 수 있습니다.
> Once you know that PyTorch generates the runnable test names during import, CI failures become much easier to map back to the source test.

### 디바이스에 무관한 테스트는 어떻게 동작할까 / How Device-Generic Tests Work

PyTorch는 CPU, CUDA, MPS, XPU 등 여러 디바이스에서 실행되며, 많은 테스트가 float16, float32, float64, bfloat16, 정수형 등 다양한 dtype에 걸쳐 동작을 검증해야 합니다. 모든 디바이스·dtype 조합마다 별도의 테스트를 작성한다면 금방 유지보수 악몽이 될 것입니다.
> PyTorch runs across devices such as CPU, CUDA, MPS, and XPU, and many tests need to validate behavior across float16, float32, float64, bfloat16, integer, and other dtypes. Writing a separate test for every device and dtype combination would quickly become a maintenance nightmare.

그래서 PyTorch는 테스트 템플릿을 사용합니다. 디바이스와 dtype을 매개변수로 받는 테스트 메서드 하나만 작성하면 됩니다:
> So PyTorch uses test templates. You write one test method with device and dtype parameters:

```python
def test_basic(self, device, dtype):
    ...
```

Python이 테스트 파일을 가져오면, `instantiate_device_type_tests()`가 선택된 디바이스 유형과 dtype 전반에 걸쳐 그 템플릿을 확장합니다. 예를 들어 템플릿 클래스 하나가 `TestMatmulCPU`, `TestMatmulCUDA`, `TestMatmulMPS` 같은 클래스와 `test_basic_cuda_float32` 같은 메서드를 생성해낼 수 있습니다.
> When Python imports the test file, instantiate_device_type_tests() expands that template across the selected device types and dtypes. For example, one template class can produce generated classes such as TestMatmulCPU, TestMatmulCUDA, and TestMatmulMPS, with generated methods such as test_basic_cuda_float32.

![테스트 클래스 계층 구조와 인스턴스화 흐름 / Test Class Hierarchy & Instantiation Flow](/assets/blog/2026-07-03-understanding-pytorchs-test-infrastructure/test-class-hierarchy-instantiation-flow.png){:style="width:100%"}
*그림 1: 테스트 클래스 계층 구조와 인스턴스화 흐름 / Figure 1: Test Class Hierarchy & Instantiation Flow*

생성된 이름은 다음 패턴을 따릅니다:
> The generated names follow this pattern:

```
<ClassName><DEVICE>.<method>_<device>_<dtype>
```

따라서 `TestMatmul.test_basic` 같은 템플릿은 `TestMatmulCUDA.test_basic_cuda_float32`가 될 수 있습니다. 디바이스는 클래스 이름에서는 대문자로, 메서드 이름에서는 소문자로 나타납니다.
> So a template like TestMatmul.test_basic may become TestMatmulCUDA.test_basic_cuda_float32. The device appears in uppercase in the class name and lowercase in the method name.

이것이 CI 실패가 여러분이 작성한 템플릿 이름만이 아니라 생성된 이름을 보여주는 이유입니다. 생성된 이름을 보면 정확히 어떤 디바이스와 dtype에서 실패했는지 알 수 있습니다.
> This is why CI failures show generated names instead of only the template name you wrote. The generated name tells you exactly which device and dtype failed.

### 한눈에 보는 아키텍처 / The Architecture at a Glance

PyTorch의 테스트 인프라는 서로 연결된 레이어(layer)의 집합으로 보면 이해하기 쉽습니다. 기여자는 보통 중간 레이어 — 디바이스 인스턴스화, 매개변수화 데코레이터, OpInfo, 테스트 유틸리티 — 와 상호작용합니다. 그 위에는 CI 오케스트레이션(orchestration)이 있고, 아래에는 공통 기반을 제공하는 베이스 유틸리티가 있습니다.
> PyTorch's test infrastructure is easier to understand as a set of connected layers. Contributors typically interact with the middle layers: device instantiation, parametrization decorators, OpInfos, and test utilities. CI orchestration sits above them, while base utilities provide the shared foundation.

![한눈에 보는 PyTorch 테스트 아키텍처 / PyTorch Testing Architecture](/assets/blog/2026-07-03-understanding-pytorchs-test-infrastructure/pytorch-testing-architecture.png){:style="width:100%"}
*그림 2: 한눈에 보는 PyTorch 테스트 아키텍처 / Figure 2: PyTorch Testing Architecture at a Glance*

#### 기여자가 자주 마주치는 핵심 파일 / Key files contributors often encounter

| 파일 / File | 역할 / What it does |
| --- | --- |
| [torch/testing/_internal/common_utils.py](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_utils.py) | `TestCase`, `run_tests()`, `load_tests`, `parametrize` 등 공통 테스트 유틸리티와 테스트 플래그·헬퍼를 포함합니다. |
| [torch/testing/_internal/common_device_type.py](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_device_type.py) | `instantiate_device_type_tests`, `@dtypes`, `@onlyCUDA`, `@onlyAccelerator`, `@ops`, `@onlyCPU` |
| [torch/testing/_internal/opinfo/core.py](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/opinfo/core.py) | 핵심 OpInfo 정의, 샘플 입력, dtype 지원 범위, 스킵(skip), 데코레이터, 허용 오차(tolerance) 메타데이터. |
| [torch/testing/_internal/common_methods_invocations.py](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_methods_invocations.py) | 일반 연산자 테스트가 사용하는 OpInfo 항목을 모아 놓은 `op_db` 레지스트리. |
| [test/run_test.py](https://github.com/pytorch/pytorch/blob/main/test/run_test.py) | CI 스타일의 러너, 샤딩(sharding), 영향받은 테스트 선택 |

### OpInfo: 메타데이터로 연산자 테스트하기 / OpInfos: Testing Operators Through Metadata

OpInfo는 PyTorch 연산자를 어떻게 테스트해야 하는지 설명하는 메타데이터 항목입니다. 모든 연산자마다 별도의 테스트를 작성하는 대신, PyTorch는 OpInfo 메타데이터를 읽어 여러 연산자에 걸쳐 동일한 검사를 실행하는 범용 테스트 템플릿을 사용합니다.
> OpInfos are metadata entries that describe how a PyTorch operator should be tested. Instead of writing a separate test for every operator, PyTorch uses generic test templates that read OpInfo metadata and run the same checks across many operators.

OpInfo는 연산자 이름, 변형(variant), 지원되는 dtype, 샘플 입력, 예상되는 스킵, 데코레이터, 허용 오차 규칙 같은 것들을 정의할 수 있습니다. 그러면 `test_ops.py` 같은 파일의 범용 테스트가 `@ops(…)`를 통해 `op_db`를 소비하며, 이때 선택된 연산자와 디바이스, dtype이 테스트에 전달됩니다.
> An OpInfo can define things like the operator name, variants, supported dtypes, sample inputs, expected skips, decorators, and tolerance rules. Generic tests in files such as test_ops.py then consume op_db through @ops(…), which passes the selected op, device, and dtype into the test.

이런 방식으로 연산자 항목 하나가 순전파(forward) 정확성, dtype·디바이스 동작, 그래디언트(gradient) 검사, 컴파일 관련 경로, Meta/FakeTensor 방식의 검증 등 다양한 종류의 커버리지에 — 테스트와 연산자 메타데이터에 따라 — 참여할 수 있습니다.
> This is how one operator entry can participate in many kinds of coverage: forward correctness, dtype and device behavior, gradient checks, compile-related paths, and Meta/FakeTensor-style validation – depending on the test and the operator metadata.

그래서 `TestCommonCUDA.test_variant_consistency_eager_torch_matmul_cuda_float32` 같은 생성된 테스트를 보게 된다면, 이는 대개 범용 OpInfo 기반 테스트가 특정 디바이스와 dtype에 대해 `torch.matmul` OpInfo를 대상으로 실행되고 있다는 뜻입니다.
> So when you see a generated test such as TestCommonCUDA.test_variant_consistency_eager_torch_matmul_cuda_float32, it usually means a generic OpInfo-based test is running against the torch.matmul OpInfo for a specific device and dtype.

`@ops(…)` 데코레이터는 PyTorch의 더 넓은 매개변수화 패턴의 한 예일 뿐입니다. 연산자가 아닌 경우에는, 테스트가 `@parametrize(…)`를 사용해 모드, 형상(shape), 레이아웃, 설정 플래그 같은 커스텀 값들에 걸쳐 변형을 생성할 수도 있습니다. PyTorch는 모듈 전용 테스트를 위한 `@modules`도 제공합니다. 이런 패턴들 전반에 걸친 아이디어는 동일합니다 — 테스트 본문 하나를 유지하고, 유용한 조합을 생성하는 일은 테스트 인프라에 맡기는 것입니다.
> The @ops(…) decorator is one example of PyTorch's broader parametrization pattern. For non-operator cases, tests can also use @parametrize(…) to generate variants over custom values such as modes, shapes, layouts, or configuration flags. PyTorch also provides @modules for module-specific tests. The idea across these patterns is the same: keep one test body and let the test infrastructure generate the useful combinations.

예를 들어, 테스트는 `@parametrize(…)`를 사용해 동일한 테스트 본문을 여러 커스텀 값에 걸쳐 실행할 수 있습니다:
> For example, a test can use @parametrize(…) to run the same test body across multiple custom values:

```python
@parametrize("reduction", ["mean", "sum"])
def test_loss(self, device, dtype, reduction):
    ...
```

### 로컬에서 테스트 실행하기 / Running Tests Locally

일상적인 디버깅에는 `pytest -k`로 시작하세요. 생성된 테스트 이름과 잘 맞물려 동작하며, 테스트가 인스턴스화된 뒤에는 더 이상 직접 찾을 수 없을 수도 있는 템플릿 클래스 이름에 의존하지 않아도 됩니다.
> For day-to-day debugging, start with pytest -k. It works well with generated test names and avoids relying on template class names that may no longer be directly discoverable after test instantiation.

```bash
# 생성된 이름 패턴에 매치되는 테스트 실행
pytest test/test_torch.py -k "test_matmul"

# 특정 디바이스/dtype 조합의 생성된 테스트 실행
pytest test/test_torch.py -k "test_matmul_cuda_float32" -x
```

CI와 비슷하게 실행하려면 `test/run_test.py`를 사용하세요. 이는 테스트 파일 실행, 영향받은 테스트 선택, 샤딩 같은 CI 관련 동작에 쓰이는 PyTorch 테스트 러너입니다.
> For CI-like runs, use test/run_test.py. It is the PyTorch test runner used for running test files, affected-test selection, and CI-related behavior such as sharding.

```bash
# PyTorch 테스트 러너로 테스트 파일 실행
python test/run_test.py test_torch

# 샤드 관련 플래그를 포함해 사용 가능한 옵션 확인
python test/run_test.py -h
```

로컬에서도 CI와 비슷한 동작을 원한다면 환경 변수도 유용합니다. 예를 들어 `PYTORCH_TESTING_DEVICE_ONLY_FOR`는 테스트를 선택한 디바이스 유형으로 좁혀주고, `PYTORCH_TEST_WITH_SLOW=1`은 `@slowTest`로 표시된 테스트를 포함시키며, `PYTORCH_TEST_WITH_DYNAMO=1`은 일반 PyTorch 테스트를 TorchDynamo 커버리지와 함께 실행합니다.
> Environment variables are also useful when you want CI-like behavior locally. For example, PYTORCH_TESTING_DEVICE_ONLY_FOR narrows tests to selected device types, PYTORCH_TEST_WITH_SLOW=1 includes tests marked with @slowTest, and PYTORCH_TEST_WITH_DYNAMO=1 runs regular PyTorch tests with TorchDynamo coverage.

### CI 실패 디버깅하기 / Debugging CI Failures

PyTorch CI 작업이 실패하면, 대개 가장 유용한 정보는 생성된 테스트 이름과 그 디바이스/dtype 접미사, 그리고 샤드입니다. PyTorch CI 작업이 실패하면, 대개 가장 유용한 정보는 생성된 테스트 이름과 그 디바이스/dtype 접미사입니다. 테스트는 원자적(atomic)이어야 하므로, 먼저 `pytest -k`로 그 특정 생성된 테스트를 로컬에서 재현하는 것부터 시작하세요. 샤드 정보는 CI 작업을 찾는 데 도움이 될 수 있지만, 재현을 위한 핵심 정보는 대개 생성된 테스트 이름입니다.
> When a PyTorch CI job fails, the most useful details are usually the generated test name and its device/dtype suffix, and the shard. When a PyTorch CI job fails, the most useful details are usually the generated test name and its device/dtype suffix. Since tests are expected to be atomic, start by reproducing the specific generated test locally with pytest -k. Shard information can help locate the CI job, but the generated test name is usually the key detail for reproduction.

#### Dr. CI와 실패 분류(triage) / Dr. CI and Failure Triage

PyTorch 풀 리퀘스트에서는 기여자가 자동화된 Dr. CI 댓글을 보게 될 수도 있습니다. Dr. CI는 실패한 작업을 요약하고, 반복되는 실패 패턴을 묶고, 기여자를 관련 로그로 안내하는 데 도움을 줍니다. 전체 CI 출력을 읽는 것을 대체하지는 않지만, 분류(triage)를 시작하기에 유용한 출발점이 되는 경우가 많습니다.
> On PyTorch pull requests, contributors may also see automated Dr. CI comments. Dr. CI helps summarize failing jobs, group recurring failure patterns, and point contributors toward relevant logs. It is not a replacement for reading the full CI output, but it is often a useful starting point for triage.

실용적인 디버깅 흐름은 다음과 같습니다: Dr. CI 요약으로 시작해서, [hud.pytorch.org](https://hud.pytorch.org/hud/pytorch/pytorch/main/1?per_page=50)에서 실패한 작업 로그를 열어보고, 생성된 테스트 이름과 샤드를 확인한 뒤, `pytest -k`나 `run_test.py`로 로컬에서 실패를 재현하는 것입니다.
> A practical debugging flow is: start with the Dr. CI summary, open the failing job logs on [hud.pytorch.org](https://hud.pytorch.org/hud/pytorch/pytorch/main/1?per_page=50), identify the generated test name and shard, then reproduce the failure locally with pytest -k or run_test.py.

![PyTorch CI 테스트 파이프라인 흐름 / PyTorch CI Testing Pipeline Flow](/assets/blog/2026-07-03-understanding-pytorchs-test-infrastructure/pytorch-ci-testing-pipeline-flow.png){:style="width:100%"}
*그림 3: PyTorch CI 테스트 파이프라인 흐름 / Figure 3: PyTorch CI Testing Pipeline Flow*

CI에서만 나타나는 흔한 실패는 대개 환경 차이, 테스트 오염(pollution), 샤딩 가정, 또는 수치 정밀도 차이에서 비롯됩니다. 테스트는 실행 순서나 다른 테스트가 남긴 전역 상태에 의존해서는 안 됩니다.
> Common CI-only failures usually come from environment differences, test pollution, sharding assumptions, or numeric precision differences. Tests should not depend on execution order or global state left behind by another test.

### 흔한 함정 / Common Pitfalls

- **템플릿 이름을 직접 지정하기:** 대신 `pytest -k` 필터나 생성된 클래스·메서드 이름을 사용하세요. 테스트 템플릿이 인스턴스화된 뒤에는 원본 템플릿 이름을 직접 찾을 수 없을 수 있습니다.
- **dtype에 무관한 테스트에서 torch.randn 사용하기:** `torch.randn`은 부동소수점과 복소수 입력에는 동작하지만 정수형과 불리언(boolean) dtype에서는 실패합니다. dtype에 무관한 테스트에는 `make_tensor`를 쓰는 편이 낫습니다 — 명시적으로 디바이스와 dtype을 요구하면서도 모든 dtype 범주를 처리하기 때문입니다.
- **디바이스를 하드코딩하기:** `device="cuda"` 같은 상수 대신 생성된 테스트가 제공하는 device 인자를 사용하세요. 이렇게 하면 테스트가 디바이스 유형 전반에서 이식 가능(portable)하게 유지됩니다.

> - **Targeting template names directly:** Use pytest -k filters or generated class and method names instead. After test templates are instantiated, the original template name may not be directly discoverable.
> - **Using torch.randn in dtype-generic tests:** torch.randn works for floating-point and complex inputs but fails on integer and boolean dtypes. Prefer make_tensor for dtype-generic tests as it handles all dtype categories while still requiring explicit device and dtype.
> - **Hardcoding devices:** Use the device argument provided by the generated test instead of constants like device="cuda". This keeps the test portable across device types.

### 빠른 참고 / Quick Reference

#### 데코레이터와 헬퍼 / Decorators and helpers

| 데코레이터 / 헬퍼 / Decorator / helper | 목적 / Purpose |
| --- | --- |
| [@dtypes(…)](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_device_type.py#L1809) | 테스트를 dtype 전반으로 확장합니다 |
| [@ops(…)](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_device_type.py#L1257) | 일반 테스트를 OpInfo 항목 전반에 걸쳐 실행합니다 |
| [@onlyCUDA](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_device_type.py#L1874) / [@onlyCPU](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_device_type.py#L1870) | 테스트를 CUDA 전용 또는 CPU 전용 실행으로 제한합니다 |
| [@onlyAccelerator](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_device_type.py#L1899) | 테스트를 가속기(accelerator) 디바이스로 제한합니다 |
| [@skipIfTorchDynamo("reason")](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_utils.py#L1917) | torch.compile 실행에서 테스트를 건너뜁니다 |
| [@toleranceOverride({…})](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_device_type.py#L1782) | dtype/백엔드별 수치 허용 오차를 정의합니다 |
| [load_tests](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_utils.py#L5444) | CI에서의 테스트 검색과 샤딩을 지원합니다 |

#### 유용한 환경 변수 / Useful environment variables

| 변수 / Variable | 목적 / Purpose |
| --- | --- |
| PYTORCH_TESTING_DEVICE_ONLY_FOR | 선택한 디바이스에 대해서만 테스트를 실행합니다 |
| PYTORCH_TEST_WITH_SLOW | 느린 테스트를 포함합니다 |
| PYTORCH_TEST_WITH_DYNAMO | torch.compile 커버리지로 테스트를 실행합니다 |
| EXPECTTEST_ACCEPT | 예상 출력 스냅샷을 갱신합니다 |

### 요약 / Summary

PyTorch의 테스트 인프라는 매우 큰 테스트 스위트(suite)를 관리 가능하게 만들도록 설계되었습니다. 디바이스에 무관한 템플릿은 `instantiate_device_type_tests()`를 통해 구체적인 테스트가 되고, OpInfo는 연산자 메타데이터를 한곳에 모으며, CI 샤딩은 테스트 실행을 워커(worker) 전반에 나눕니다.
> PyTorch's testing infrastructure is designed to make a very large test suite manageable. Device-generic templates become concrete tests through instantiate_device_type_tests(), OpInfos centralize operator metadata, and CI sharding splits test execution across workers.

핵심 아이디어는, CI에서 보는 테스트 이름이 여러분이 작성한 소스 수준의 템플릿이 아니라 흔히 생성된 이름이라는 점입니다. 그 생성된 이름 — 테스트, 디바이스, dtype, 때로는 연산자까지 — 을 읽는 법을 익히고 나면 디버깅이 훨씬 쉬워집니다.
> The key idea is that the test name you see in CI is often a generated name, not just the source-level template you wrote. Once you learn to read that generated name – the test, device, dtype, and sometimes operator – debugging becomes much easier.

### 더 읽어보기 / Further Reading

더 자세한 내용은 공식 PyTorch 테스트 문서와 소스 파일을 참고하세요:
> For more details, refer to the official PyTorch testing documentation and source files:

**PyTorch 위키: [테스트 실행 및 작성하기(Running and Writing Tests)](https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests)**
> **PyTorch Wiki: [Running and Writing Tests](https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests)**

테스트 실행, 생성된 테스트 선택, PyTorch 테스트 워크플로우 이해를 위한 기여자 가이드입니다. 이 문서에는 테스트 시스템이 사용하는 환경 변수 목록도 있습니다.
> Contributor guide for running tests, selecting generated tests, and understanding PyTorch's test workflow. This document lists the environment variables used by the test system.

**공개 테스트 API: [torch.testing 문서](https://docs.pytorch.org/docs/2.12/testing.html)**
> **Public Testing APIs: [torch.testing documentation](https://docs.pytorch.org/docs/2.12/testing.html)**

PyTorch 저장소 밖의 프로젝트를 위한 `torch.testing.assert_close` 같은 공개 테스트 API입니다.
> Public testing APIs such as torch.testing.assert_close for projects outside the PyTorch repository.

**디바이스에 무관한 테스트 인프라: [common_device_type.py](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_device_type.py)**
> **Device-Generic Test Infrastructure: [common_device_type.py](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_device_type.py)**

`instantiate_device_type_tests()`, `@dtypes`, `@ops`, 그리고 디바이스별 테스트 인스턴스화를 포함한 디바이스에 무관한 테스트 유틸리티입니다.
> Device-generic testing utilities, including instantiate_device_type_tests(), @dtypes, @ops, and device-specific test instantiation.

**연산자 정의: [opinfo/core.py](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/opinfo/core.py)**
> **OpInfo Core Definitions: [opinfo/core.py](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/opinfo/core.py)**

핵심 OpInfo 정의, 샘플 입력 메타데이터, dtype 지원 범위, 스킵, 데코레이터, 허용 오차 설정입니다.
> Core OpInfo definitions, sample input metadata, dtype support, skips, decorators, and tolerance configuration.

**연산자 레지스트리: [common_methods_invocations.py](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_methods_invocations.py)**
> **Operator Registry: [common_methods_invocations.py](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_methods_invocations.py)**

일반 연산자 테스트가 사용하는 OpInfo 항목을 모아 놓은 `op_db` 레지스트리입니다.
> The op_db registry that collects OpInfo entries used by generic operator tests.

**CI 테스트 러너: [test/run_test.py](https://github.com/pytorch/pytorch/blob/main/test/run_test.py)**
> **CI Test Runner: [test/run_test.py](https://github.com/pytorch/pytorch/blob/main/test/run_test.py)**

테스트 선택, 샤딩, 영향받은 테스트 실행을 포함한 PyTorch의 CI 스타일 테스트 러너입니다.
> PyTorch's CI-style test runner, including test selection, sharding, and affected-test execution.
