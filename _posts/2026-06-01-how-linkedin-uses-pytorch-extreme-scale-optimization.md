---
layout: blog_detail
title: "LinkedIn은 PyTorch로 어떻게 극단적 규모의 최적화 문제를 푸는가"
author: Aida Rahmattalabi, Sanjana Garg, Gregory Dexter, Zhipeng Wang, Ruby Tu, Yuan Gao, Yi Zhang
ext_author: Junghwan Park (박정환)
category: ["pytorch.org", "translation"]
date: 2026-06-01 12:00:00
org_title: "How LinkedIn Uses PyTorch to Solve Extreme-Scale Optimization Problems"
org_link: https://pytorch.org/blog/how-linkedin-uses-pytorch-to-solve-extreme-scale-optimization-problems/
---

![PyTorch LinkedIn 사례 연구 / PyTorch LinkedIn Case Study](/assets/blog/2026-06-01-how-linkedin-uses-pytorch-extreme-scale-optimization/case-study.png){:style="width:100%"}

**요약(TL;DR)**: 이 사례 연구는 LinkedIn이 분산 선형 계획법(linear programming) 솔버인 DuaLip을 GPU로 가속한 PyTorch 버전으로 개발하여, 웹 애플리케이션과 같은 극단적 규모의 최적화 과제를 다루기 위해 어떻게 재설계했는지를 보여줍니다. CPU에 묶여 있던 기존 스택에서 벗어난 이번 전환으로 자릿수 단위의 속도 향상과 효율적인 멀티 GPU 확장을 달성하는 동시에 엔지니어링 부담도 줄였습니다.
> **TL;DR**: This case study demonstrates how LinkedIn re-architected its distributed linear programming solver, DuaLip, by developing a GPU-accelerated PyTorch version to handle extreme-scale optimization challenges like web applications. This transition from a CPU-bound stack achieved order-of-magnitude speedups and efficient multi-GPU scaling while reducing engineering overhead.

## 들어가며 / Introduction

오늘날의 인터넷 플랫폼은 단순히 예측만 하는 것이 아니라 의사결정도 합니다. LinkedIn 같은 회사에서는 이러한 의사결정이 대규모 웹 애플리케이션의 지능적인 동작을 떠받칩니다.
이런 시스템 중 상당수는 그 이면을 들여다보면, 겉보기에는 단순해 보이는 하나의 질문으로 환원됩니다.
> Modern internet platforms don't just make predictions; they also make decisions. At companies like LinkedIn, these decisions power the intelligent behavior of large-scale web applications.
> Behind the scenes, many of these systems reduce to a deceptively simple question:

*수백만(혹은 수십억) 가지 선택지가 주어졌을 때, 주어진 제약 조건 아래에서 취할 수 있는 최선의 행동 집합은 무엇인가?*
> *Given millions (or billions) of options, what is the best set of actions to take under constraints?*

바로 여기서 선형 계획법(linear programming, LP)이 제약 조건 아래에서 목적 함수를 최적화하는 근본적인 수학적 틀로 등장합니다. LinkedIn 규모에서는 이러한 LP가 **수억 명의 사용자**와 **수조 개의 결정 변수(decision variable)** 를 포함할 수 있으며, 제약 행렬은 희소(sparse)하지만 매우 구조화되어 있습니다. 전통적인 LP 솔버인 심플렉스(simplex)법과 내부점(interior-point)법은 역사적으로 최적화의 주력 도구였습니다. 하지만 이들은 행렬 분해(matrix factorization)나 기저 갱신(basis update)에 의존하는데, 극단적 규모에서는 이 연산이 메모리와 계산 양쪽 모두에서 감당하기 어려울 만큼 비싸집니다. 그 결과 현대의 웹 규모 문제를 효율적으로 처리하지 못하는 경우가 많습니다.
> This is where linear programming (LP) comes in as a foundational mathematical framework for optimizing an objective under constraints. At LinkedIn scale, these LPs can involve **hundreds of millions of users** and **trillions of decision variables**, with sparse but highly structured constraint matrices. Traditional LP solvers, such as simplex and interior-point methods, have historically been the workhorses of optimization. However, they rely on matrix factorizations or basis updates that become prohibitively expensive in both memory and computation at extreme scale. As a result, they often fail to handle modern web-scale problems efficiently.

## 비즈니스 과제 / The Business Challenge

**우리의 목표는 서로 경쟁하는 목적들 아래에서 대규모 의사결정 시스템을 최적화하는 것이었습니다.**
> **Our goal was to optimize large-scale decision systems under competing objectives.**

예를 들면 다음과 같습니다.
> Examples include:

- 잠재적인 구직자에게 일자리를 매칭하는 것
- 랭킹이나 추천 시스템에서 여러 비즈니스 지표 간의 균형을 맞추는 것
- 사용자에게 발송할 이메일 양을 최적화하는 것

> - Matching jobs to potential job seekers
> - Balancing multiple business metrics in a ranking or recommendation system.
> - Optimizing the volume of emails to be sent to users

이들은 본질적으로 **어려운 최적화** 문제로, 한 지표(예: 클릭 수)를 개선하면 다른 지표(예: 불만 신고)가 악화될 수 있습니다. 형식적으로 이러한 문제는 **선형 계획법(linear program)** 으로 표현됩니다.
> These are inherently **challenging optimization** problems, where improving one metric (e.g., clicks) may hurt another (e.g., complaints). Formally, these problems are expressed as **linear programs**:

- 목적 함수: 비즈니스 가치(예: 참여도, 매출)를 최대화
- 제약 조건: 한계(예: 예산, 공정성, 빈도)를 강제

> - Objective: maximize business value (e.g., engagement, revenue)
> - Constraints: enforce limits (e.g., budget, fairness, frequency)

**핵심 병목은 확장성입니다. 문제의 크기가 커질수록, 프로덕션에서 빠르고 반복 가능한 최적화를 지원하려면 메모리와 시간 양쪽 모두에서 효율적이면서도 안정성과 해(solution)의 품질을 유지하는 구현이 필요합니다.**
> **The key bottleneck is scalability: as the problem size grows, supporting fast, repeatable optimization in production requires implementations that are both memory- and time-efficient, while maintaining stability and solution quality.**

최근 몇 년 사이, 이러한 대규모 LP를 푸는 실용적인 대안으로 1차(first-order) 방법이 부상했습니다. 고전적인 접근과 달리 이 방법들은 변화도(gradient) 정보에만 의존하고 비싼 행렬 분해를 피하기 때문에, 핵심 연산이 행렬-벡터 곱(matrix–vector multiplication)으로 지배됩니다. 그중에서도 주-쌍대(primal-dual) 정식화가 특히 효과적임이 입증되었습니다. 이 방식은 LP를 안장점(saddle-point) 문제로 다시 표현한 뒤, 수렴할 때까지 주 변수와 쌍대 변수를 반복적으로 갱신하며, 종종 프로덕션 시스템에 충분히 정확한 해를 얻어냅니다.
이러한 흐름의 연구는 Google의 PDLP, LinkedIn의 DuaLip 같은 새로운 세대의 대규모 솔버를 낳았습니다. 그중에서도 DuaLip은 능형 정규화(ridge-regularized) 쌍대 상승법(dual ascent)과 1차 최적화에 기반한 분산 솔버입니다. 매칭 문제의 분해 가능한 구조를 활용하며, 가속화된 변화도 기반 갱신과 효율적인 사영(projection) 연산자를 함께 사용해 극단적인 문제 크기까지 확장합니다.
> In recent years, first-order methods have emerged as a practical alternative for solving such massive LPs. Unlike classical approaches, these methods rely only on gradient information and avoid expensive matrix factorizations, making their core operations dominated by matrix–vector multiplications. In particular, primal-dual formulations have proven especially effective: they recast the LP as a saddle-point problem and iteratively update primal and dual variables until convergence, often achieving sufficiently accurate solutions for production systems.
> This line of work has led to a new generation of large-scale solvers, including systems like PDLP at Google and DuaLip at LinkedIn. DuaLip, in particular, is a distributed solver based on ridge-regularized dual ascent and first-order optimization. It exploits the decomposable structure of matching problems and uses accelerated gradient-based updates along with efficient projection operators to scale to extreme problem sizes.

DuaLip은 1차 방법이 프로덕션에서 웹 규모의 LP를 처리할 수 있음을 보여주지만, 원래 구현은 Scala/Spark 스택 위에 만들어져 근본적으로 CPU에 묶여 있습니다. 이 때문에 현대적인 하드웨어 가속기를 온전히 활용하는 데 한계가 있습니다. 게다가 스키마에 묶이고 템플릿에 의존하는 인터페이스 탓에 새로운 문제 정식화로 확장하기가 어려워, 변화하는 활용 사례에 대한 반복 속도를 늦춥니다.
이러한 한계에 자극을 받아, DuaLip 솔버 스택을 **GPU 가속을 적용한 PyTorch** 로 재설계했고, 그 결과 산업 규모의 최적화를 위한 현대적이고 유연하며 확장 가능한 시스템인 DuaLip-GPU를 얻었습니다.
> While DuaLip demonstrates that first-order methods can handle web-scale LPs in production, its original implementation, built on a Scala/Spark stack, remains fundamentally CPU-bound. This limits its ability to fully leverage modern hardware accelerators. Additionally, its schema-bound, template-driven interface makes it difficult to extend to new problem formulations, slowing iteration for evolving use cases.
> Motivated by these limitations, we re-architect the DuaLip solver stack in **PyTorch with GPU acceleration**, resulting in DuaLip-GPU as a modern, flexible, and scalable system for industrial-scale optimization.

## LinkedIn은 PyTorch를 어떻게 사용하는가 / How LinkedIn Uses PyTorch

이러한 과제를 해결하기 위해, 딥러닝만이 아니라 대규모 최적화를 위한 핵심 실행 엔진으로서 **DuaLip-PyTorch** 를 제안합니다. 이 시스템은 "솔버를 호출하는" 태스크 수준의 API가 아니라, 연산자 수준의 배열/텐서 프로그래밍 모델(PyTorch의 실행 시 정의(define-by-run) 패러다임을 따르는 방식)을 중심으로 구축되었습니다.
> To address these challenges, we propose **DuaLip-PyTorch** as a core execution engine for large-scale optimization—not just deep learning. The system is built around an operator-level array/tensor programming model (in the style of PyTorch's define-by-run paradigm), rather than a task-level "call a solver" API.

구체적으로, 핫 패스(hot path)는 희소 행렬-벡터 연산과 블록 단위 사영(blockwise projection)에 대한 명시적인 데이터플로(dataflow)로 표현되며, 가벼운 최대화기(maximizer)가 이를 조율합니다. 이러한 설계 경계는 의도적인 것입니다. 실행 시간을 지배하는 커널을 드러내고, 희소 레이아웃과 사영 연산자를 유연하게 선택할 수 있게 하며, GPU 실행에 자연스럽게 대응됩니다. 이 모든 것을 핵심 최적화 루프를 바꾸지 않고도 이뤄냅니다.
> Concretely, the hot path is expressed as an explicit dataflow over sparse matrix–vector operations and blockwise projections, orchestrated by a lightweight maximizer. This design boundary is intentional: it exposes the kernels that dominate runtime, enables flexible choices of sparse layouts and projection operators, and maps naturally to GPU execution—all without requiring changes to the core optimization loop.

## PyTorch로 AI 과제 해결하기 / Solving AI Challenges with PyTorch

PyTorch는 네이티브 GPU 가속, 희소 연산과 밀집(dense) 연산 모두를 위한 유연한 텐서 추상화, 그리고 변화도 계산을 위한 효율적인 행렬-벡터 연산을 제공합니다. 이러한 기능들이 결합되어, 대규모 LP 풀이가 구조적으로는 신경망 학습과 비슷하게 보이게 하되, 최적화에 특화된 기본 연산을 갖추도록 만들어 줍니다. LinkedIn에서는 이러한 기능들이 세 가지 주요 시스템·최적화 과제를 해결하는 데 도움이 되었습니다.
> PyTorch provides native GPU acceleration, flexible tensor abstractions for both sparse and dense computation, and efficient matrix-vector operations for gradient computation. Together, these capabilities allow large-scale LP solving to look structurally similar to neural network training, but with optimization-specific primitives. At LinkedIn, these features helped address three major systems and optimization challenges.

첫째, 수십억에서 수조 개의 변수를 담은 극단적 규모의 LP를 **희소 텐서 연산(sparse tensor operation)** 과 **배치 사영 커널(batched projection kernel)** 로 구현하여 GPU에서 효율적으로 실행할 수 있게 했습니다.
> First, extreme-scale LPs containing billions to trillions of variables were implemented using **sparse tensor operations** and **batched projection kernels**, enabling efficient execution on GPUs.

둘째, 변수를 여러 GPU에 분할하는 한편, all-reduce와 broadcast 같은 집합 통신(collective communication) 패턴을 통해 쌍대 변수를 복제·동기화함으로써 분산 최적화를 달성했고, 이를 통해 장치 수에 거의 선형적으로 확장할 수 있었습니다.
> Second, distributed optimization was achieved by partitioning variables across GPUs while replicating and synchronizing dual variables through collective communication patterns such as all-reduce and broadcast, allowing near-linear scaling across devices.

셋째, 더 나은 조건화(conditioning)를 위한 행 정규화(row normalization)와 스케일링, 정규화 연속법(regularization continuation) 전략, 그리고 AGD와 FISTA 계열 변형을 포함한 확장 가능한 1차 최적화 방법을 조합하여 수렴 속도를 개선했습니다. 이러한 개선은 정확도를 유지하면서도 풀이 시간을 크게 줄여 줍니다.
> Third, convergence speed was improved through a combination of row normalization and scaling for better conditioning, regularization continuation strategies, and scalable first-order optimization methods including AGD and FISTA-style variants. These improvements significantly reduce solve time while maintaining accuracy.

![분산 변화도 계산 후 NCCL 수행 / Distributed Gradient Computation Followed by NCCL](/assets/blog/2026-06-01-how-linkedin-uses-pytorch-extreme-scale-optimization/fig1.png){:style="width:100%"}
*그림 1. DuaLip-PyTorch의 상위 수준 아키텍처 / Figure 1. High-level architecture of Dualip-Pytorch*

## PyTorch 사용의 이점 / The Benefits of Using PyTorch

PyTorch를 사용함으로써 LinkedIn은 다음을 이룰 수 있었습니다.
> Using PyTorch allowed LinkedIn to:

- CPU 기반 시스템 대비 **자릿수 단위의 속도 향상** 달성
- 단일 GPU에서 멀티 GPU 시스템으로 효율적인 확장
- **유연하고 확장 가능한 LP 정식화** 지원
- 새로운 최적화 문제에 대한 엔지니어링 부담 감소
- ML과 최적화를 하나의 통합된 스택으로 연결

> - **Achieve order-of-magnitude speedups** over CPU-based systems
> - Scale efficiently from single GPU to multi-GPU systems
> - Support **flexible, extensible LP formulations**
> - Reduce engineering overhead for new optimization problems
> - Bridge ML and optimization into a unified stack

무엇보다도, 솔버를 GPU에 효율적인 희소 선형대수 중심으로 재구성함으로써 **이전에는 불가능했던 규모에서 프로덕션 수준의 최적화** 를 가능하게 했습니다.
> Most importantly, it enabled **production-grade optimization at previously infeasible scales** by restructuring the solver around GPU-efficient sparse linear algebra.

DuaLip-PyTorch의 지배적인 계산은 반복되는 희소 행렬-벡터 곱과 사영 갱신으로 이루어지는데, 이는 고처리량 GPU 실행에 자연스럽게 대응됩니다. 이러한 연산을 PyTorch에서 배치 텐서 커널로 표현하고 동기식 집합 통신으로 여러 GPU에 분산함으로써, 시스템은 원래의 CPU 기반 구현 대비 반복(iteration)당 풀이 시간을 크게 낮췄습니다.
> The dominant computation in DuaLip-Pytorch consists of repeated sparse matrix–vector multiplications and projection updates, which map naturally to high-throughput GPU execution. By expressing these operations as batched tensor kernels in PyTorch and distributing them across multiple GPUs with synchronous collective communication, the system achieved significantly lower per-iteration solve time compared to the original CPU-based implementation.

![GPU 수에 따른 속도 향상 곡선 / Scaling plot speedup](/assets/blog/2026-06-01-how-linkedin-uses-pytorch-extreme-scale-optimization/fig2.png){:style="width:100%"}
*그림 2. 이상적인 경우(선형 직선)와 비교한 GPU 수에 따른 속도 향상 곡선. 모든 GPU는 한 노드에 위치합니다. / Figure 2. Speed up curve against the number of GPUs compared to the ideal (linear line). All GPUs are located on one node.*

![PyTorch와 Scala 비교 / PyTorch vs Scala](/assets/blog/2026-06-01-how-linkedin-uses-pytorch-extreme-scale-optimization/fig3.png){:style="width:100%"}
*그림 3. 속도와 상대 오차 측면에서 Scala-PyTorch 비교. PyTorch 솔버(8 GPU)는 반복당 벽시계 시간(wall clock time)에서 상당한 이득(75배 빠름)을 보입니다. / Figure 3. Scala-Pytorch comparison in terms of speed and relative error. Pytorch solver (8 GPUs) exhibits significant gain (75 times faster) in per-iteration wall clock time.*

## 더 알아보기 / Learn More

자세한 내용은 다음을 참고하세요.
> For more information:

- DuaLip-GPU 기술 보고서: [https://arxiv.org/abs/2603.04621](https://arxiv.org/abs/2603.04621)
- 오픈소스 구현: [https://github.com/linkedin/DuaLip](https://github.com/linkedin/DuaLip)

> - DuaLip-GPU Technical Report: [https://arxiv.org/abs/2603.04621](https://arxiv.org/abs/2603.04621)
> - Open-source implementation: [https://github.com/linkedin/DuaLip](https://github.com/linkedin/DuaLip)
