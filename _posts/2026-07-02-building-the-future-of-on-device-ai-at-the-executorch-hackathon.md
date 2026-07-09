---
layout: blog_detail
title: "ExecuTorch 해커톤에서 온디바이스 AI의 미래를 만들어가다"
author: Matt White, CTO of the PyTorch Foundation, Andrew Caples, Business Development Manager at Meta, and Lauren Lunde, Product Manager at Qualcomm
category: ["pytorch.org", "translation"]
org_title: "Building the Future of On-Device AI at the ExecuTorch Hackathon"
org_link: https://pytorch.org/blog/building-the-future-of-on-device-ai-at-the-executorch-hackathon/
---

지난 주말, 샌프란시스코에서는 빌더와 연구자, 모바일 개발자, AI 실무자들이 한자리에 모여 ExecuTorch 해커톤을 열었습니다. 강력한 AI를 손안의 기기에서 로컬로 실행하면 무엇을 만들 수 있을까라는 실용적이면서도 갈수록 중요해지는 질문에 초점을 맞춘 이틀간의 현장 행사였습니다. 2026년 6월 27~28일에 열린 이번 해커톤에서 참가팀들은 ExecuTorch를 사용해 Snapdragon 기반 모바일 기기에서 직접 동작하는 실시간 AI 애플리케이션을 만들고 최적화하는 과제에 도전했습니다. 참가자들은 Snapdragon으로 구동되는 삼성 갤럭시 S25 Ultra 기기에서 개발했으며, Qualcomm과 Meta 전문가들의 워크숍과 멘토링, 실습 지원을 받았습니다.
> This past weekend in San Francisco, builders, researchers, mobile developers, and AI practitioners came together for the ExecuTorch Hackathon, a two-day, on-site event focused on a practical and increasingly important question: what can we build when powerful AI runs locally on the device in your hand? Held June 27–28, 2026, the hackathon challenged teams to build and optimize real-time AI applications that run directly on Snapdragon-powered mobile devices using ExecuTorch. Participants built on Samsung Galaxy S25 Ultra devices powered by Snapdragon, with workshops, mentorship, and hands-on support from Qualcomm and Meta experts.

이번 행사는 Qualcomm, Meta, GitHub, PyTorch Foundation 등의 후원으로 개최되었으며, Samsung은 현장 참가팀들이 사용한 Snapdragon 기반 갤럭시 S25 Ultra 기기의 하드웨어 파트너로 참여했습니다. 행사는 철저히 실습 중심으로 진행되었습니다. 참가팀들은 PyTorch와 ExecuTorch를 사용해 로컬에서 실제로 동작하는, 사용자를 마주하는 애플리케이션을 설계하고 만들고 시연하도록 요청받았으며, 지연 시간, 오프라인 동작 가능 여부, 프라이버시에 민감한 처리, 에너지 효율, 실시간 사용자 경험에 중점을 두었습니다.
> The event was hosted with support from Qualcomm, Meta, GitHub, the PyTorch Foundation, and others, with Samsung serving as a hardware partner for the Snapdragon-powered Galaxy S25 Ultra devices used by teams on-site. The format was intentionally hands-on: teams were asked to design, build, and demo real user-facing applications that run locally using PyTorch and ExecuTorch, with an emphasis on latency, offline capability, privacy-sensitive processing, energy efficiency, and real-time user experience.

행사장의 열기는 시작부터 뚜렷했습니다. 개발자들은 무언가를 만들겠다는 각오로 왔고, 그 기세는 주말 내내 이어졌습니다. 워크숍과 멘토 세션, 디버깅 논의, 실시간 테스트, 최종 데모를 거치는 동안, AI를 클라우드 전용 배포 방식을 넘어 자원이 제한된 엣지 환경으로 옮기려는 관심이 뚜렷하고 꾸준하게 드러났습니다. 가장 돋보인 프로젝트들은 단순히 모델이 로컬에서도 돌아간다는 것을 보여주는 데 그치지 않았습니다. 응답성과 프라이버시, 비용, 연결성이 사용자 경험의 핵심인 실제 제품에서 로컬 실행이 왜 중요한지를 보여주었습니다.
> The energy in the room was unmistakable from the start. Developers arrived ready to build, and that momentum lasted all weekend. Across workshops, mentor sessions, debugging conversations, live testing, and final demos, there was a clear and sustained interest in moving AI beyond cloud-only deployment patterns and into resource-constrained edge environments. The strongest projects did more than show that models could run locally; they demonstrated why local execution matters for real products, especially where responsiveness, privacy, cost, and connectivity are central to the user experience.

![해커톤 현장 사진 콜라주 / Collage of hackathon photos](/assets/blog/2026-07-02-building-the-future-of-on-device-ai-at-the-executorch-hackathon/collage_1_building-1.jpg){:style="width:100%"}

바로 이 지점에 ExecuTorch가 자리합니다. ExecuTorch는 웨어러블, 임베디드 기기, 마이크로컨트롤러를 포함한 모바일 및 엣지 기기 전반에서 온디바이스(on-device) 추론을 지원하는 엔드투엔드(end-to-end) 솔루션입니다. PyTorch Edge 생태계의 일부로서, 비전과 음성, 생성형 AI 모델을 포함한 PyTorch 모델을 엣지 기기에 효율적으로 배포할 수 있게 해줍니다. ExecuTorch의 가치는 기기 종류를 아우르는 이식성(portability)과 익숙한 PyTorch 툴체인을 통한 생산성, 그리고 CPU와 NPU, DSP 등 다양한 하드웨어 성능을 활용할 수 있는 경량 런타임을 통한 성능이라는 세 가지 축에 있습니다.
> That is exactly where ExecuTorch fits. ExecuTorch is an end-to-end solution for enabling on-device inference across mobile and edge devices, including wearables, embedded devices, and microcontrollers. It is part of the PyTorch Edge ecosystem and enables efficient deployment of PyTorch models—including vision, speech, and generative AI models—to edge devices. Its value proposition is centered on portability across device classes, productivity through familiar PyTorch toolchains, and performance through a lightweight runtime that can take advantage of CPUs, NPUs, DSPs, and other hardware capabilities.

이번 행사에는 20개가 넘는 팀에서 100명 이상의 참가자가 모여 상당한 참여를 이끌어냈습니다. 제출된 프로젝트들은 접근성 도구, 프라이버시 우선 어시스턴트, 시각적 안전 시스템, 오프라인 문서 처리, 모바일 생성형 AI 워크플로우, 의료·산업 지원 도구 등 엣지 AI가 가진 기회의 폭을 그대로 보여주었습니다.
> The event drew significant participation, with over 100 participants across 20+ teams. The submissions reflected the breadth of the edge AI opportunity: accessibility tools, privacy-first assistants, visual safety systems, offline document intelligence, mobile generative AI workflows, medical and industrial support tools, and more.

![데모 현장 사진 콜라주 / Collage of demo photos](/assets/blog/2026-07-02-building-the-future-of-on-device-ai-at-the-executorch-hackathon/collage_2_demos.jpg){:style="width:100%"}

수상팀들을 축하합니다:
> Congratulations to the winning teams:

## 1위: SafeScreen AI / 1st Place: SafeScreen AI

SafeScreen AI는 시의적절하면서도 야심 찬 애플리케이션으로 1위를 차지했습니다. 사용자가 노골적이거나 악의적인, 혹은 조작된 미디어에 완전히 노출되기 전에 이를 막아주는 로컬 온디바이스 시각적 안전 계층입니다. 이 앱은 Snapdragon 기반 Android 기기에서 직접 ExecuTorch로 동작하며, 시각 콘텐츠를 실시간으로 온디바이스에서 분석합니다. 유해할 가능성이 있는 콘텐츠가 감지되면, 시스템은 화면에서 곧바로 경고하거나 흐림 처리, 삭제, 마스킹, 차단을 할 수 있습니다.
> SafeScreen AI took first place with a timely and ambitious application: a local, on-device visual safety layer designed to help protect users from explicit, abusive, and manipulated media before they fully engage with it. The app runs directly on a Snapdragon-powered Android device using ExecuTorch and analyzes visual content on-device in real time. When potentially harmful content is detected, the system can warn, blur, redact, mask, or block that content directly on screen.

SafeScreen AI가 돋보인 이유는 엣지 AI 성능을 명확한 인간의 필요와 연결했다는 점입니다. 시각 분석을 로컬에 두어 낮은 지연 시간과 프라이버시 보호라는 두 가지 특성을 강조했는데, 이는 민감한 이미지나 영상을 먼저 원격 서버로 보내야 한다면 달성하기 어려운 것들입니다. 이 프로젝트는 또한 온디바이스 AI가 단순히 수동적인 탐지 시스템이 아니라, 능동적인 사용자 안전 계층이 될 수 있음을 보여주었습니다.
> SafeScreen AI stood out because it connected edge AI performance to a clear human need. By keeping visual analysis local, the project emphasized low latency and privacy-preserving protection—two qualities that are difficult to achieve if sensitive images or videos must be sent to a remote server first. The project also showed how on-device AI can become a proactive user safety layer, not just a passive detection system.

## 2위: SixthSense / 2nd Place: SixthSense

2위는 시각장애인을 위한 햅틱 비전(Haptic Vision)인 SixthSense에게 돌아갔습니다. 폰에 장착한 카메라와 로컬 AI 모델을 사용해 시각장애인과 저시력자가 물리적 공간을 탐색하도록 돕는 보조 웨어러블입니다. 이 시스템은 카메라가 보는 것을 해석해 장애물을 좌·중앙·우 구역으로 나누고, 허리에 착용한 벨트로 방향성 진동 신호를 보내 사용자가 장애물의 위치를 느끼고 더 트인 경로 쪽으로 움직일 수 있게 해줍니다.
> Second place went to SixthSense: Haptic Vision for the Blind, an assistive wearable that uses a phone-mounted camera and local AI models to help blind and low-vision users navigate physical spaces. The system interprets what the camera sees, separates obstacles into left, center, and right zones, and sends directional vibration signals to a belt worn at the waist so users can feel where obstacles are and move toward clearer paths.

SixthSense는 엣지 AI가 지닌 가장 매력적인 가능성 중 하나를 보여주었습니다. 바로 보조 기술을 더 접근하기 쉽고 반응성 있고 저렴하게 만드는 것입니다. 이 프로젝트는 폰에서 ExecuTorch를 통해 동작하는 온디바이스 모델로 객체 감지, 깊이 추정, 음성 상호작용, 텍스트 읽기 기능을 구현했습니다. 지속적인 연결성에 의존하지 않음으로써, 팀은 즉각적인 피드백이 중요한 혼잡하거나 시끄럽거나 네트워크가 제한된 환경에서 더 유용할 수 있는 프로토타입을 만들어냈습니다.
> SixthSense captured one of the most compelling promises of edge AI: making assistive technology more accessible, responsive, and affordable. The project used on-device models for object detection, depth estimation, spoken interaction, and text reading, with models running through ExecuTorch on the phone. By avoiding dependence on continuous connectivity, the team created a prototype that could be more useful in crowded, noisy, or network-constrained environments where immediate feedback matters.

## 3위: Toddle AI / 3rd Place: Toddle AI

Toddle AI는 부모가 유아의 보행 패턴을 기록하고 이해하도록 돕는, 프라이버시를 우선하는 Android 프로토타입으로 3위를 차지했습니다. 이 앱은 보행 영상을 녹화하거나 불러올 수 있고, 해당 영상이 사용 가능한 품질인지 평가한 뒤, 33개 랜드마크 신체 모델을 사용해 로컬에서 자세 추정(pose estimation)을 수행하고, 설명 가능한 보행 분석(gait-analysis) 파이프라인을 적용해 걸음을 식별하고 구조화된 관찰 결과를 생성합니다. 또한 원본 영상을 서버로 보내지 않고도 결과를 부모가 이해하기 쉬운 언어로 설명해주는 로컬 AI 어시스턴트도 포함하고 있습니다.
> Toddle AI earned third place with a privacy-first Android prototype designed to help parents capture and understand toddler walking patterns. The app can record or import walking videos, evaluate whether the clip is usable, run pose estimation locally using a 33-landmark body model, and apply an explainable gait-analysis pipeline to identify steps and generate structured observations. It also includes a local AI assistant that explains results in parent-friendly language without sending raw video to a server.

Toddle AI는 로컬 AI가 민감한 개인적 워크플로우를 어떻게 뒷받침할 수 있는지 보여주는 좋은 사례였습니다. 이 프로젝트는 블랙박스식 진단을 피하고, 대신 측정 가능한 관찰과 촬영 품질, 설명 가능성, 프라이버시에 초점을 맞췄습니다. 민감한 영상을 기기에 그대로 두면서도, 일상 환경에서 촬영한 데이터를 사용자가 더 잘 이해하도록 엣지 AI가 어떻게 도울 수 있는지를 보여주었습니다.
> Toddle AI was a strong example of how local AI can support sensitive personal workflows. The project avoided black-box diagnosis and instead focused on measurable observations, capture quality, explainability, and privacy. It showed how edge AI can help users better understand data captured in everyday environments while keeping sensitive video on the device.

세 수상 프로젝트 모두에서 공통된 패턴이 나타났습니다. 가장 인상적인 애플리케이션들은 단순히 "폰에서 도는 AI"가 아니었습니다. 로컬 실행이 제품 경험 자체에 필수적인 애플리케이션들이었습니다. SafeScreen AI는 즉각적이고 사적인 개입이 필요했습니다. SixthSense는 실시간 방향성 피드백이 필요했습니다. Toddle AI는 민감한 가족 영상에 대한 프라이버시 보호 분석이 필요했습니다. 바로 이런 유형의 사용 사례에서 엣지 AI는 개발자들이 만들 수 있는 것 자체를 바꿔놓습니다.
> Across all three winning projects, a common pattern emerged: the most compelling applications were not simply "AI on a phone." They were applications where local execution was essential to the product experience. SafeScreen AI needed immediate, private intervention. SixthSense needed real-time, directional feedback. Toddle AI needed privacy-preserving analysis of sensitive family video. These are precisely the kinds of use cases where edge AI can change what developers are able to build.

이번 주말은 실제 제약 조건에 맞춰 최적화하는 일의 중요성도 부각시켰습니다. 개발자들은 추상적인 환경에서 개발하고 있지 않았습니다. 실제 모바일 하드웨어와 실제 성능 고려사항, 실제 배터리·지연 시간 트레이드오프, 실제 사용자 경험 기대치를 다루고 있었습니다. 그래서 이번 행사는 더욱 값진 자리였습니다. 참가자들은 모델 아이디어에서 배포 가능한 애플리케이션으로, 데모에서 실제로 사용자 손에서 동작할 법한 시스템으로 나아가야 했습니다.
> The weekend also highlighted the importance of optimizing for real constraints. Developers were not building in an abstract environment. They were working with real mobile hardware, real performance considerations, real battery and latency tradeoffs, and real user experience expectations. That made the event especially valuable: participants had to move from model ideas to deployable applications, and from demos to systems that could plausibly work in users' hands.

PyTorch 커뮤니티에게 이는 흥미로운 신호입니다. PyTorch는 오랫동안 유연성과 사용성, 연구에서 프로덕션으로 이어지는 워크플로우로 가치를 인정받아 왔습니다. ExecuTorch는 이러한 개발자 경험을 엣지까지 확장하여, 팀들이 컴퓨팅과 메모리, 전력, 연결성이 제한된 환경으로 PyTorch 모델을 가져갈 수 있도록 돕습니다. 이번 해커톤은 개발자들이 이러한 전환을 받아들일 준비가 되어 있음을 보여주었습니다. 그들은 로컬에서 동작하고, 반응성이 있으며, 프라이버시를 지키고, 효율적이면서도 유용한 AI를 만들고 싶어합니다.
> For the PyTorch community, this is an exciting signal. PyTorch has long been valued for its flexibility, usability, and research-to-production workflow. ExecuTorch extends that developer experience to the edge, helping teams bring PyTorch models into environments where compute, memory, power, and connectivity are constrained. The hackathon showed that developers are ready for this shift. They want to build AI that is local, responsive, private, efficient, and useful.

ExecuTorch 해커톤을 가능하게 해준 모든 참가자와 멘토, 심사위원, 주최자, 파트너 여러분께 감사드립니다. 그 흥분이 주말 내내 이어진 이유는 이 기회가 진짜이기 때문입니다. 온디바이스 AI는 실용적인 개발 목표가 되어가고 있고, 커뮤니티는 이를 탐구하는 데 열정적입니다. SafeScreen AI와 SixthSense, Toddle AI를 비롯해 제출된 모든 프로젝트는 PyTorch 모델이 ExecuTorch를 통해 효율적으로 엣지 기기로 옮겨갈 때 빌더들이 무엇을 만들어낼 수 있는지를 미리 보여주었습니다.
> We are grateful to every participant, mentor, judge, organizer, and partner who made the ExecuTorch Hackathon possible. The excitement lasted all weekend because the opportunity is real: on-device AI is becoming a practical development target, and the community is eager to explore it. SafeScreen AI, SixthSense, Toddle AI, and the full set of submitted projects offered an early look at what builders can create when PyTorch models move efficiently onto edge devices through ExecuTorch.

AI의 미래는 하나의 배포 모델로 정의되지 않을 것입니다. 클라우드와 엣지, 하이브리드, 완전히 로컬인 경험을 모두 아우르게 될 것입니다. 이번 주말은 한 가지를 분명히 보여주었습니다. 엣지는 빌더들을 맞이할 준비가 되어 있고, 빌더들은 엣지로 나아갈 준비가 되어 있습니다.
> The future of AI will not be defined by one deployment model. It will include cloud, edge, hybrid, and fully local experiences. This weekend made one thing clear: the edge is ready for builders, and builders are ready for the edge.
