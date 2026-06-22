---
layout: blog_detail
title: "ExecuTorch로 음성 에이전트 구축하기: 온디바이스 오디오를 위한 크로스 플랫폼 기반"
author: Mergen Nachin, Manuel Candales, Mengwei Liu, Jacob Szwejbka, Young Han, Songhao Jia, Stephen Jia, Scott Roy, Alban Desmaison, Hansong Zhang from PyTorch Team at Meta; Yagil Burowski, Matt Clayton, Will Burford from LM Studio
category: ["pytorch.org", "translation"]
org_title: "Building Voice Agents with ExecuTorch: A Cross-Platform Foundation for On-Device Audio"
org_link: https://pytorch.org/blog/building-voice-agents-with-executorch-a-cross-platform-foundation-for-on-device-audio/
---

### 요약 / TL;DR

- 오픈소스 음성 모델은 빠르게 늘어나고 있지만, 다양한 디바이스와 하드웨어에서 음성 에이전트 워크로드(전사, 실시간 스트리밍, 화자 분리, 음성 활동 감지, 실시간 번역)를 처리할 수 있는 통합 네이티브 추론 플랫폼은 아직 없습니다.
- ExecuTorch가 이 공백을 채워줍니다. 범용 PyTorch 네이티브 추론 플랫폼으로서, 개발자가 PyTorch에서 음성 모델을 직접 내보내고(export) Linux, macOS, Windows, Android, iOS의 CPU, GPU, NPU에서 실행할 수 있게 해줍니다.
- 4가지 작업에 걸친 5개 음성 모델의 레퍼런스 구현과 함께, 바로 빌드할 수 있는 C++ 애플리케이션 레이어 및 모바일 앱을 제공합니다. LM Studio는 이미 ExecuTorch 기반의 음성 전사 기능을 프로덕션에서 제공하고 있습니다.

> - Open source voice models are proliferating, but there's no unified native inference platform for voice agent workloads (transcription, real-time streaming, diarization, voice activity detection, live translation) across devices and hardware.
> - ExecuTorch fills this gap. As a general-purpose PyTorch-native inference platform, it enables developers to export voice models directly from PyTorch and run them across CPU, GPU, and NPU on Linux, macOS, Windows, Android, and iOS.
> - We provide reference implementations for five voice models spanning four distinct tasks, with working C++ application layers and mobile apps ready to build on. LM Studio is already shipping voice transcriptions powered by ExecuTorch in production.

### 엣지에서의 음성 기술 현황 / Voice on the Edge Today

AI 에이전트에게 듣고 말하는 능력이 점점 더 기대되고 있습니다. 스마트 글래스의 개인 비서, 휴대폰의 실시간 번역기, 노트북의 음성 기반 코딩 도우미 등 음성은 에이전트가 사용자와 상호작용하는 핵심 모달리티(modality)가 되고 있습니다. 음성 기능을 갖춘 에이전트는 단순한 오프라인 전사(transcription) 이상이 필요합니다. 스트리밍 음성 인식, 화자 분리(speaker diarization), 음성 활동 감지(voice activity detection), 노이즈 억제, 음성-텍스트 변환, 실시간 번역, 전이중(full-duplex) 지원까지 모두 낮은 지연 시간으로 로컬에서 실행되어야 합니다.
> AI agents are increasingly expected to hear and speak. Whether it's a personal assistant on smart glasses, a real-time translator on a phone, or a voice-driven coding companion on a laptop, voice is becoming a key modality for how agents interact with users. A voice-capable agent needs more than just offline transcription: it needs streaming speech recognition, speaker diarization, voice activity detection, noise suppression, speech-to-text, live translation, and full-duplex support, all running locally with low latency.

이러한 수요에 힘입어 오픈소스 음성 모델의 물결이 밀려오고 있습니다. 지난 몇 달간 Qwen3-ASR, Parakeet ASR, Voxtral Realtime, Kyutai Hibiki-Zero, Kokoro TTS, SAM-3-Audio, Liquid LFM2.5-Audio, Sortformer Diarization 등 수많은 모델이 등장했습니다. 하지만 부족한 것은 이러한 모델을 엣지 디바이스에서 네이티브로, 즉 Python 런타임이나 클라우드 의존 없이 디바이스 하드웨어에서 직접 실행되는 컴파일된 C/C++ 라이브러리로 배포하는 통합된 방법입니다.
> This demand is fueling a wave of open source voice models. In just the past few months we've seen Qwen3-ASR, Parakeet ASR, Voxtral Realtime, Kyutai Hibiki-Zero, Kokoro TTS, SAM-3-Audio, Liquid LFM2.5-Audio, Sortformer Diarization, and many more. What's missing is a uniform way to deploy them natively on edge devices, as compiled C/C++ libraries that run directly on device hardware without a Python runtime or cloud dependency.

이러한 모델 대부분은 Python에서 실행할 수 있지만, 프로덕션 수준의 엣지 배포에는 네이티브 C++ 라이브러리가 필요합니다. 기존 네이티브 솔루션은 대부분 새로운 아키텍처마다 다시 작성해야 하는 모델별 C++ 재구현이거나, 단일 하드웨어 생태계에 종속된 플랫폼별 프레임워크입니다. 음성 모델의 아키텍처와 복잡성이 다양해짐에 따라 어느 쪽 접근 방식도 확장성이 부족합니다.
> Most of these models can run in Python, but production level edge deployments require native C++ libraries. Existing native solutions tend to be either model-specific C++ rewrites that need to be rebuilt for each new architecture, or platform-specific frameworks tied to a single hardware ecosystem. As voice models diversify in architecture and complexity, neither approach scales.

ExecuTorch는 모델, 백엔드, 디바이스 전반에서 동작하는 범용 네이티브 추론 플랫폼으로 구축되었습니다. 지난해 LLM, 비전, 멀티모달 모델을 포함한 프로덕션 지원으로 [정식 출시(GA)](https://pytorch.org/blog/introducing-executorch-1-0/)에 도달했습니다. 이제 같은 플랫폼을 음성으로 확장하고 있습니다. 음성은 온디바이스 AI의 핵심 영역이며, ExecuTorch의 아키텍처가 다양한 하드웨어에서 음성 워크로드의 다양성을 처리할 수 있음을 증명하고자 했습니다. 이번 글에서는 4가지 작업에 걸친 5개 음성 모델의 레퍼런스 구현과 함께, 샘플 애플리케이션 및 바로 빌드할 수 있는 모바일 앱을 제공합니다. [LM Studio](https://lmstudio.ai/)는 이미 데스크탑 애플리케이션에서 ExecuTorch 기반의 [음성 전사](https://lmstudio.ai/transcribe) 기능을 제공하고 있습니다.
> We built ExecuTorch as a general-purpose native inference platform that works across models, backends, and devices. Last year we reached [general availability](https://pytorch.org/blog/introducing-executorch-1-0/) with production-ready support including LLMs, vision, and multimodal models. Now we're extending the same platform to voice. We see voice as a key frontier for on-device AI, and we wanted to prove that ExecuTorch's architecture could handle the diversity of voice workloads across diverse hardwares. In this post, we provide reference implementations for five voice models spanning four distinct tasks, along with sample applications and mobile apps ready to build on. [LM Studio](https://lmstudio.ai/) is already shipping [voice transcription](https://lmstudio.ai/transcribe) powered by ExecuTorch in their desktop application.

### 설계 원칙 / Design Principles

이 접근 방식은 세 가지 원칙을 기반으로 합니다:
> Three principles underpin this approach:

**전체 재작성이 아닌 최소한의 모델 수정**. 모델 작성자의 PyTorch 코드가 출발점입니다. 모델을 다른 언어로 재작성하거나 다른 형식으로 변환하는 대신, 원본 PyTorch 모델의 핵심 구성요소(오디오 인코더, 텍스트 디코더, 토큰 임베딩, 멜 스펙트로그램)에 최소한의 수정만 가하여 `torch.export()`를 직접 적용합니다. 예를 들어, Mistral이 Voxtral Realtime을, NVIDIA가 Parakeet TDT와 Sortformer를 출시했을 때, 해당 PyTorch 소스를 `torch.export()` 제약 조건을 충족하도록 필요한 부분만 수정하여 직접 내보냈습니다. 형식 변환도 없고, C++ 재구현도 없습니다.
> **Minimal model changes, not full rewrites**. The model author's PyTorch code is the starting point. Instead of rewriting models in other languages or converting them to other formats, we use torch.export() directly on the original PyTorch model's core components (audio encoder, text decoder, token embedding, mel spectrogram) with minimal edits. For example, when Mistral released Voxtral Realtime and NVIDIA published Parakeet TDT and Sortformer, we exported their PyTorch source directly with targeted edits to satisfy torch.export() constraints. No format conversion, no reimplementation in C++.

**모델을 내보내고 C++로 오케스트레이션**. 모델과 애플리케이션 로직은 서로 다른 레이어에 존재합니다. 모델 구성요소는 컴파일된 아티팩트(artifact)로 내보내지고, 가벼운 C++ 애플리케이션 레이어가 모든 것을 엮어줍니다. 스트리밍 윈도우 관리, 오디오 오버랩 처리, 스펙트로그램 정렬, 상태를 유지하는 디코딩 루프 등 복잡한 오케스트레이션을 담당합니다. ExecuTorch는 어려운 부분인 하드웨어 백엔드 간의 효율적 추론을 처리합니다.
> **Export the model, orchestrate in C++**. The model and the application logic live in different layers. Model components are exported into a compiled artifact. A thin C++ application layer ties everything together, handling the complex orchestration: streaming-window bookkeeping, audio overlap handling, spectrogram alignment, stateful decoding loops. ExecuTorch handles the hard part: efficient inference across hardware backends.

**한 번 작성하면 어떤 백엔드에서나 실행**. 한 번의 내보내기로 모든 대상 플랫폼을 지원합니다. 동일한 내보내기 모델이 XNNPACK(CPU), Metal Performance Shaders(Apple GPU), CUDA(NVIDIA GPU), 또는 Qualcomm(NPU)에서 모델이나 내보내기 스크립트에 최소한의 백엔드별 로직만으로 실행됩니다. 양자화(int4, int8)는 내보내기 전에 PyTorch에서 적용되어, 수동 커널 작업 없이 모델 크기를 크게 줄여줍니다.
> **Write once, run on any backend**. One export serves every target platform. The same exported model runs on XNNPACK (CPU), Metal Performance Shaders (Apple GPU), CUDA (NVIDIA GPU), or Qualcomm (NPU) with minimal backend-specific logic in the model or export script. Quantization (int4, int8) is applied in PyTorch before export, shrinking models significantly without manual kernel work.

### 실전에서의 음성 모델 / Voice Models in Practice

매우 다양한 아키텍처를 가진 5개의 음성 모델에서 이 접근 방식을 검증했습니다:
> We've validated this approach across five voice models with very different architectures:

**Voxtral Realtime** (스트리밍 전사, 약 40억 매개변수). Mistral의 스트리밍 전사 모델은 오프라인 수준의 정확도로 실시간 전사를 제공하며, "모델을 내보내고 C++로 오케스트레이션" 접근 방식의 좋은 예시입니다. C++ 애플리케이션 레이어는 오디오 신호 처리를 담당합니다. 과거 컨텍스트와 미리보기(lookahead)를 포함한 겹치는 오디오 윈도우, 스펙트로그램 프레임 정렬, 인코더 위치 추적 등을 처리합니다. [내보내진 모델](https://huggingface.co/mistral-labs/Voxtral-Mini-4B-Realtime-2602-ExecuTorch)은 무거운 연산을 담당합니다. 링 버퍼 KV 캐시를 갖춘 트랜스포머로 고정 메모리 내에서 무제한 길이의 스트리밍을 수행합니다. 모든 스트리밍 상수는 내보내기 시점에 결정되어 자기 서술적 메타데이터(self-describing metadata)로 내보내진 모델에 포함됩니다. int4 양자화로 모델 크기를 20GB에서 5~6GB로 줄였습니다.
> **Voxtral Realtime** (streaming transcription, ~4B params). Mistral's streaming transcription model delivers real-time transcription with offline-level accuracy, and is a good example of the "export the model, orchestrate in C++" approach. The C++ application layer handles audio signal processing: overlapping audio windows with past context and lookahead, spectrogram frame alignment, and encoder position tracking. The [exported model](https://huggingface.co/mistral-labs/Voxtral-Mini-4B-Realtime-2602-ExecuTorch) handles the heavy compute: transformers with ring-buffer KV caches that enable unlimited-duration streaming within fixed memory. All streaming constants are derived at export time and baked into the exported model as self-describing metadata. Int4 quantization shrinks the model from 20GB to 5–6GB.

**Parakeet TDT** (오프라인 전사, 6억 매개변수). NVIDIA의 고정확도 음성 인식 모델은 Token-and-Duration Transducer 아키텍처를 사용하며, 각 단계에서 어떤 토큰을 출력할지와 오디오에서 얼마나 전진할지를 동시에 예측합니다. 이 비표준 디코딩 루프는 ExecuTorch의 다중 메서드 내보내기(multi-method export)의 좋은 예시입니다. 인코더, 디코더, 조인트 네트워크가 단일 아티팩트 내에 3개의 개별 메서드로 내보내지고, C++ 애플리케이션 레이어가 LSTM 상태 관리를 포함한 TDT 전용 그리디 디코드를 구현합니다. 또한 C++로 타임스탬프 추출(단어 경계, 문장 분할)을 포함하고 있어, 완전히 독립적인 온디바이스 전사 파이프라인을 구성합니다.
> **Parakeet TDT** (offline transcription, 0.6B params). NVIDIA's high-accuracy speech recognition model uses a Token-and-Duration Transducer architecture, where the model predicts both what token to emit and how far to advance in the audio at each step. This non-standard decoding loop is a good example of ExecuTorch's multi-method export: the encoder, decoder, and joint network are exported as three separate methods in a single artifact, while the C++ application layer implements the TDT-specific greedy decode with LSTM state management. The application layer also includes timestamp extraction in C++ (word boundaries, sentence segmentation), making this a fully standalone on-device transcription pipeline.

**Sortformer** (화자 분리, 1.17억 매개변수). NVIDIA의 화자 분리 모델은 오디오 스트림에서 최대 4명의 화자에 대해 "누가 언제 말했는가"에 답합니다. 모델 자체는 상태를 갖지 않습니다(stateless). 오디오 임베딩을 입력받아 프레임별 화자 확률을 출력합니다. 모든 스트리밍 복잡성은 C++ 애플리케이션 레이어에 존재합니다. 가장 변별력 있는 프레임을 유지하는 화자 캐시, 단기 컨텍스트를 위한 슬라이딩 FIFO 윈도우, 메모리가 가득 차면 정보량이 가장 적은 프레임을 제거하는 캐시 압축 등을 포함합니다. ExecuTorch의 모델과 오케스트레이션 분리를 가장 명확하게 보여주는 사례 중 하나입니다.
> **Sortformer** (speaker diarization, 117M params). NVIDIA's diarization model answers "who spoke when" for up to four speakers in an audio stream. The model itself is stateless: it takes audio embeddings in and outputs per-frame speaker probabilities. All streaming complexity lives in the C++ application layer: a speaker cache that retains the most discriminative frames, a sliding FIFO window for short-term context, and cache compression that drops the least informative frames when memory fills up. This is one of the clearest demonstrations of ExecuTorch's separation between model and orchestration.

**Whisper** (오프라인 전사, 3,900만~15억 매개변수). OpenAI의 널리 사용되는 음성 인식 모델로, ExecuTorch에서 가장 폭넓은 백엔드를 지원합니다(CPU, Apple GPU, NVIDIA GPU, Qualcomm NPU).
> **Whisper** (offline transcription, 39M–1.5B params). OpenAI's widely adopted speech recognition model, with the widest backend coverage in ExecuTorch (CPU, Apple GPU, NVIDIA GPU, and Qualcomm NPU).

**Silero VAD** (음성 활동 감지, 2MB). 누군가 말하고 있는지를 감지하는 경량 모델입니다. 모든 음성 에이전트의 기본 구성요소이며, 기여자에게 좋은 시작점이 됩니다.
> **Silero VAD** (voice activity detection, 2MB). A lightweight model that detects whether someone is speaking. A building block for any voice agent, and a good starting point for contributors.

| 모델 / Model | 작업 / Task | 백엔드 / Backends | 플랫폼 / Platforms |
|---|---|---|---|
| [**Parakeet TDT**](https://github.com/pytorch/executorch/blob/main/examples/models/parakeet/README.md) | 전사 / Transcription | XNNPACK, CUDA, Metal Performance Shaders, Vulkan | Linux, macOS, Windows, Android |
| [**Voxtral Realtime**](https://github.com/pytorch/executorch/tree/main/examples/models/voxtral_realtime) | 스트리밍 전사 / Streaming Transcription | XNNPACK, Metal Performance Shaders, CUDA | Linux, macOS, Windows |
| [**Whisper**](https://github.com/pytorch/executorch/blob/main/examples/models/whisper/README.md) | 전사 / Transcription | XNNPACK, Metal Performance Shaders, CUDA, Qualcomm | Linux, macOS, Windows, Android |
| [**Sortformer**](https://github.com/pytorch/executorch/tree/main/examples/models/sortformer) | 화자 분리 / Speaker Diarization | XNNPACK, CUDA | Linux, macOS, Windows |
| [**Silero VAD**](https://github.com/pytorch/executorch/tree/main/examples/models/silero_vad) | 음성 활동 감지 / Voice Activity Detection | XNNPACK | Linux, macOS |

### 샘플 애플리케이션 / Sample Applications

모델 지원을 넘어, 무엇이 가능한지를 보여주는 몇 가지 엔드투엔드 애플리케이션도 구축했습니다. 이는 시작점이며, 애플리케이션 개발자들이 자신의 사용 사례에 맞게 확장하길 권장합니다:
> Beyond model enablement, we've built a few end-to-end applications to demonstrate what's possible. These are starting points, and we encourage application developers to build on them for their own use cases:

**데스크탑에서의 [실시간 전사](https://github.com/meta-pytorch/executorch-examples/tree/main/voxtral_realtime/macos)**. 이 데모는 마이크에서 실시간 오디오를 읽어 말하는 동안 전사된 텍스트를 출력하며, 모두 온디바이스로 실행됩니다. 코딩 어시스턴트, 노트 필기 도구, 접근성 기능 등 모든 데스크탑 애플리케이션의 음성 입력 기반이 됩니다. [dmg](https://github.com/meta-pytorch/executorch-examples/releases/tag/voxtral_realtime-v1.0.0) 파일을 다운로드하여 오늘 바로 사용해보세요:
> **Real-time [transcription on desktop](https://github.com/meta-pytorch/executorch-examples/tree/main/voxtral_realtime/macos).** The demo reads live audio from the microphone and outputs transcribed text as you speak, running entirely on-device. This is the foundation for voice input in any desktop application: coding assistants, note-taking tools, accessibility features. Download the [dmg](https://github.com/meta-pytorch/executorch-examples/releases/tag/voxtral_realtime-v1.0.0) file and try the app today:

![ExecuTorch와 Voxtral Realtime 기반의 독립형 실시간 음성 전사 macOS 애플리케이션 / Standalone realtime voice transcription macOS application powered by ExecuTorch and Voxtral Realtime.](/assets/blog/2026-03-15-building-voice-agents-with-executorch/VoxtralApp-no_intro.gif){:style="width:100%"}

*ExecuTorch와 Voxtral Realtime 기반의 독립형 실시간 음성 전사 macOS 애플리케이션 / Standalone realtime voice transcription macOS application powered by ExecuTorch and Voxtral Realtime.* [*영상 / Video*](https://github.com/user-attachments/assets/6d6089fc-5feb-458b-a60b-08379855976a)

**Android에서의 음성 인식**. [Parakeet](https://github.com/meta-pytorch/executorch-examples/tree/main/parakeet/android/ParakeetApp)과 [Whisper](https://github.com/meta-pytorch/executorch-examples/tree/main/whisper/android/WhisperApp) Android 앱을 사용하면 오디오를 녹음하고 온디바이스로 전사할 수 있습니다. 모델 다운로드, 마이크 녹음, 전사 기능을 갖춘 완전한 기능의 앱으로 [executorch-examples](https://github.com/meta-pytorch/executorch-examples/) 저장소에서 확인할 수 있습니다.
> **Speech recognition on Android.** The [Parakeet](https://github.com/meta-pytorch/executorch-examples/tree/main/parakeet/android/ParakeetApp) and [Whisper](https://github.com/meta-pytorch/executorch-examples/tree/main/whisper/android/WhisperApp) Android apps let users record audio and transcribe it on-device. These are fully functional apps with model download, microphone recording, and transcription, available in the [executorch-examples](https://github.com/meta-pytorch/executorch-examples/) repository.

![ExecuTorch와 Parakeet 기반의 Android 타임스탬프 포함 음성 전사 (삼성 갤럭시 S24) / Voice transcription with timestamp on Android (Samsung Galaxy S24) powered by ExecuTorch and Parakeet.](/assets/blog/2026-03-15-building-voice-agents-with-executorch/android.gif){:style="width:100%"}

*ExecuTorch와 Parakeet 기반의 Android 음성 전사(삼성 갤럭시 S24, 타임스탬프 포함) / Voice transcription with timestamp on Android (Samsung Galaxy S24) powered by ExecuTorch and Parakeet.* [*영상 / Video*](https://github.com/user-attachments/assets/9793d2d0-0d23-4627-a8dc-4334b97b07ab)

### 프로덕션 적용 사례: LM Studio / Adoption Case Study in production: LM Studio

[LM Studio](https://lmstudio.ai/)는 LLM을 로컬에서 실행하는 인기 데스크탑 애플리케이션입니다. 최근 ExecuTorch로 Parakeet TDT 모델을 실행하여 [음성 전사](https://lmstudio.ai/transcribe) 기능을 제품에 추가했습니다. LM Studio는 앱 UI에서 전사 기능을 제공하며, API 엔드포인트도 곧 출시 예정입니다. 이를 통해 LM Studio는 개발자들이 로컬 음성 인식을 자신의 워크플로우에 통합할 수 있도록 지원할 것입니다. 크로스 플랫폼 지원과 경쟁력 있는 성능을 이유로 ExecuTorch를 선택했으며, 동일한 모델과 애플리케이션 레이어에서 macOS(Metal Performance Shaders)와 Windows(CUDA)를 모두 지원합니다.
> [LM Studio](https://lmstudio.ai/) is a popular desktop application for running LLMs locally. They recently added [voice transcription](https://lmstudio.ai/transcribe) to their product, powered by ExecuTorch running the Parakeet TDT model. LM Studio exposes transcription in the app UI, with an API endpoint coming soon. With this, LM Studio will be enabling developers to integrate local speech recognition into their workflows. They chose ExecuTorch for its cross-platform support and competitive performance, shipping on macOS (Metal Performance Shaders) and Windows (CUDA) from the same model and application layer.

![LM Studio의 ExecuTorch 기반 크로스 플랫폼 온디바이스 전사 적용 / LM Studio adopts ExecuTorch for cross-platform, on-device transcription](/assets/blog/2026-03-15-building-voice-agents-with-executorch/lm-executorch.png){:style="width:100%"}

*LM Studio의 ExecuTorch 기반 크로스 플랫폼 온디바이스 전사 적용 / LM Studio adopts ExecuTorch for cross-platform, on-device transcription*

### 참여하기 / Get Involved

이 레퍼런스 구현들은 시작점이며, 지원하고자 하는 음성 모델의 범위는 훨씬 넓습니다. Qwen3-ASR, Kyutai Hibiki-Zero, Kokoro TTS, SAM-3-Audio, Liquid LFM2.5-Audio 등은 모두 PyTorch 네이티브이며 ExecuTorch 지원의 자연스러운 후보입니다. 이를 위해 커뮤니티의 도움이 필요합니다:
> These reference implementations are starting points, and the landscape of voice models we want to support is much larger. Models like Qwen3-ASR, Kyutai Hibiki-Zero, Kokoro TTS, SAM-3-Audio, and Liquid LFM2.5-Audio are all PyTorch-native and natural candidates for ExecuTorch enablement. We want the community's help to get there:

- 여러분의 프레임워크와 애플리케이션에서 음성 추론에 ExecuTorch를 도입해 보세요.
- 새로운 모델을 기여해 주세요. 음성 모델을 선택하고, 내보내고, 애플리케이션 레이어를 작성하여 PR을 열어주세요. 실시간 번역, 음성 향상(speech enhancement), 웨이크 워드 감지, 노이즈 감소, TTS(Text-to-Speech) 등 아키텍처는 이 모든 것을 지원할 준비가 되어 있습니다.
- 백엔드와 플랫폼에 기여해 주세요. 남아 있는 부분을 채우고 하드웨어 전반의 성능을 개선하는 데 도움을 주세요.

> - Adopt ExecuTorch for voice inference in your frameworks and applications.
> - Contribute new models — pick a voice model, export it, write an application layer, and open a PR. Live translation, speech enhancement, wake word detection, noise reduction, text-to-speech. The architecture is ready for all of them.
> - Contribute backends and platforms — help us close the remaining gaps and improve performance across hardware.

ExecuTorch는 음성 전용이 아닙니다. 온디바이스 LLM, 비전 모델, 멀티모달 AI를 구동하는 동일한 플랫폼입니다.
> ExecuTorch isn't just for voice. It's the same platform powering on-device LLMs, vision models, and multimodal AI.

빌드 시작하기: [ExecuTorch 문서](https://docs.pytorch.org/executorch/main/index.html) | [ExecuTorch 저장소](https://github.com/pytorch/executorch) | [ExecuTorch 예제](https://github.com/meta-pytorch/executorch-examples) | [ExecuTorch Discord](https://discord.com/invite/Dh43CKSAdc)
> Start building: [ExecuTorch Documentation](https://docs.pytorch.org/executorch/main/index.html) | [ExecuTorch repo](https://github.com/pytorch/executorch) | [ExecuTorch Examples](https://github.com/meta-pytorch/executorch-examples) | [ExecuTorch Discord](https://discord.com/invite/Dh43CKSAdc)

### 감사의 글 / Acknowledgement

이 작업은 PyTorch 팀 멤버들의 지원과 핵심 기여 없이는 불가능했습니다. Bilgin Cagatay, Tanvir Islam, Hamid Shojanazeri, Siddartha Pothapragada, Jack Khuu, Kaiming Cheng, Nikita Shulga, Angela Yi, Bin Bao, Shangdi Yu, Sherlock Huang, Yanan Cao, Digant Desai, Anthony Shoumikhin, Mark Saroufim, Chris Gottbrath, Joe Spisak, Jerry Zhang, Supriya Rao에게 감사드립니다.
> This work wouldn't have been possible without support and core contributions from PyTorch team members, including Bilgin Cagatay, Tanvir Islam, Hamid Shojanazeri, Siddartha Pothapragada, Jack Khuu, Kaiming Cheng, Nikita Shulga, Angela Yi, Bin Bao, Shangdi Yu, Sherlock Huang, Yanan Cao, Digant Desai, Anthony Shoumikhin, Mark Saroufim, Chris Gottbrath, Joe Spisak, Jerry Zhang, Supriya Rao

Voxtral Realtime 모델을 구축하고 오픈소스로 공개하며, 통합 코드를 리뷰하고 테스트해 주신 Mistral AI의 Patrick von Platen에게 감사드립니다.
> Thank you to Patrick von Platen from Mistral AI for building the Voxtral Realtime model, open sourcing it, and reviewing and testing our integration code.
