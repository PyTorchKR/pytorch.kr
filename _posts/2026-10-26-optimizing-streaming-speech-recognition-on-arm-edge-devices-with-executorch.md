---
layout: blog_detail
title: "ExecuTorch로 Arm® 엣지 기기에서 스트리밍 음성 인식 최적화하기"
author: Josiah Davis (Arm), Kshitij Sisodia (Arm)
ext_author: Junghwan Park (박정환)
image: /assets/blog/2026-10-26-optimizing-streaming-speech-recognition-on-arm-edge-devices-with-executorch/hero.png
category: ["pytorch.org", "translation"]
date: 2026-10-26 12:00:00
org_title: "Optimizing Streaming Speech Recognition on Arm® Edge Devices with ExecuTorch"
org_link: https://pytorch.org/blog/optimizing-streaming-speech-recognition-on-arm-edge-devices-with-executorch/
---

![ExecuTorch로 Arm® 엣지 기기에서 스트리밍 음성 인식 최적화하기 대표 이미지 / Optimizing Streaming Speech Recognition on Arm® Edge Devices with ExecuTorch](/assets/blog/2026-10-26-optimizing-streaming-speech-recognition-on-arm-edge-devices-with-executorch/hero.png){:style="width:100%"}

기기에 대고 말할 때는 말하는 동안 텍스트가 바로바로 나타나기를 기대합니다. 그래서 스트리밍 자동 음성 인식(automatic speech recognition, ASR)은, 모델이 처리를 시작하기 전에 전체 녹음을 모두 갖고 있는 일괄(batch) 전사와는 다른 과제입니다. 일괄 처리에서 정확도가 높다고 해서 스트리밍에서도 반드시 그런 것은 아닙니다. 한 [모델 간 ASR 비교 연구](https://arxiv.org/html/2604.14493v2)가 이 격차를 잘 보여줍니다. Qwen3-ASR-1.7B는 시험한 모델 중 일괄 정확도가 가장 높았지만, 2.4초 오디오 청크(chunk)로 평가하자 단어 오류율(word error rate, WER)이 **77%** 증가했습니다.
> When you’re speaking to a device, you expect text to appear as you talk. That makes streaming automatic speech recognition (ASR) a different challenge from batch transcription, where the model has the complete recording before it starts. Strong batch accuracy does not necessarily carry over into streaming. One [cross-model ASR study](https://arxiv.org/html/2604.14493v2) illustrates this gap. Qwen3-ASR-1.7B achieved the best batch accuracy among the models tested, but its word error rate (WER) rose by **77%** when evaluated with 2.4-second audio chunks.

이 연구의 모델 간 스트리밍 비교(표 5)에서는 NVIDIA® Nemotron™ Speech Streaming 0.6B가 560ms 오디오 청크로 동작할 때 평균 스트리밍 WER이 가장 낮았습니다. 이 비교가 짚는 핵심은, 정확도를 배포 환경에서 쓰는 청크 주기(chunk cadence)로 측정해야 한다는 점입니다.
> In that study’s cross-model streaming comparison (Table 5), NVIDIA® Nemotron™ Speech Streaming 0.6B had the lowest average streaming WER when operating over 560 ms audio chunks. The comparison makes a critical point: accuracy must be measured at the chunk cadence used in deployment.

이번 사례 연구에서는 더 새롭고 다국어를 지원하는 [Nemotron 3.5](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)를 [PyTorch](https://pytorch.org/)에서 [ExecuTorch](https://docs.pytorch.org/executorch/stable/intro-how-it-works)로 옮기고, Google Pixel 11에서 온디바이스 실행을 프로파일링합니다. 정밀도와 청크 크기가 스트리밍 품질을 어떻게 바꾸는지, 추론 스레드 하나로 오디오를 처리하는 데 시간이 얼마나 걸리는지, 워크로드가 어디에서 시간을 쓰는지, 어떤 네이티브 커널이 실행되는지 살펴봅니다. 목표는 정확도와 지연 시간을 함께 평가하는 실용적인 방법을 제시하는 것입니다.
> This case study takes the newer, multilingual [Nemotron 3.5](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) from [PyTorch](https://pytorch.org/) to [ExecuTorch](https://docs.pytorch.org/executorch/stable/intro-how-it-works) and profiles its on-device execution on a Google Pixel 11. We ask how precision and chunk size change streaming quality, how much processing time the audio requires with one inference thread, where the workload spends its time, and which native kernels run. The goal is a practical way to evaluate accuracy and latency together.

이번 글에서는 다음 내용을 배웁니다:

1. ExecuTorch가 PyTorch 모델을 최적화된 Arm CPU 커널에 연결하는 방식
2. 스트리밍 ASR에 별도의 품질 평가가 필요한 이유와, 청크가 길수록 지연 시간이 늘어나는 대신 최종 전사 결과의 정확도가 높아지는 방식
3. [Arm SME2(Scalable Matrix Extension 2)](https://learn.arm.com/learning-paths/cross-platform/multiplying-matrices-with-sme2/overview/)가 FP16, 그리고 채널별(per-channel) 가중치를 사용하는 INT8 양자화(QC8W)로 전체 ASR 처리 시간을 줄이는 방식
4. 모델 연산자(operator) 시간 측정과 네이티브 커널 샘플을 결합해 성능을 이해하고 남은 병목을 찾는 방법
5. 정확도와 지연 시간 목표를 맞추기 위해 청크 크기와 정밀도를 함께 고르는 방법

> In this blog post, you will learn:
> 1. How ExecuTorch connects PyTorch models to optimized Arm CPU kernels.
> 2. Why streaming ASR needs its own quality evaluation, and how longer chunks improve final-transcript accuracy at the cost of higher latency.
> 3. How [Arm Scalable Matrix Extension 2 (SME2)](https://learn.arm.com/learning-paths/cross-platform/multiplying-matrices-with-sme2/overview/) reduces total ASR processing time with FP16 and INT8 quantization using per-channel weights (QC8W).
> 4. How to combine model operator timings and native kernel samples to understand performance and identify remaining bottlenecks.
> 5. How to choose chunk size and precision together to meet your accuracy and latency targets.

## PyTorch 모델에서 CPU 명령어까지 / From PyTorch model to CPU instructions

ExecuTorch는 내보내기(export) 도구와 경량 런타임을 갖춘 PyTorch의 엣지 추론 프레임워크입니다. 내보내기 과정은 모델을 추론용으로 준비하고, 기기의 런타임이 불러와 실행할 수 있는 .pte 파일을 만듭니다.
> ExecuTorch is PyTorch’s edge inference framework, with export tools and a lightweight runtime. The export process prepares the model for inference and produces a .pte file that the device runtime can load and execute.

지원되는 연산은 CPU 추론 라이브러리인 [XNNPACK](https://docs.pytorch.org/executorch/1.3/backends/xnnpack/xnnpack-arch-internals.html)에 위임(delegate)됩니다. Android™에서 XNNPACK은 [Arm® KleidiAI™](https://github.com/ARM-software/kleidiai) 소프트웨어의 *마이크로커널(microkernel)* 을 선택할 수 있습니다. 마이크로커널은 내부 행렬 계산을 수행하는 작고 최적화된 루틴입니다. XNNPACK에 위임되지 않은 연산자는 ExecuTorch의 이식 가능한(portable) CPU 커널로 실행됩니다.
> Supported operations are delegated to [XNNPACK](https://docs.pytorch.org/executorch/1.3/backends/xnnpack/xnnpack-arch-internals.html), a CPU inference library. On Android™, XNNPACK can select *microkernels* from [Arm® KleidiAI™](https://github.com/ARM-software/kleidiai) software: small, optimized routines that perform the inner matrix calculations. Operators not delegated to XNNPACK run using ExecuTorch’s portable CPU kernels.

![모델 내보내기에서 CPU 명령어까지 / From model export to CPU instructions](/assets/blog/2026-10-26-optimizing-streaming-speech-recognition-on-arm-edge-devices-with-executorch/1-2.png){:style="width:100%"}

*그림 1. 모델 내보내기에서 CPU 명령어까지 / Figure 1. From model export to CPU instructions.*

SME2는 일반 행렬 곱셈(general matrix multiplication, GEMM)을 포함한 행렬 계산을 가속하도록 Arm의 명령어 집합을 확장합니다. SME2를 갖춘 [Arm C1 CPU 클러스터 구현](https://support.arm.com/documentation/107831/0102/The-C1-SME2--unit?lang=en)에서는 CPU 클러스터의 코어들이 행렬 유닛을 공유하며, KleidiAI는 지원되는 연산에서 이 유닛을 사용하는 마이크로커널을 제공합니다. 빌드에서 SME2를 활성화하면 그 경로를 쓸 수 있게 됩니다. 하지만 실제로 어떤 커널이 실행되었는지 확인하려면 네이티브 프로파일링이 필요합니다.
> SME2 extends Arm’s instruction set to accelerate matrix calculations, including general matrix multiplication (GEMM). In the [Arm C1 CPU cluster implementation](https://support.arm.com/documentation/107831/0102/The-C1-SME2--unit?lang=en) with SME2, the matrix unit is shared by cores in the CPU cluster, and KleidiAI provides microkernels that use it for supported operations. Enabling SME2 in a build makes those paths available. However, native profiling is needed to show which kernels actually ran.

## 스트리밍 호출 한 번에 담긴 것 / What one streaming call contains

Nemotron 3.5는 RNN-T 아키텍처를 바탕으로 만든 다국어 스트리밍 ASR 모델이며, 세 부분으로 구성됩니다:

1. **인코더(encoder)** 는 FastConformer 어텐션 계층과 합성곱 계층으로 오디오를 표현합니다.
2. **예측기(predictor)** 는 LSTM으로 지금까지 출력한 토큰의 이력을 표현합니다.
3. **결합기(joiner)** 는 작은 피드포워드(feedforward) 계층으로 인코더의 오디오 표현과 예측기의 토큰 이력 표현을 결합해 다음 토큰을 예측합니다. *공백(blank)* 을 출력할 수도 있는데, 이때 디코딩은 다음 오디오 위치로 넘어갑니다.

> Nemotron 3.5 is a multilingual streaming ASR model built around an RNN-T architecture with three parts:
> 1. The **encoder** uses FastConformer attention and convolution layers to represent the audio.
> 2. The **predictor** uses an LSTM to represent the history of emitted tokens.
> 3. The **joiner** uses a small feedforward layer to combine the encoder’s audio representation with the predictor’s token-history representation and predict the next token. It can also produce a *blank*, which advances decoding to the next audio position.

프런트엔드(frontend)는 오디오를 로그 멜(log-mel) 특징 벡터로 변환하며, **10ms**마다 벡터를 하나씩 만듭니다. 인코더는 시간 위치(time position)의 수를 **8**분의 1로 줄이므로, 인코더 출력은 오디오 **80ms**마다 위치 하나를 갖습니다. 이렇게 하면 같은 길이의 오디오를 더 압축된 시퀀스로 표현할 수 있습니다.
> The frontend converts audio into log-mel feature vectors, producing one vector for every **10 ms**. The encoder reduces the number of time positions by a factor of **8**, so its output has one position for every **80 ms** of audio. This creates a more compact sequence representing the same audio duration.

![파형 윈도우에서 스트리밍 인코더 위치까지 / From waveform windows to streaming encoder positions](/assets/blog/2026-10-26-optimizing-streaming-speech-recognition-on-arm-edge-devices-with-executorch/2-3.png){:style="width:100%"}

*그림 2. 파형 윈도우에서 스트리밍 인코더 위치까지 / Figure 2. From waveform windows to streaming encoder positions.*

*미리보기(lookahead)* 는 청크 안에서 쓸 수 있는 미래 문맥이고, *청크 주기(chunk cadence)* 는 호출 한 번을 위해 새 오디오를 얼마나 자주 모으는지를 뜻합니다. 미리보기 13에서는 호출 한 번이 **1,120ms**의 오디오를 모아 인코더 위치 14개를 만듭니다. 예측기와 결합기는 **유효한 위치 14개 모두**를 처리하고, 거기서 나온 토큰을 전사 결과에 누적합니다.
> *Lookahead* is the future context available within a chunk; *chunk cadence* is how often fresh audio is collected for a call. At lookahead 13, a call collects **1,120 ms** of audio and produces 14 encoder positions. The predictor and joiner processes **all 14 valid positions**, accumulating their tokens into the transcript.

미리보기 0, 3, 6은 각각 **80, 320, 560ms** 청크에 해당합니다. 청크가 길수록 첫 텍스트와 이후 업데이트를 기다리는 시간이 늘어나며, 오디오를 모은 뒤에는 전처리와 추론 시간이 더해집니다.
> Lookaheads 0, 3, and 6 give **80, 320, and 560 ms** chunks. Longer chunks increase the wait for the first text and subsequent updates; preprocessing and inference add time after audio collection.

## 청크 크기와 정밀도를 함께 고르기 / Choose chunk size and precision together

호스트에서 XNNPACK CPU 백엔드를 사용하는 ExecuTorch 런타임으로 전사 정확도를 측정했습니다. 데이터셋 준비, 전사 결과 정규화, WER 채점에는 [Hugging Face Open ASR Leaderboard 평가 도구(harness)](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)를 사용하는 로컬 스트리밍 어댑터를 만들어 썼습니다.
> We measured transcription accuracy on the host using the ExecuTorch runtime with its XNNPACK CPU backend. We created a local streaming adapter that uses the [Hugging Face Open ASR Leaderboard evaluation harness](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) for dataset preparation, transcript normalization, and WER scoring.

FP32, FP16, INT8을 네 가지 청크 크기에서 평가했습니다. INT8 사양은 QC8W로, 대상이 되는 선형 연산에 동적 INT8 활성화와 채널별 INT8 가중치를 사용하고 출력은 FP32로 냅니다.
> We evaluated FP32, FP16, and INT8 at four chunk sizes. The INT8 specification is QC8W which uses dynamic INT8 activations and per-channel INT8 weights for eligible linear operations, with FP32 outputs.

![네 개 코퍼스에 걸친 청크 크기별 스트리밍 WER / Streaming WER versus chunk size across four corpora](/assets/blog/2026-10-26-optimizing-streaming-speech-recognition-on-arm-edge-devices-with-executorch/3-3.png){:style="width:100%"}

*그림 3. 네 개 코퍼스에 걸친 청크 크기별 스트리밍 WER / Figure 3. Streaming WER versus chunk size across four corpora.*

위 그림의 각 점은 LibriSpeech Other(LS Other), AMI Cleaned, Earnings22, VoxPopuli Cleaned AA 데이터셋의 **발화(utterance) 14,019개**를 포괄합니다. WER은 네 코퍼스 전체의 단어 오류 총수를 참조 단어 총수로 나눈 값입니다.
> Each point on the above figure covers **14,019 utterances** from LibriSpeech Other (LS Other), AMI Cleaned, Earnings22, and VoxPopuli Cleaned AA datasets. WER divides total word errors by total reference words across all four corpora.

**청크가 길수록 최종 전사 결과의 정확도가 높아집니다.** 80ms에서 1,120ms로 늘리면 FP16의 WER은 **15.2%에서 11.5%로**, QC8W는 **15.5%에서 11.7%로** 떨어집니다. **FP16은 FP32와 사실상 같으며**, 모든 청크 크기에서 차이가 **0.03%p 미만**이므로 이후 성능 분석에서는 FP32를 제외합니다. **QC8W도 높은 정확도를 유지하며**, FP32 대비 WER이 **0.21~0.31%p** 늘어나는 데 그칩니다.
> **Longer chunks improve final-transcript accuracy.** From 80 to 1,120 ms, FP16 WER falls from **15.2% to 11.5%**, and QC8W from **15.5% to 11.7%**. **FP16 essentially matches FP32**, differing by less than **0.03 percentage points** at every chunk size, so we exclude FP32 from further performance analysis. **QC8W preserves strong accuracy**, adding **0.21–0.31 percentage points** of WER relative to FP32.

이 호스트 WER 결과는 정밀도와 미리보기를 고르는 근거가 됩니다. 하지만 기기에서 텍스트가 나타나기까지 얼마나 걸리는지는 측정하지 않습니다.
> These host WER results inform precision and lookahead choices. However, it does not measure how long it takes for text to appear on the device.

## 엣지 기기에서 실시간 예산 검증하기 / Test the real-time budget on an edge device

Google Pixel 11 스마트폰에서 **1,120ms 청크**와 **중간 코어(mid-core) CPU 하나에 고정한 추론 스레드 하나**로 **24.9초 길이의 녹음**을 처리했습니다. 시스템에서 지속적으로 도는 스트리밍 워크로드를 대표하는 대상으로 중간 코어 CPU를 골랐습니다. ExecuTorch 1.3.1을 SME2 지원을 켠 빌드와 끈 빌드, 두 가지로 만들어 시험했습니다. 두 빌드 모두 XNNPACK과 KleidiAI를 포함했고, 같은 모델 아티팩트를 사용했습니다.
> We processed a **24.9-second recording** on a Google Pixel 11 smartphone device using **1,120 ms chunks** and **one inference thread pinned to a mid-core CPU**. We chose the mid-core CPU as a representative target for a sustained streaming workload on the system. We tested two builds of ExecuTorch 1.3.1, toggling SME2 support on and off. Each build retained XNNPACK and KleidiAI, and used identical model artifacts.

각 구성은 워밍업 두 번과 측정용 트레이싱 실행 한 번으로 진행했습니다. 총 시간은 불러온 녹음을 오디오 전처리부터 최종 디토큰화(detokenization)까지 처리하는 시간입니다.
> Each configuration had two warm-ups and one measured tracing run. Totals cover processing the loaded recording from audio preprocessing through final detokenization.

![SME2 비활성화 및 활성화 시 전체 녹음 처리 시간 / Whole-recording processing time with SME2 disabled and enabled](/assets/blog/2026-10-26-optimizing-streaming-speech-recognition-on-arm-edge-devices-with-executorch/4-2.png){:style="width:100%"}

*그림 4. SME2 비활성화 및 활성화 시 전체 녹음 처리 시간 / Figure 4. Whole-recording processing time with SME2 disabled and enabled.*

**SME2는 처리 시간을 크게 줄입니다:** FP16은 **9.79초에서 4.61초로**(**2.12배** 빠름), QC8W는 **5.20초에서 3.24초로**(**1.61배** 빠름) 줄었습니다.
> **SME2 substantially reduces processing time:** FP16 falls from **9.79 to 4.61 seconds** (**2.12x** faster), and QC8W from **5.20 to 3.24 seconds** (**1.61x** faster).

SME2 유무와 관계없이 두 정밀도 모두 녹음을 오디오 길이보다 훨씬 짧은 시간에 처리합니다. 처리 시간을 오디오 길이로 나누면, (SME2를 켠 상태에서) 실시간 계수(real-time factor)는 **FP16이 0.19, QC8W가 0.13**입니다. 추론 스레드 하나만으로도 처리 여유(headroom)가 상당하다는 뜻입니다. 다만 오디오는 디코딩하기 전에 먼저 도착해야 하므로, 이 여유가 청크 수집 지연을 없애지는 않습니다. 하지만 청크 사이에 다른 CPU 작업을 할 여력을 남기고, 지속적인 전사 중 발열 부하를 관리하는 데 도움이 될 수 있습니다.
> With and without SME2, both process the recording in much less time than its audio duration. Dividing the processing time by the audio duration we calculate real-time factors of **0.19 for FP16** and **0.13 for QC8W** (with SME2 on). This provides substantial processing headroom with only a single inference thread. Audio still has to arrive before it can be decoded, so this headroom does not remove the chunk-collection delay. But it leaves capacity for other CPU work between chunks and can help manage thermal load during sustained transcription.

## ETDump로 병목 찾기 / Find the bottleneck with ETDump

[ETDump](https://docs.pytorch.org/executorch/1.3/devtools-overview.html)는 ExecuTorch 연산자 이벤트와 시간을 기록합니다. **FP16의 2.12배**와 **QC8W의 1.61배** 속도 향상이 어디에서 오는지 이해하기 위해, 인코더, 예측기, 결합기 전반의 머신러닝(ML) 연산자를 다음 범주로 묶었습니다:

- **GEMM(파란색):** 행렬 곱셈으로, 주로 인코더의 밀집(dense) 계층입니다.
- **데이터 이동(주황색):** 복사, 연결(concatenation), 반복(repeat), 임베딩 조회입니다.
- **요소별 연산(빨간색):** 활성화, 비교 등 값 하나하나에 하는 작업입니다.
- **합성곱과 정규화(초록색과 보라색):** 합성곱 필터와 정규화 연산자이며, 여기서는 둘 다 비교적 작습니다.
- **기타(자주색):** 텐서 생성처럼 시간이 측정된 나머지 작은 연산입니다.
- **오버헤드(회색):** 묶은 연산자 이벤트 바깥의 전체 실행 시간이며, 러너(runner)와 프레임워크 오버헤드가 포함될 수 있습니다.

> [ETDump](https://docs.pytorch.org/executorch/1.3/devtools-overview.html) records ExecuTorch operator events and timings. To understand where the **2.12X FP16** and **1.61X QC8W** speedups come from, we grouped machine learning (ML) operators across the encoder, predictor, and joiner into categories:
> - **GEMM (blue):** Matrix multiplication, mainly the encoder’s dense layers.
> - **Data movement (orange):** Copies, concatenation, repeats, and embedding lookup.
> - **Elementwise (red):** Activations, comparisons, and other per-value work.
> - **Convolution and normalization (green and purple):** Convolution filters and normalization operators; both are relatively small here.
> - **Other (violet):** Small remaining timed operations, such as tensor creation.
> - **Overhead (gray):** Full-run time outside the grouped operator events; this can include runner and framework overhead.

![전체 실행 속도 향상 뒤의 연산자 비용과 오버헤드 / Operator costs and overhead behind the whole-run speedup](/assets/blog/2026-10-26-optimizing-streaming-speech-recognition-on-arm-edge-devices-with-executorch/5-2.png){:style="width:100%"}

*그림 5. 전체 실행 속도 향상 뒤의 연산자 비용과 오버헤드 / Figure 5. Operator costs and overhead behind the whole-run speedup.*

ETDump 시간은 인코더, 예측기, 결합기 네트워크 전체에 걸쳐 합산한 값입니다. 여기에 남은 오버헤드를 더하면 그림 4의 전체 처리 시간이 됩니다.
> ETDump times are summed across the encoder, predictor, and joiner networks. Adding residual overhead gives the full processing time shown in Figure 4.

**속도 향상의 대부분은 GEMM에서 나옵니다.**
> **GEMM drives most of the speedup.**

SME2를 켜면 GEMM 시간이 **FP16은 8.2초에서 3.1초로**, **QC8W는 4.1초에서 2.0초로** 줄어듭니다. GEMM은 여전히 실행 시간에서 가장 큰 부분이지만, 전체 처리 시간에서 차지하는 비중은 FP16에서 84%에서 67%로, QC8W에서 78%에서 62%로 줄어듭니다. 데이터 이동은 두 빌드에서 걸리는 시간이 거의 같으므로, GEMM이 빨라질수록 상대적 비중이 커집니다. FP16은 8%에서 17%로, QC8W는 9%에서 17%로 늘어납니다. 이렇게 남은 작업이 전체 속도 향상을 제한하며, 추가 최적화의 대상이 됩니다.
> Enabling SME2 reduces GEMM time from **8.2 to 3.1 seconds for FP16** and **4.1 to 2.0 seconds for QC8W**. GEMM remains the largest part of the run, though its share of total processing time falls from 84% to 67% for FP16 and 78% to 62% QC8W. Data movement takes roughly the same time for both builds, so its relative share grows as GEMM accelerates: from 8% to 17% for FP16 and 9% to 17% for QC8W. This remaining work limits the overall speedup and offers a target for further optimization.

## 어떤 커널이 실행되었을까? / Which kernels ran?

ETDump는 비용이 큰 연산자 계열이 어디인지 알려주지만, 그 연산자를 실행하는 네이티브 마이크로커널의 이름은 알려주지 않습니다.
> ETDump locates the expensive operator families, but does not name the native microkernels that execute them.

[Arm Streamline](https://developer.arm.com/tools-and-software/streamline-performance-analyzer)으로 네이티브 호출 경로를 샘플링하고 하드웨어 카운터를 측정할 수 있습니다. Pixel 11에서 측정한 SME 활용률은 **FP16이 38.4%**, **QC8W가 53.7%** 였습니다. 이 값은 각 캡처 시간 중 행렬 하드웨어가 할당된 비율을 나타내며, 최대 연산 처리량 대비 비율이 아닙니다. 또한 GEMM 함수를 샘플링해 각 빌드가 선택한 마이크로커널을 확인했습니다.
> [Arm Streamline](https://developer.arm.com/tools-and-software/streamline-performance-analyzer) allows us to sample native call paths and measure hardware counters. On the Pixel 11, we measured **38.4% SME utilization for FP16** and **53.7% for QC8W**. This indicates how much of each capture the matrix hardware was allocated; it is not a percentage of peak arithmetic throughput. We also sampled GEMM functions to identify the microkernels selected by each build.

함수 이름을 보면 샘플링된 경로를 알 수 있습니다. `_sme2_mopa` 접미사는 행렬 외적 누적(matrix outer-product accumulation) 루틴을 뜻합니다. NEON과 I8MM은 대조(control) 경로에서 쓰는 벡터 명령어 계열입니다.
> The function names reveal the sampled paths. The `_sme2_mopa` suffix denotes a matrix outer-product accumulation routine. NEON and I8MM are the vector instruction families used by the control paths.

| 정밀도 / Precision | SME2 | 샘플링된 GEMM 경로와 함수 / Sampled GEMM paths and functions |
| --- | --- | --- |
| FP16 | True | **KleidiAI FP16 SME2**<br>`kai_kernel_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa`<br><br>**XNNPACK FP16 NEON**<br>`xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75` |
| FP16 | False | **XNNPACK FP16 NEON**<br>`xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75` |
| QC8W | True | **KleidiAI INT8 SME2, FP32 출력 / FP32 output**<br>`kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa`<br><br>**KleidiAI FP32 SME2**<br>`kai_kernel_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa` |
| QC8W | False | **KleidiAI INT8 NEON I8MM, FP32 출력 / FP32 output**<br>`kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm` |

이 캡처로 Pixel 11에서 FP16과 QC8W 모두 SME2를 대상으로 하는 KleidiAI GEMM 커널이 다른 커널 경로와 함께 실행되었음을 확인했습니다. FP16 빌드에서는 일부 GEMM 연산이 여전히 NEON을 사용합니다. QC8W 빌드에서는 일부 GEMM 연산이 여전히 FP32로 실행됩니다.
> These captures confirm that SME2-targeted KleidiAI GEMM kernels ran on the Pixel 11 for both FP16 and QC8W, alongside other kernel paths. In the FP16 build, some GEMM operations still use NEON. In the QC8W build, some GEMM operations still run in FP32.

이는 생태계의 **도구 공백(tooling gap)** 을 보여줍니다. ETDump는 모델 연산자 시간을, Streamline은 네이티브 함수 샘플을 보여주지만, 어느 도구도 이 두 관점을 직접 연결하지 않습니다. 개별 모델 계층을 관찰된 커널 함수에 대응시키는 일은 향후 과제로 남아 있습니다.
> This highlights a **tooling gap** in the ecosystem. ETDump exposes model operator timings, while Streamline samples native functions, but neither tool directly connects these views. Mapping individual model layers to the observed kernel functions remains future work.

## 동작점 고르기 / Choosing an operating point

배포할 때는 정밀도와 청크 크기를 조합하며 응답 지연 시간과 정확도의 균형을 맞춰야 합니다. 애플리케이션은 정확도 목표에서 출발할 수도, 지연 시간 예산에서 출발할 수도 있습니다. 아래 예시는 두 접근을 모두 보여줍니다.
> Deployment involves balancing response latency and accuracy across precision and chunk size. An application might start with an accuracy target or a latency budget; the following examples illustrate both approaches.

아래 그림 6은 호스트 WER 결과를 **SME2를 켠** FP16과 QC8W의 Pixel 11 구성 요소 벤치마크와 짝지어 보여줍니다. 회색 눈금은 오디오 수집이 끝나는 시점을 표시하고, 점선 구간은 측정한 연산 지연을 더한 것입니다.
> Figure 6 below pairs the host WER results with a Pixel 11 component benchmark for **SME2-enabled** FP16 and QC8W. Grey ticks mark when audio collection finishes; dotted segments add the measured compute delay.

x축은 **조건부 첫 토큰까지의 시간 추정값**으로, 오디오 수집 시간에 인코더-예측기-결합기 시퀀스 시간의 중앙값을 더한 값입니다. 구성 요소 벤치마크는 합성 입력과 새로 초기화한 스트림 상태를 사용했고, 구성마다 세 라운드에 걸쳐 75개 샘플을 측정했습니다. 첫 출력에서 텍스트가 나온다고 가정합니다.
> The x-axis is a **conditional time to first-token estimate**: audio collection time plus the median encoder–predictor–joiner sequence time. The component benchmark used synthetic input, fresh stream state, and 75 measured samples per configuration across three rounds. It assumes the first output emits text.

![SME2를 켠 상태의 WER과 추정 첫 토큰 지연 시간 / WER versus estimated first-token latency with SME2 enabled](/assets/blog/2026-10-26-optimizing-streaming-speech-recognition-on-arm-edge-devices-with-executorch/6-1.png){:style="width:100%"}

*그림 6. SME2를 켠 상태의 WER과 추정 첫 토큰 지연 시간 / Figure 6. WER versus estimated first-token latency with SME2 enabled.*

예시로 **WER 상한을 13%** 로 두면, FP16과 QC8W 모두 **320, 560, 1,120ms** 청크 크기에서 기준을 충족합니다. 320ms 청크는 더 빠른 응답에 유리하며, WER은 **FP16이 12.67%**, **QC8W가 12.97%** 입니다. 청크가 길수록 정확도가 높아져 1,120ms에서 각각 **11.50%** 와 **11.72%** 에 이르지만, 전사 결과가 나타나기 시작하기 전에 더 많은 오디오가 도착해야 합니다.
> For an illustrative **13% WER ceiling**, both FP16 and QC8W qualify at **320, 560, and 1,120 ms** chunk sizes. The 320 ms chunks favor earlier responses, with **12.67% WER for FP16** and **12.97% for QC8W**. Longer chunks improve accuracy, reaching **11.50%** and **11.72%**, respectively, at 1,120 ms, but require more audio to arrive before transcripts begin to appear.

**지연 시간 예산이 800ms**라면, 두 정밀도 모두 **320ms**와 **560ms**에서 예산 안에 들어오고, 예시로 든 **WER 상한 13%** 보다도 낮습니다. 320ms 구성은 더 빠른 응답에, 560ms 구성은 더 나은 정확도에 유리합니다. 560ms에서 FP16은 **12.01% WER**과 추정 지연 **691ms**를, QC8W는 **12.26%** 와 **660ms**를 보입니다. 이 경우 80ms 구성은 두 정밀도 모두 WER 목표를 충족하지 못합니다. 반대로 1,120ms 구성은 두 정밀도 모두 지연 시간 예산을 넘습니다.
> For an **800 ms latency budget**, both precisions at **320 ms** and **560 ms** fall within the budget and also below the illustrated **13% WER ceiling**. The 320 ms configurations favor earlier responses while 560 ms gives better accuracy. At 560 ms, FP16 combines **12.01% WER** with an estimated **691 ms** delay, while QC8W combines **12.26%** with **660 ms**. Here, both 80 ms configurations miss the WER target. Conversely, both 1,120 ms configurations miss the latency budget.

## 결론: 개발자에게 주는 의미 / Conclusion: What does this mean for developers

- **의도한 청크 크기로 스트리밍 ASR을 평가하세요.** 전체 녹음 기준 정확도로는 스트리밍 품질을 예측할 수 없습니다. 여기서는 청크가 길수록 최종 전사 결과의 정확도가 높아졌지만, 응답 지연 시간도 늘어났습니다.
- **ExecuTorch는 PyTorch 모델을 최적화된 Arm CPU 커널에 연결합니다.** ExecuTorch의 내보내기와 런타임 경로 덕분에 XNNPACK이 대상 연산에 KleidiAI 마이크로커널을 선택할 수 있습니다.
- **SME2는 처리 여유를 더 만들어 줍니다.** 행렬 작업을 가속하면 ASR에 쓰는 시간이 줄어, 기기에서 다른 애플리케이션 작업이나 동시에 도는 워크로드에 더 많은 여유를 남깁니다.
- **ETDump는 연산자 수준의 속도 향상과 남은 병목을 드러냅니다.** 연산자 시간을 범주별로 묶어 보면 이득이 어디서 오는지, 그리고 데이터 이동이 처리 시간에서 차지하는 비중이 어떻게 커지는지 알 수 있습니다.
- **Streamline은 실제로 실행된 네이티브 마이크로커널을 식별합니다.** 네이티브 샘플로 SME2 커널이 선택되었음을 확인하고, 하드웨어 카운터로 행렬 하드웨어 할당량을 볼 수 있습니다.
- **청크 크기와 정밀도를 함께 고르세요.** 정밀도와 청크 크기 조합을 비교하면 애플리케이션에 맞게 응답 지연 시간과 전사 정확도의 균형을 맞출 수 있습니다.

> - **Evaluate streaming ASR at the intended chunk size.** Whole-recording accuracy does not predict streaming quality. Longer chunks improve final-transcript accuracy here, while increasing response latency.
> - **ExecuTorch connects PyTorch models to optimized Arm CPU kernels.** Its export and runtime path lets XNNPACK select KleidiAI microkernels for eligible operations.
> - **SME2 creates more processing headroom.** Accelerating matrix work reduces the time spent on ASR, leaving more room for other application tasks or concurrent workloads on the device.
> - **ETDump reveals operator speedups and remaining bottlenecks.** Grouping operator timings shows where the gains come from and how data movement becomes a larger share of processing time.
> - **Streamline identifies the native microkernels that ran.** Native samples confirm selection of SME2 kernels, while hardware counters show matrix hardware allocation.
> - **Choose chunk size and precision together.** Comparing precision and chunk-size combinations helps developers balance response latency against transcript accuracy for their application.
