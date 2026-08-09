# 🛡️ AI Voice Safety Agent

<p align="right">
  <strong>English</strong> · <a href="./README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <strong>A multimodal AI safety system for voice spoof detection, scam-risk reasoning, and trusted human assistance.</strong>
</p>

<p align="center">
  Voice Spoof Detection × Speech Transcription × LLM Risk Reasoning × Human-Centered AI
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/Laura-smith/voice-spoof-detector">🌐 Live Demo</a>
  ·
  <a href="#6-demo">🎥 Demo</a>
  ·
  <a href="#10-research-findings">🔬 Research Findings</a>
  ·
  <a href="#14-quick-start">⚙️ Quick Start</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="FastAPI" src="https://img.shields.io/badge/Backend-FastAPI-009688">
  <img alt="PyTorch" src="https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C">
  <img alt="Hugging Face" src="https://img.shields.io/badge/Deployment-Hugging%20Face-yellow">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

> [!IMPORTANT]
> AI Voice Safety Agent is a research and engineering prototype.
> It provides supporting evidence and safety recommendations, but it must not be treated as a final legal, financial, identity-verification, or law-enforcement decision system.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Why This Project Matters](#2-why-this-project-matters)
- [3. From a Voice Classifier to a Safety System](#3-from-a-voice-classifier-to-a-safety-system)
- [4. Core Features](#4-core-features)
- [5. User Roles and Workflow](#5-user-roles-and-workflow)
- [6. Demo](#6-demo)
- [7. System Architecture](#7-system-architecture)
- [8. AI Analysis Pipeline](#8-ai-analysis-pipeline)
- [9. CNN Spoof Detection Model](#9-cnn-spoof-detection-model)
- [10. Research Findings](#10-research-findings)
- [11. Deployment Strategy](#11-deployment-strategy)
- [12. Technology Stack](#12-technology-stack)
- [13. Repository Structure](#13-repository-structure)
- [14. Quick Start](#14-quick-start)
- [15. Training](#15-training)
- [16. Configuration](#16-configuration)
- [17. Data, Privacy and Responsible Use](#17-data-privacy-and-responsible-use)
- [18. Current Limitations](#18-current-limitations)
- [19. Future Work](#19-future-work)
- [20. What This Project Demonstrates](#20-what-this-project-demonstrates)
- [21. Author and License](#21-author-and-license)

---

## 1. Project Overview

**AI Voice Safety Agent** is the open-source engineering repository for the product prototype **GemmaShield**.

GemmaShield is a multimodal AI safety system designed to help older adults, family members, and community workers respond to suspicious voice messages and potential AI voice scams.

The system combines:

- CNN-based voice spoof detection;
- Faster-Whisper speech transcription;
- scam-related signal extraction;
- evidence-source confidence analysis;
- Gemma-based risk reasoning;
- conservative local fallback reasoning;
- an older-adult interface;
- a family/community helper interface;
- trusted-contact binding and assistance workflows.

The project originally started as a binary classifier for distinguishing AI-generated speech from genuine human speech.

However, real-world experiments revealed an important problem:

> Strong performance on a laboratory benchmark did not automatically transfer to mobile recordings and real-world spoofing conditions.

This finding changed the project.

Instead of treating the CNN prediction as the final answer, the current system treats the model score as **one piece of acoustic evidence** and combines it with:

```text
Acoustic Evidence
+
Speech Transcript
+
Scam Risk Signals
+
Evidence Source Reliability
+
LLM Reasoning
+
Human Verification
```

The project therefore explores both:

**AI research**

```text
Deepfake Speech Detection
Domain Shift
Generalization
Robustness
```

and:

**AI application engineering**

```text
Backend
Workflow
LLM Integration
Human-in-the-loop Safety
Deployment
```

---

## 2. Why This Project Matters

Traditional voice spoof detection systems often return only a numerical prediction:

```text
Fake Probability: 0.82
```

For a non-technical user, especially an older adult, this number does not answer the most important questions:

- Why is the voice suspicious?
- Is the caller asking for money?
- Is the caller pretending to be a family member?
- Is the caller creating urgency?
- Is the audio evidence itself reliable?
- Should the user continue the conversation?
- Should a trusted family member be contacted?
- What should happen if the model is uncertain?

AI Voice Safety Agent attempts to bridge this gap.

The system transforms raw AI outputs into:

```text
Risk Level
+
Evidence Explanation
+
Safety Recommendation
+
Trusted Human Assistance
```

The design follows one central principle:

> A safety-oriented AI system should not only make a prediction. It should communicate uncertainty, explain available evidence, and allow trusted humans to intervene.

---

## 3. From a Voice Classifier to a Safety System

### Initial System

The first version focused mainly on AI-generated voice detection:

```text
Audio
  ↓
Log-Mel Spectrogram
  ↓
CNN
  ↓
Real / Fake
```

This was useful for model experimentation, but real-world evaluation revealed limitations.

### Current System

The system gradually evolved into:

```text
Suspicious Audio
        ↓
Audio Preprocessing
        ↓
 ┌───────────────┬────────────────┐
 ↓               ↓
CNN              Faster-Whisper
 ↓               ↓
Acoustic         Transcript
Evidence          ↓
        └──────┬──┘
               ↓
      Scam Risk Signals
               +
     Evidence Source Info
               ↓
      Gemma Risk Reasoning
               ↓
 Risk Level + Explanation
               ↓
Older Adult + Trusted Helper
```

The main engineering lesson is:

> When a model is imperfect under domain shift, uncertainty communication, evidence management, access control, fallback logic, and human escalation become part of the safety solution.

---

## 4. Core Features

### AI and Audio Analysis

- 16 kHz mono audio normalization;
- fixed-length waveform processing;
- 64-bin Log-Mel spectrogram extraction;
- custom CNN2D spoof detector;
- real / spoof probability estimation;
- Faster-Whisper speech recognition;
- scam-related risk-signal extraction;
- evidence-source confidence analysis;
- Gemma-based structured risk reasoning;
- conservative local fallback reasoning.

### Risk Levels

The current prototype uses four safety levels:

```text
HIGH
MEDIUM
VERIFY
LOW
```

Where:

| Level | Meaning |
|---|---|
| `HIGH` | Strong spoof evidence or serious scam signals |
| `MEDIUM` | Suspicious or uncertain evidence |
| `VERIFY` | Evidence is not reliable enough to reach a safe conclusion |
| `LOW` | No obvious high-risk signal under the available evidence |

`LOW` does **not** mean that the system has proven the audio to be safe.

---

### Product Workflow

The application supports:

- older-adult accounts;
- family/community helper accounts;
- demo phone verification;
- six-digit trusted-helper binding codes;
- helper binding requests;
- older-adult approval;
- up to five trusted helpers;
- one-click assistance requests;
- helper-side request inbox;
- permission control;
- audio recording;
- audio upload;
- evidence review;
- unbinding and trusted-contact replacement;
- simulated notification delivery;
- health monitoring;
- Hugging Face deployment.

---

### Human-Centered Design

The older-adult interface emphasizes:

- large readable text;
- high contrast;
- simple actions;
- minimal technical information;
- clear risk categories;
- actionable safety suggestions.

The helper interface provides more detailed technical evidence.

---

## 5. User Roles and Workflow

### Older-Adult Side

The older adult can:

1. complete first-time setup;
2. obtain a six-digit binding code;
3. approve family or community helpers;
4. record or upload suspicious audio;
5. send an assistance request;
6. receive a simple safety notice;
7. review previous assistance requests;
8. replace or remove trusted helpers.

The older-adult interface intentionally avoids requiring the user to interpret raw AI probability scores.

---

### Family / Community Helper Side

A trusted helper can:

1. create or access a helper account;
2. enter the older adult's six-digit binding code;
3. send a binding request;
4. wait for approval;
5. receive assistance requests;
6. inspect the evidence source;
7. review available audio;
8. upload additional evidence;
9. run CNN + Whisper analysis;
10. request Gemma risk reasoning;
11. review the detailed report;
12. help the older adult verify the situation.

---

### Two-Stage AI Analysis

The system separates evidence extraction from deeper reasoning.

#### Stage 1

```text
CNN
+
Faster-Whisper
+
Risk Signal Extraction
+
Evidence Source Confidence
```

#### Stage 2

```text
Gemma Risk Reasoning
```

If Gemma is unavailable:

```text
Local Conservative Reasoning
```

is used instead.

---

## System Screenshots

### Home Page

<p align="center">
  <img src="assets/homepage.png" alt="GemmaShield Home Page" width="92%">
</p>

---

### Older-Adult Interface

<p align="center">
  <img src="assets/elder-interface.png" alt="GemmaShield Older Adult Interface" width="92%">
</p>

---

### Family / Community Helper Interface

<p align="center">
  <img src="assets/helper-interface.png" alt="GemmaShield Family and Community Helper Interface" width="92%">
</p>

---

## 6. Demo

### Live Application

🌐 [Open GemmaShield on Hugging Face Spaces](https://huggingface.co/spaces/Laura-smith/voice-spoof-detector)

### Full Workflow Video

🎥 [Watch the complete workflow demonstration](assets/demo-video.mp4)

The video demonstrates the full interaction because the application contains multiple states that cannot be shown clearly through one screenshot.

The complete workflow includes:

```text
Older-Adult Setup
      ↓
Helper Registration
      ↓
Binding Request
      ↓
Older-Adult Approval
      ↓
One-Click Help Request
      ↓
Audio Evidence
      ↓
CNN + Whisper
      ↓
Helper Evidence Review
      ↓
Gemma Reasoning
      ↓
Safety Recommendation
```

> If the Hugging Face Space is temporarily unavailable because of regional or network restrictions, please use the recorded demonstration or run the project locally.

---

## 7. System Architecture

```mermaid
flowchart TD

    A[Suspicious Voice Evidence]

    A --> B[FFmpeg / Audio Normalization]

    B --> C[16 kHz Mono Waveform]

    C --> D[64-bin Log-Mel Spectrogram]
    D --> E[CNN2D Spoof Detector]
    E --> F[Real / Spoof Probability]

    C --> G[Faster-Whisper ASR]
    G --> H[Speech Transcript]

    H --> I[Scam Risk Signal Extraction]

    J[Evidence Source Metadata]
    J --> K[Source-Aware Confidence]

    F --> L[Evidence Package]
    I --> L
    K --> L

    L --> M{Gemma Available?}

    M -->|Yes| N[Gemma Risk Reasoning]
    M -->|No / Error| O[Local Conservative Reasoning]

    N --> P[Risk Level]
    N --> Q[Evidence Explanation]
    N --> R[Safety Recommendation]

    O --> P
    O --> Q
    O --> R

    P --> S[Older-Adult Simple Notice]

    Q --> T[Family / Community Detailed View]
    R --> T

    U[Local JSON Database]
    U <--> V[Users / Bindings / Requests / Notifications]

    V --> S
    V --> T
```

---

### Application Layers

| Layer | Responsibility |
|---|---|
| Input Layer | Recorded or uploaded suspicious voice evidence |
| Acoustic Layer | CNN-based real/spoof estimation |
| Semantic Layer | Whisper transcription and scam-signal analysis |
| Confidence Layer | Evidence-source reliability |
| Reasoning Layer | Gemma or fallback risk reasoning |
| Workflow Layer | User roles, binding, requests and permissions |
| Interaction Layer | Older-adult and helper interfaces |

---

## 8. AI Analysis Pipeline

### 8.1 Audio Normalization

Audio is converted to:

```text
Sample Rate: 16,000 Hz
Channels: Mono
Target Length: 64,000 samples
Approximate Window: 4 seconds
```

The waveform is normalized before feature extraction.

FFmpeg is used when audio format conversion is required.

---

### 8.2 Log-Mel Spectrogram

The model uses:

```text
n_fft = 1024
hop_length = 512
n_mels = 64
```

Processing:

```text
Waveform
   ↓
Mel Spectrogram
   ↓
Log Transform
   ↓
Feature Normalization
```

---

### 8.3 CNN Acoustic Evidence

The CNN produces a binary logit.

The application converts it to probabilities using:

```text
Real Probability = sigmoid(logit)

Spoof Probability = 1 - Real Probability
```

The result is treated as acoustic evidence rather than a final safety judgment.

---

### 8.4 Faster-Whisper Transcription

Faster-Whisper extracts speech content.

The current deployment is designed to remain lightweight enough for a Hugging Face Space environment.

The transcript provides semantic evidence that the CNN cannot capture.

For example:

```text
"Send the money immediately."

"Don't tell anyone."

"I am your grandson."

"Give me the verification code."
```

may contain strong fraud indicators even when acoustic evidence is uncertain.

---

### 8.5 Scam Risk Signals

The application checks for risk signals related to:

```text
Money Transfer
Verification Codes
Passwords
Urgency Pressure
Family Impersonation
Investment Promises
Bank Accounts
```

These signals provide structured information for later reasoning.

They are not used as a replacement for the LLM.

---

### 8.6 Evidence Source Confidence

One of the most important deployment findings is that the audio source affects model reliability.

The system therefore records how the evidence was obtained.

| Evidence Source | Confidence Treatment | Main Concern |
|---|---|---|
| Browser / microphone recording | Needs human review | Replay, room acoustics, device noise |
| Older-adult uploaded audio | Medium | Unknown source processing |
| Social media voice message | Medium | Compression |
| Voicemail | Medium | Codec and channel effects |
| Saved call recording | Relatively higher | Telephone channel effects |
| Helper-uploaded evidence | Relatively higher | Depends on original source |

A microphone re-recording may change acoustic artifacts.

Therefore:

```text
Low Spoof Probability
```

does not automatically become:

```text
Safe
```

---

### 8.7 Gemma Risk Reasoning

Gemma receives structured evidence such as:

```text
Real Probability
Spoof Probability
Transcript
Preliminary Risk Level
Scam Type
Risk Signals
Evidence Source
Evidence Confidence
```

Gemma generates:

```text
Risk Level
Reason
Acoustic Evidence Explanation
Textual Evidence Explanation
Advice for the Older Adult
Advice for the Helper
One-Sentence Warning
```

Gemma is used as a reasoning and communication layer.

It does not replace the CNN and does not “correct” the CNN prediction.

---

### 8.8 Fallback Reasoning

The application includes local conservative reasoning.

It is activated when:

```text
HF_TOKEN is missing
Gemma API initialization fails
Gemma API call fails
Gemma response is invalid
```

This prevents the entire application from failing because one external AI service is unavailable.

---

## 9. CNN Spoof Detection Model

The custom CNN is implemented in `model.py`.

### Architecture

```text
Input
 ↓
Conv2D
1 → 32
 ↓
BatchNorm
 ↓
ReLU
 ↓
MaxPool
 ↓
Conv2D
32 → 64
 ↓
BatchNorm
 ↓
ReLU
 ↓
MaxPool
 ↓
Conv2D
64 → 128
 ↓
ReLU
 ↓
Adaptive Average Pooling
 ↓
Flatten
 ↓
Linear
128 → 64
 ↓
ReLU
 ↓
Linear
64 → 1
```

### Input Representation

```text
Normalized 64-bin Log-Mel Spectrogram
```

Approximate tensor structure:

```text
[Batch, 1, 64, Time]
```

### Training

The model uses:

```text
BCEWithLogitsLoss
Adam Optimizer
```

The task is binary classification:

```text
Real
vs.
Spoof
```

The main research evaluation metric reported in this project is:

```text
Equal Error Rate (EER)
```

Lower EER indicates better performance.

---

## 10. Research Findings

### 10.1 Research Question

The central research question became:

> Does strong performance on a controlled anti-spoofing benchmark transfer to realistic mobile and deployment conditions?

---

### 10.2 ASVspoof Benchmark

The CNN was initially trained and evaluated using ASVspoof research data.

Under the controlled benchmark setting:

```text
ASVspoof Benchmark EER
≈ 0.00134
```

This indicated very strong matched-condition performance.

---

### 10.3 Real-World Evaluation

To evaluate deployment robustness, approximately:

```text
100 real-world audio samples
```

were additionally collected and tested.

The real-world set included conditions such as:

- mobile recordings;
- replayed speech;
- AI-generated voices;
- microphone variation;
- environmental noise;
- compressed audio;
- transmitted audio.

The result changed dramatically:

```text
Initial Real-World EER
≈ 0.30
```

This revealed a major domain gap.

---

### 10.4 Data Augmentation

Because the original real-world dataset was small, a custom audio augmentation pipeline was developed.

Approximately:

```text
100 original samples
```

were expanded into approximately:

```text
50,000 augmented samples
```

using techniques including:

```text
Noise Injection
Pitch Shifting
Replay Simulation
Frequency Perturbation
Audio Distortion
Waveform Modification
```

---

### 10.5 Transfer Learning and Fine-Tuning

Instead of completely abandoning the benchmark-trained CNN, the project explored transfer learning.

The general strategy was:

```text
ASVspoof-Pretrained CNN
       ↓
Retain Learned Representation
       ↓
Adapt Selected Layers
       ↓
Train with Augmented Real-World Data
```

After augmentation and adaptation:

```text
EER
≈ 0.03
```

---

### 10.6 Experimental Summary

| Evaluation Setting | EER |
|---|---:|
| ASVspoof Benchmark | **0.00134** |
| Initial Real-World Evaluation | **0.30** |
| Augmented + Fine-Tuned Evaluation | **0.03** |

---

### 10.7 Key Research Finding

> Extremely low benchmark EER does not guarantee real-world robustness.

The experiments indicate that spoof detection performance can be strongly affected by:

```text
Dataset Mismatch
Device Differences
Microphone Variation
Replay Channels
Room Acoustics
Codec Compression
Environmental Noise
Unseen Synthesis Methods
```

The difference between:

```text
EER ≈ 0.00134
```

and:

```text
EER ≈ 0.30
```

became one of the most important findings of the project.

---

### 10.8 Why the Finding Matters

A system may appear highly accurate when:

```text
Training Distribution
≈
Testing Distribution
```

but fail when:

```text
Deployment Distribution
≠
Training Distribution
```

This is a practical example of:

```text
Domain Shift
```

and motivates future research into:

```text
Domain Generalization
Domain Adaptation
Cross-Dataset Robustness
Unseen Attack Detection
```

---

### 10.9 Interpretation of the Improved Result

The improved:

```text
EER ≈ 0.03
```

shows that augmentation and adaptation can substantially improve experimental performance.

However, it should **not** be interpreted as a universal production accuracy guarantee.

The three reported EER values correspond to different evaluation conditions.

The most important lesson is:

> Benchmark success is not the same as deployment reliability.

---

### 10.10 How the Research Changed the Product

The research findings directly changed the application architecture.

The deployed prototype no longer uses:

```text
CNN Prediction
=
Final Decision
```

Instead:

```text
CNN Evidence
+
Transcript
+
Risk Signals
+
Evidence Source
+
LLM Reasoning
+
Human Verification
```

is used.

This connects model research with product safety design.

---

## 11. Deployment Strategy

The system follows a conservative deployment strategy.

### Unsafe Design

The application avoids:

```text
Low Spoof Probability
=
Definitely Safe
```

### Current Design

Instead:

```text
Low Spoof Probability
+
Unreliable Evidence Source
=
Needs Verification
```

Similarly:

```text
Suspicious Transcript
+
Moderate Acoustic Evidence
=
Caution / High Risk
```

The final goal is not to maximize confidence.

The goal is to avoid presenting uncertain AI predictions as absolute truth.

---

## 12. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| Application Server | Uvicorn |
| Deep Learning | PyTorch |
| Audio Processing | Torchaudio |
| Audio Reading | SoundFile |
| Audio Conversion | FFmpeg |
| Feature Representation | Log-Mel Spectrogram |
| Spoof Detection | Custom CNN2D |
| Speech Recognition | Faster-Whisper |
| LLM Reasoning | Google Gemma |
| API Client | Hugging Face Inference Client |
| Demo Database | Local JSON |
| Notifications | Simulated SMS / WeChat / Push |
| Deployment | Hugging Face Spaces |
| Evaluation | scikit-learn ROC / EER |

---

## 13. Repository Structure

```text
AI-Voice-Safety-Agent/
│
├── app.py
├── model.py
├── train.py
├── best_model.pth
├── requirements.txt
├── README.md
├── README.zh-CN.md
├── LICENSE
├── .gitignore
│
├── assets/
│   ├── homepage.png
│   ├── elder-interface.png
│   ├── helper-interface.png
│   └── demo-video.mp4
│
├── docs/
│   ├── experiment-notes.md
│   └── system-design.md
│
├── experiments/
│   └── README.md
│
└── sample_audio/
    └── README.md
```

### Main Files

| File | Purpose |
|---|---|
| `app.py` | Full FastAPI application and workflow |
| `model.py` | CNN2D model |
| `train.py` | Training and EER evaluation pipeline |
| `best_model.pth` | CNN model checkpoint |
| `requirements.txt` | Python dependencies |
| `README.md` | English documentation |
| `README.zh-CN.md` | Chinese documentation |

---

## 14. Quick Start

### Clone Repository

```bash
git clone https://github.com/lwenxuan420-lgtm/AI-Voice-Safety-Agent.git
cd AI-Voice-Safety-Agent
```

---

### Create Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Suggested dependencies include:

```text
fastapi
uvicorn[standard]
python-multipart
torch
torchaudio
soundfile
faster-whisper
huggingface-hub
numpy
pandas
scikit-learn
tqdm
```

---

### Install FFmpeg

Check:

```bash
ffmpeg -version
```

FFmpeg must be available from the command line.

---

### Model Checkpoint

Place:

```text
best_model.pth
```

in the project root.

---

### Configure Gemma

#### Windows PowerShell

```powershell
$env:HF_TOKEN="your_hugging_face_token"
$env:GEMMA_MODEL_ID="google/gemma-4-26B-A4B-it"
$env:GEMMA_PROVIDER="deepinfra"
```

#### macOS / Linux

```bash
export HF_TOKEN="your_hugging_face_token"
export GEMMA_MODEL_ID="google/gemma-4-26B-A4B-it"
export GEMMA_PROVIDER="deepinfra"
```

Gemma is optional for basic application availability.

Without `HF_TOKEN`, the application uses local conservative reasoning.

---

### Start Application

```bash
python app.py
```

Open:

```text
http://localhost:7860
```

Routes:

```text
/          Home
/elder     Older-adult interface
/helper    Family/community helper interface
/health    System health information
```

---

## 15. Training

### Dataset Configuration

Update your local dataset paths inside `train.py`.

Example:

```python
ASV_ROOTS = [
    "path/to/asvspoof/data"
]

ASV_CSV = "path/to/train.csv"
```

Real-world datasets should not be uploaded publicly unless consent and licensing requirements are satisfied.

---

### Training Pipeline

```text
Audio
 ↓
Read Waveform
 ↓
Stereo → Mono
 ↓
Resample to 16 kHz
 ↓
Waveform Normalization
 ↓
Pad / Truncate
 ↓
Log-Mel Spectrogram
 ↓
Feature Normalization
 ↓
CNN
 ↓
BCEWithLogitsLoss
 ↓
Validation EER
 ↓
Save Best Model
```

Run:

```bash
python train.py
```

---

### Reproducibility

Future research releases should document:

- exact ASVspoof subset;
- protocol files;
- random seed;
- train / validation / test split;
- augmentation parameters;
- transfer-learning configuration;
- layer-freezing strategy;
- model checkpoint selection;
- per-condition evaluation;
- cross-dataset testing;
- repeated-run variance.

---

## 16. Configuration

### Environment Variables

| Variable | Required | Purpose |
|---|---:|---|
| `HF_TOKEN` | No | Hugging Face / Gemma API access |
| `GEMMA_MODEL_ID` | No | Gemma model identifier |
| `GEMMA_PROVIDER` | No | Inference provider |
| `TORCH_NUM_THREADS` | No | CPU thread control |
| `PORT` | No | Application port |

Default application port:

```text
7860
```

---

### Optional Notification Configuration

The prototype contains optional hooks for:

```text
SMS_API_KEY
WECHAT_APP_ID
WECHAT_APP_SECRET
PUSH_API_KEY
```

The public demo currently simulates these notification channels unless a real provider is configured.

---

## 17. Data, Privacy and Responsible Use

### Research Data

The project uses:

```text
ASVspoof Research Data
```

and separately collected real-world samples for robustness experiments.

Publicly shared audio should only contain:

- synthetic audio;
- openly licensed audio;
- or recordings collected with explicit consent.

---

### Prototype Storage

The current application uses local prototype storage such as:

```text
requests_db.json
uploads/
```

for:

- demo users;
- sessions;
- binding requests;
- assistance requests;
- notifications;
- uploaded evidence.

This architecture is intended for demonstration and research.

It is **not** production-grade storage.

---

### Production Deployment Would Require

```text
Secure Authentication
Encrypted Storage
Encrypted Transport
Access Control
Secret Management
Consent Management
Data Retention Policies
Deletion Controls
Audit Logging
Abuse Monitoring
Secure Object Storage
Production Database
```

---

### Intended Use

This project is intended for:

- AI safety research;
- speech deepfake detection research;
- human-centered AI research;
- educational demonstration;
- AI application engineering;
- scam-risk assistance;
- trusted-helper workflow prototyping.

---

### Out-of-Scope Use

This project should not be used for:

- surveillance;
- unauthorized identity tracking;
- speaker identification without consent;
- secret recording;
- automatic criminal accusations;
- automatic legal decisions;
- automatic financial decisions;
- replacing banks or law-enforcement agencies;
- treating AI scores as proof of authenticity.

---

## 18. Current Limitations

The project currently has several important limitations.

### Model Limitations

- unseen spoofing methods may reduce performance;
- microphone recordings may alter spoof artifacts;
- replay channels introduce domain shift;
- compression may change acoustic characteristics;
- benchmark performance does not guarantee deployment performance.

### Speech Recognition Limitations

- noisy recordings may reduce transcription accuracy;
- accents and speech quality may affect Whisper;
- incomplete transcripts may affect risk reasoning.

### LLM Limitations

- Gemma reasoning depends on available evidence;
- incorrect transcription can influence reasoning;
- LLM output is not a legal or financial judgment.

### Engineering Limitations

- JSON persistence is prototype-only;
- current OTP is a demonstration flow;
- notification delivery is simulated;
- uploaded files are stored locally;
- the system does not yet use a production database;
- the current CNN analyzes a fixed audio window;
- the system is not yet a real-time call-monitoring service.

### Product Limitations

- current UI is Chinese-first;
- larger user testing is still needed;
- accessibility evaluation is still needed;
- production escalation protocols remain future work.

---

## 19. Future Work

### Research

Future research directions include:

```text
Cross-Dataset Evaluation
Domain Adaptation
Domain Generalization
Unseen Attack Detection
Self-Supervised Speech Models
Contrastive Learning
One-Class Learning
Uncertainty Estimation
Calibration
Replay Simulation
Robust Feature Learning
```

---

### Engineering

Future engineering improvements include:

```text
PostgreSQL / Supabase
Secure Object Storage
Real Notification Services
Task Queues
Monitoring
Logging
Automated Tests
Docker
CI/CD
Model Versioning
Streaming Audio Analysis
Edge Deployment
```

---

### Product

Future product directions include:

```text
Multilingual Interface
Older-Adult User Testing
Family User Testing
Community Worker Testing
Better Uncertainty Visualization
Emergency Escalation
Institutional Dashboard
Permission-Controlled Function Calling
```

---

## 20. What This Project Demonstrates

This project documents more than a model-training experiment.

It demonstrates the complete evolution of an AI application:

```text
Problem Discovery
        ↓
Dataset Preparation
        ↓
Model Development
        ↓
Benchmark Evaluation
        ↓
Real-World Failure Discovery
        ↓
Domain Gap Analysis
        ↓
Data Augmentation
        ↓
Transfer Learning
        ↓
Risk-Aware Product Redesign
        ↓
Backend Development
        ↓
LLM Integration
        ↓
Human-in-the-Loop Workflow
        ↓
Deployment
```

The project integrates knowledge from:

```text
Machine Learning
Audio Processing
Deepfake Detection
Model Evaluation
Backend Engineering
API Integration
LLM Reasoning
Workflow Design
AI Safety
Responsible AI
Human-Centered Product Design
```

The most important outcome is not one single metric.

It is the process of:

> discovering where a model fails, understanding why it fails, and redesigning the surrounding system so that uncertainty is visible and trusted humans remain involved.

---

## 21. Author and License

Maintainer:

**lwenxuan420-lgtm**

GitHub:

https://github.com/lwenxuan420-lgtm

Project Areas:

```text
AI Safety
Voice Security
Deepfake Speech Detection
Explainable AI
Multimodal AI Systems
Human-Centered AI
AI Application Engineering
```

This project is released under the:

**MIT License**

See:

```text
LICENSE
```

for details.

---

<p align="center">
  <strong>GemmaShield</strong>
</p>

<p align="center">
  From benchmark voice detection to real-world, human-centered AI safety.
</p>

<p align="center">
  <a href="./README.zh-CN.md">🇨🇳 阅读中文版</a>
</p>
