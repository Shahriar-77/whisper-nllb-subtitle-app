![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/github/license/Shahriar-77/whisper-nllb-subtitle-app)
![Streamlit App](https://img.shields.io/badge/Streamlit-ready-brightgreen?logo=streamlit)
![GPU-Optimized](https://img.shields.io/badge/GPU%20Optimized-NVIDIA-green)

# 🎧 Whisper-NLLB Subtitle Translator App

### A Fast, Fully-Offline Subtitle Generator and Translator Powered by Whisper & NLLB-200

A GPU-accelerated, fully offline application built with **OpenAI Whisper**, **Meta’s NLLB-200**, and **Streamlit** to:

- Automatically transcribe or translate audio/video content into English using state-of-the-art ASR (Automatic Speech Recognition)
- Generate subtitles in `.txt`, `.srt`, `.ass`, and `.vtt` formats
- Translate subtitle files into multiple languages with high-quality NLLB models — preserving timing and formatting
- Provide an intuitive step-wise interface for managing media, models, and subtitle pipelines

Ideal for content creators, educators, researchers, and localization specialists who need accurate and customizable subtitles — all without relying on cloud APIs or sending data externally.

---

## ✨ Key Features

- ✅ **Offline, Privacy-Friendly**: No internet required after setup. All transcription and translation happens locally on your machine.
- 🧠 **Powered by OpenAI Whisper (via Faster-Whisper)**: Supports accurate transcription and direct audio translation to English.
- 🌍 **NLLB-200 Translation Support**: Translates `.srt`, `.ass`, and `.vtt` subtitle files to over 200 languages using Meta’s multilingual model.
- 🗂 **Batch Processing**: Transcribe and subtitle multiple media files in a single run.
- 📼 **Multi-Format Subtitle Generation**: Outputs subtitles in `.txt`, `.srt`, `.ass`, and `.vtt` with precise timestamps.
- 🧹 **Smart Preprocessing**: Auto-renames files for compatibility and converts to 16kHz mono `.wav` format using `ffmpeg`.
- 🧪 **Reprocessing Safeguards**: Automatically detects existing transcripts and subtitles; warns users before overwriting.
- 🧩 **Streamlit UI**: Clean, step-by-step interface with sidebar configuration and real-time status tracking.
- 🌐 **Manual Language Control**: Whisper tasks support high-accuracy languages only; NLLB-200 offers freeform code input with guidance.
- 🔓 **Optional Whisper Language Expansion**: Unlock support for all Whisper languages (with potential tradeoffs in accuracy).
- 🎬 **Expanded Format Support**: Supports `.ts` and `.m4a` input files in addition to standard formats.

---

## 📚 Table of Contents

1. [Key Features](#-key-features)  
2. [Demo](#-demo)  
3. [Installation & Setup](#-installation--setup)  
4. [Features](#-features)  
5. [Project Structure](#-project-structure)  
6. [Contributing](#-contributing)  
7. [License](#-license)  
8. [Acknowledgements](#-acknowledgements)  
9. [Future Work](#-future-work)  

---

## 📸 Demo

👉 [Click to watch the demo](https://github.com/shahriar-77/whisper-nllb-subtitle-app/blob/main/demo/demo-video.mp4)

---

## 🚀 Installation & Setup

### 🔧 Prerequisites

- Python 3.8 or newer  
- [ffmpeg](https://ffmpeg.org/download.html) (for audio/video conversion)  
- NVIDIA GPU with CUDA support (for optimal Whisper + NLLB inference)  
- [Git](https://git-scm.com/) (optional, for cloning)

### 📦 Step-by-Step Setup

#### 1. Clone the repository

```bash
git clone https://github.com/your-username/whisper-nllb-subtitle-app.git
cd whisper-nllb-subtitle-app
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If you encounter `torch` or `transformers` compatibility issues:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate
```

#### 3. Verify `ffmpeg` is installed

```bash
ffmpeg -version
```

If not, install it from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your PATH.

#### 4. Run the App


```bash
streamlit run app.py
```

---

### 📁 Default Folder Structure

```bash
samples/                  # Your input audio/video files 
outputs/
  └── audio_wav/          # Preprocessed .wav files
  └── subtitles/          # Generated .txt, .srt, .ass, .vtt subtitles
```

---

## 🧠 Features

### 🔊 Audio & Video Transcription (Whisper-powered)

- ✅ Supports `.mp4`, `.mkv`, `.avi`, `.ts`, `.m4a`, `.mp3`, `.wav`, `.flac`
- 🔁 Converts to 16kHz mono WAV for Whisper compatibility
- 🧠 GPU-accelerated transcription using [`faster-whisper`](https://github.com/guillaumekln/faster-whisper)
- 🌍 Supports transcription (`Transcribe`) or translation-to-English (`Translate`)
- 🎯 Source language restricted to **24+ Whisper-medium languages with WER < 20%**
- 🔓 Option to unlock support for all Whisper languages **(with potential tradeoffs in accuracy)**
- 📦 Batch transcription with skipping logic and progress bar

### 💬 Subtitle Generation

- 📄 Generates `.txt`, `.srt`, `.ass`, `.vtt` files
- 🧠 Smart segmentation using:
  - Word-level chunking if timestamps exist
  - Time-bound fallback slicing
- 🛠 Auto-repair of overlapping/malformed segments

### 🌐 Subtitle Translation (Text-based)

- Uses Meta’s [`nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)
- 🔗 Input source/target language codes manually (no auto-detect)
- 📁 Translates `.srt`, `.ass`, `.vtt` formats
- ✅ Skips already translated versions
- 🔘 Select All option for batch translation

### 🖥️ Streamlit Interface

- 🧱 Step-by-step navigation (scan → preprocess → model → transcribe → translate)
- 🎛 Sidebar settings for task, model size, and language
- 🔁 Reprocessing safeguards & overwrite warnings
- 📤 Live transcript previews for completed files (first 2–4 lines per format)
- 🔄 Full app reset button

---

## 📦 Project Structure

```bash
📁 your-project-root/
├── app.py            # Streamlit UI controller
├── main.py           # Whisper processing, audio conversion, subtitle formatting
├── translator.py     # NLLB-based subtitle translation
├── outputs/
│   ├── audio_wav/    # WAV-processed audio
│   └── subtitles/    # Subtitle outputs (.txt, .srt, .ass, .vtt)
├── samples/          # Input files (audio/video)
├── requirements.txt  # Dependencies
└── README.md         # Documentation
```

---

## 🤝 Contributing

Contributions welcome! Feel free to:

- Fix bugs or improve performance
- Add support for new subtitle formats
- Extend UI with preview/download/export
- Create Docker or Hugging Face Space deployment

Fork the repo, create a feature branch, and submit a PR 🚀

---

## 📜 License

This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper)
- [Meta NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb)
- [Streamlit](https://streamlit.io/)
- [DL-Translate Language Codes](https://dl-translate.readthedocs.io/en/latest/available_languages/#nllb-200)

---

## 🔮 Future Work

- [ ] Add subtitle preview/download directly in Streamlit UI
- [ ] Auto-detect source language using Whisper/NLLB hybrid logic
- [ ] Subtitle QC: WER/translation quality scoring
- [ ] Deploy on Hugging Face or Streamlit Cloud
- [ ] Add voice activity detection and silence trimming preprocessor

---