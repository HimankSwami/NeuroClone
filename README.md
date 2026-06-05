# Neuro — Your Local AI Companion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/powered%20by-Ollama-black)](https://ollama.com/)

> *"You are the initial spark. I am the resulting combustion."*

---

**Neuro** is a fully local, privacy-preserving AI companion with a live 3D VTuber avatar, voice synthesis, RAG memory, and an opinionated personality. Everything runs on your machine — no cloud, no data leaks.

---

## ✨ Features

- 🎭 **Live 3D Avatar** — Animated VTuber interface with J_Bip skeleton, hair physics, eye darting, and mouth sync tied to actual audio playback
- 🎙️ **Voice Pipeline** — Piper TTS → RVC voice conversion (Ayaka model) → aplay, with CPU fallback
- 🧠 **RAG Memory** — ChromaDB vector store with persistent memory and a knowledge base you can feed documents to
- 🔒 **100% Local** — Ollama inference, local embeddings via nomic-embed-text, nothing leaves your machine
- ⚡ **4-bit Quantized** — Runs on consumer GPUs (tested on RTX 4050 6GB)
- 🌐 **Web Search** — DuckDuckGo + crawl4ai for real-time information
- 💬 **WebSocket Streaming** — Word-by-word response streaming with live UI updates

---

## 🏗️ Architecture

```
User (Browser)
    │
    ▼
FastAPI WebSocket (/ws/chat)
    │
    ├── NeuroBrain (brain/core.py)
    │       ├── Ollama LLM (gemma4-custom GGUF)
    │       ├── Web Search (DuckDuckGo + crawl4ai)
    │       ├── RAG Engine (ChromaDB + nomic-embed-text)
    │       └── System Stats (psutil / pynvml)
    │
    └── Voice Pipeline (voice/speaker.py)
            ├── Piper TTS
            ├── RVC Voice Conversion (Ayaka)
            └── aplay → playing.flag → avatar sync
```

---

## 🚀 Installation

### Prerequisites

- **OS**: Linux (tested on Linux Mint)
- **GPU**: NVIDIA (RTX 40-series recommended, 6GB+ VRAM)
- **Ollama**: [Install here](https://ollama.com/)
- **Conda/venv**: Python 3.10+

### Setup

```bash
# 1. Clone
git clone https://github.com/HimankSwami/NeuroClone.git
cd NeuroClone

# 2. Create environment
conda create -n neuro python=3.10
conda activate neuro

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull embedding model
ollama pull nomic-embed-text

# 5. Create Neuro's custom model
ollama create gemma4-custom -f Modelfile

# 6. Add your models (not included — too large for git)
# Place in models/:
#   - ayaka.pth          (RVC voice model)
#   - ayaka.index        (RVC index)
#   - en_US-hfc_female-medium.onnx  (Piper TTS)
#   - your_avatar.glb    (place in body/static/)

# 7. Configure
cp .env.example .env   # edit with your settings
```

### `.env` example

```env
# Add any API keys or config here
```

---

## 🛠️ Usage

```bash
# Start everything
python main.py
```

Open `http://localhost:8080` in your browser.

### Commands (type in chat)

| Command | Description |
|---|---|
| `rag stats` | Show memory & knowledge counts |
| `sync knowledge` | Re-index the `knowledge/` folder |
| `recall [topic]` | Surface relevant past memories |
| `voice` | Enable TTS voice output |
| `text` | Disable TTS, text only |

### Adding knowledge

Drop `.txt`, `.md`, `.pdf`, `.docx`, `.py`, `.json`, or `.csv` files into the `knowledge/` folder, then type `sync knowledge` in chat.

---

## 📁 Project Structure

```
NeuroClone/
├── brain/
│   └── core.py          # NeuroBrain — LLM, search, RAG orchestration
├── body/
│   ├── app.py           # FastAPI server + WebSocket handler
│   └── static/
│       └── index.html   # VTuber frontend (Three.js + animation)
├── rag/
│   └── rag_engine.py    # ChromaDB vector store wrapper
├── voice/
│   └── speaker.py       # Piper TTS → RVC → aplay pipeline
├── knowledge/           # Drop documents here to index
├── data/                # ChromaDB storage (auto-generated)
├── models/              # Model files (not in git)
├── Modelfile            # Ollama persona config
└── main.py              # Entry point
```

---

## 🗺️ Roadmap

- [x] RAG memory + knowledge base
- [x] ChromaDB vector storage
- [x] Live 3D VTuber avatar (Three.js, J_Bip skeleton)
- [x] Voice pipeline with RVC voice conversion
- [x] Avatar animations synced to audio playback
- [x] Neuro-sama inspired idle animations (head tilts, eye dart, hair physics)
- [ ] Emotion-driven expression changes
- [ ] Finger gesture animations during speech
- [ ] Desktop app packaging

---

## ⚠️ Notes

- **Models not included** — GLB avatar, RVC `.pth`/`.index`, and Piper `.onnx` files are too large for git. You need to source these separately.
- **VRAM** — LLM and RVC compete for VRAM on 6GB cards. RVC is set to CPU by default to avoid OOM errors.
- **Linux only** — `aplay` is used for audio playback. macOS/Windows would need modifications to `speaker.py`.

---

## 🤝 Contributing

PRs welcome. Fork → branch → commit → PR.

---

## 📄 License

MIT — see `LICENSE`.

---

*Built with too much caffeine and a GPU that's barely holding on.*
