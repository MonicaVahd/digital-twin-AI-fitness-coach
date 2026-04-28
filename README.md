# Digital Twin AI Fitness Coach

Streamlit-based multimodal fitness coaching app with:
- pose analysis (MediaPipe + OpenPose subprocess path),
- LLM feedback (OpenAI / Groq via CrewAI + LangChain),
- voice input/output (Whisper + gTTS),
- memory storage (PostgreSQL + Redis).

## Quick Start

### 1) Create environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Configure environment
```bash
cp .env.example .env
```
Fill keys and DB settings in `.env`.

### 4) Run app
```bash
python -m streamlit run app.py
```

## Required Services
- PostgreSQL (for long-term memory)
- Redis (for short-term memory)

## Notes
- OpenPose is executed through local binary paths configured in `app.py`.
- Do not commit `.env`, generated media, caches, or build artifacts.
