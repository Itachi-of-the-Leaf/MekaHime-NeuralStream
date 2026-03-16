# MekaHime Streaming Core: Real-Time Multi-Target BSS

MekaHime Streaming Core is a state-of-the-art server designed for **16kHz Real-Time Continuous Speech Separation (CSS)** and **Live Transcription**. It enables simultaneous extraction of multiple enrolled speakers from a single audio stream, delivering high-fidelity voice isolation even in overlapping speech scenarios.

## 🚀 Core Features

- **Multi-Target Blind Source Separation (BSS)**: Leverages **Asteroid Conv-TasNet** for aggressive, high-quality audio separation.
- **Smart Speaker Tracking**: Real-time channel locking powered by **TitaNet** voiceprint induction and Asymmetric EMA scoring.
- **Real-Time Speech-to-Text**: Integrated **faster-whisper** (tiny.en) for near-instant transcription of separated segments.
- **Hardware Stability**: Robust WebSocket streaming with a **Byte Accumulator** to handle variable Windows WASAPI hardware buffer sizes.
- **Dynamic Enrollment**: Live enrollment trigger allows registering new voiceprints via HTTP without server restart.

## 🏗️ Architecture Flow

`Mic` → `WebSocket` → `Byte Accumulator` → `VAD` → `BSS (Conv-TasNet)` → `TitaNet EMA Locking` → `faster-whisper` → `OLA Buffer` → `Output WAV`

## 🛠️ Setup & Installation

### 1. Prerequisites
- **Python 3.10+** (Conda recommended)
- **NVIDIA GPU** with CUDA support.
- **ffmpeg** installed on the system.

### 2. Dependencies
```bash
conda activate mekahime_core
pip install fastapi uvicorn websockets sounddevice numpy torch torchaudio faster-whisper
# Additional model-specific packages (NeMo/Asteroid) should be installed per environment guidelines.
```

## 🏃 Performance & Usage

### Starting the Server
```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

### Running the Live Client (Windows)
```bash
python src/utils/live_enroll_and_test.py
```
1. **Stage 1**: Automated HTTP trigger for enrollment.
2. **Stage 2**: Speak for 6 seconds to register your voiceprint.
3. **Stage 3**: Play background interference and speak to test BSS separation.
4. **Stage 4**: Auto-exit after 3s of silence or 15s hard timeout.

## 📂 Project Structure
- `src/api/server.py`: FastAPI server with WebSocket entry point.
- `src/core/inference.py`: Multi-target BSS and scoring logic.
- `src/utils/live_enroll_and_test.py`: Stage-based testing client.
- `data/chroma_db/`: Local vector store for speaker voiceprints.
- `data/test_samples/`: Extracted speaker-isolated WAV files.
