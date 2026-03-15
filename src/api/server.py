from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import logging
import torch
from src.core.memory import RingBuffer
from src.core.vad import VADEngine
from src.core.speaker_db import SpeakerDB
from src.core.inference import InferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MekaHime Streaming Core")

# Global reference to components (Lazy Initialization)
audio_buffer: RingBuffer = None
vad_engine: VADEngine = None
speaker_db: SpeakerDB = None
inference_engine: InferenceEngine = None

TARGET_SPEAKER_ID = "primary_user"

# Task 15: Global EMA AGC tracking
ema_energy = 0.01

@app.on_event("startup")
async def startup_event():
    global audio_buffer, vad_engine, speaker_db, inference_engine
    if audio_buffer is None:
        audio_buffer = RingBuffer(name="/tmp/mekahime_audio_buffer.bin", size=16000*60) # 60s buffer
    vad_engine = VADEngine()
    speaker_db = SpeakerDB()
    inference_engine = InferenceEngine()
    logger.info("All components (VAD, DB, Inference) initialized and warm.")

@app.websocket("/ws/audio_stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    global audio_buffer, vad_engine, speaker_db, inference_engine
    
    import torchaudio
    session_audio = []
    prev_chunk = None
    overlap_len = 480 # 30ms overlap for aggressive transient suppression
    
    # Pre-calculate crossfade curves (Linear slope)
    fade_out = torch.linspace(1.0, 0.0, overlap_len)
    fade_in = torch.linspace(0.0, 1.0, overlap_len)
        
    # Rule 8: Accumulator for speech duration telemetry
    speech_duration = 0.0
    
    # Pre-fetch target voiceprint for minimal latency
    target_match = speaker_db.collection.get(ids=[TARGET_SPEAKER_ID], include=['embeddings'])
    embeddings = target_match.get('embeddings')
    if embeddings is not None and len(embeddings) > 0:
        target_embedding = torch.tensor(target_match['embeddings'][0], dtype=torch.float32).cuda().unsqueeze(0)
    else:
        logger.warning(f"Target speaker {TARGET_SPEAKER_ID} not found in SpeakerDB. Using dummy.")
        target_embedding = torch.randn(1, 192).cuda()
    
    # Task 16: Overlap-Discard Context Buffer
    CONTEXT_SIZE = 24000  # 1.5 seconds of context at 16kHz
    context_buffer = np.zeros(CONTEXT_SIZE, dtype=np.float32)
    
    try:
        while True:
            # Receive raw PCM16 bytes
            data = await websocket.receive_bytes()
            if len(data) != 1024:
                continue

            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # 1. Update rolling context buffer (Overlap-Discard)
            context_buffer = np.roll(context_buffer, -512)
            context_buffer[-512:] = audio_float32
            
            # 2. Step VAD state silently
            _ = vad_engine.process_chunk(audio_float32)
            
            # 3. Infer strictly once per frame IF speech is active
            cleaned_audio = None
            if vad_engine.is_speech_active:
                chunk_tensor = torch.from_numpy(context_buffer).cuda().view(1, 1, CONTEXT_SIZE).to(torch.float32)
                cleaned_audio = inference_engine.extract_voice(chunk_tensor, target_embedding)
            else:
                # Output silence when VAD is negative (Match context buffer size)
                cleaned_audio = torch.zeros(1, 1, CONTEXT_SIZE, device='cuda')

            if cleaned_audio is not None and cleaned_audio.numel() > 0:
                # OLA: Grab 512-hop + overlap_len lookahead.
                # [-1024 : -(512-overlap_len)] = [-1024 : -32] for overlap_len=480.
                extracted_audio_full = cleaned_audio.squeeze().cpu()
                current_chunk = extracted_audio_full[-1024:-32] 
                
                if current_chunk.numel() == (512 + overlap_len):
                    if prev_chunk is not None:
                        # Crossfade the tail of prev and head of current
                        blended = (prev_chunk[-overlap_len:] * fade_out) + (current_chunk[:overlap_len] * fade_in)
                        
                        prev_chunk[-overlap_len:] = blended
                        current_chunk[:overlap_len] = blended
                        
                        # COMMIT: Save the 512 samples. 
                        session_audio.append(prev_chunk[:512])
                        
                        prev_chunk = current_chunk
                    else:
                        # Start of stream ramp-in
                        initial_ramp = torch.linspace(0.0, 1.0, 512)
                        current_chunk = current_chunk.clone()
                        current_chunk[:512] *= initial_ramp
                        prev_chunk = current_chunk
                
                if vad_engine.is_speech_active:
                    speech_duration += 0.032
            
            if not vad_engine.is_speech_active and speech_duration > 0:
                logger.info(f"[SYSTEM] Speaker identified: {TARGET_SPEAKER_ID}. Total duration: {round(speech_duration, 4)}s.")
                speech_duration = 0.0
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("Closing connection. Saving session audio...")
        if prev_chunk is not None:
            # Commit the last 512 samples
            session_audio.append(prev_chunk[:512])
            
        if session_audio:
            final_audio = torch.cat(session_audio, dim=-1).unsqueeze(0)
            
            # GLOBAL EOS NORMALIZATION:
            max_amp = final_audio.abs().max()
            if max_amp > 0.0:
                final_audio = (final_audio / max_amp) * 0.90
                
            torchaudio.save("data/test_samples/extracted_output.wav", final_audio, 16000)
            logger.info(f"Successfully saved globally normalized audio ({final_audio.shape[-1]/16000:.2f}s).")
        
        vad_engine.reset()
        logger.info("WebSocket endpoint cleanup complete.")
