from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import logging
import torch
from src.core.memory import RingBuffer
from src.core.vad import VADEngine
from src.core.speaker_db import SpeakerDB
from src.core.inference import InferenceEngine
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MekaHime Streaming Core")

# Global reference to components (Lazy Initialization)
audio_buffer: RingBuffer = None
vad_engine: VADEngine = None
speaker_db: SpeakerDB = None
inference_engine: InferenceEngine = None
whisper_node = None

TARGET_SPEAKER_ID = "primary_user"

# Task 15: Global EMA AGC tracking
ema_energy = 0.01

# Enrollment State Tracker
ENROLLMENT_STATE = {
    "is_active": False,
    "speaker_id": None,
    "buffer": []
}

@app.post("/enroll/{speaker_id}")
async def trigger_enrollment(speaker_id: str):
    """
    HTTP trigger to bypass the BSS filters and record the next ~5 seconds of raw audio for enrollment.
    """
    global ENROLLMENT_STATE
    ENROLLMENT_STATE["is_active"] = True
    ENROLLMENT_STATE["speaker_id"] = speaker_id
    ENROLLMENT_STATE["buffer"] = []
    logger.info(f"🎙️ ENROLLMENT MODE ACTIVE: Listening for '{speaker_id}'...")
    return {"status": "success", "message": f"Enrollment active for {speaker_id}"}

@app.on_event("startup")
async def startup_event():
    global audio_buffer, vad_engine, speaker_db, inference_engine, whisper_node
    if audio_buffer is None:
        audio_buffer = RingBuffer(name="/tmp/mekahime_audio_buffer.bin", size=16000*60) # 60s buffer
    vad_engine = VADEngine()
    speaker_db = SpeakerDB()
    inference_engine = InferenceEngine()
    
    logger.info("Initializing Whisper ASR Node (tiny.en)...")
    whisper_node = WhisperModel("tiny.en", device="cuda", compute_type="float16")
    
    logger.info("All components (VAD, DB, Inference, Whisper) initialized and warm.")

@app.websocket("/ws/audio_stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    global audio_buffer, vad_engine, speaker_db, inference_engine, whisper_node
    
    import torchaudio
    
    # Load all enrolled profiles from ChromaDB
    db_results = speaker_db.collection.get(include=['embeddings'])
    active_profiles = {}
    if db_results is not None and db_results.get('embeddings') is not None and len(db_results['embeddings']) > 0:
        for i, spk_id in enumerate(db_results['ids']):
            active_profiles[spk_id] = torch.tensor(db_results['embeddings'][i], dtype=torch.float32).cuda().unsqueeze(0)
    else:
        logger.warning("No speakers enrolled. Using dummy.")
        active_profiles["dummy"] = torch.randn(1, 192).cuda()

    # OLA Buffers per speaker
    session_audio = {spk: [] for spk in active_profiles.keys()}
    prev_chunks = {spk: None for spk in active_profiles.keys()}
    speech_duration = {spk: 0.0 for spk in active_profiles.keys()}
    
    # Tracking variables for Byte Accumulator and ASR
    incoming_buffer = bytearray()
    segment_start_idx = {spk: 0 for spk in active_profiles.keys()}
    
    overlap_len = 480 # 30ms overlap for aggressive transient suppression
    
    # Pre-calculate crossfade curves (Linear slope)
    fade_out = torch.linspace(1.0, 0.0, overlap_len)
    fade_in = torch.linspace(0.0, 1.0, overlap_len)
    
    # Task 16: Overlap-Discard Context Buffer
    CONTEXT_SIZE = 24000  # 1.5 seconds of context at 16kHz
    context_buffer = np.zeros(CONTEXT_SIZE, dtype=np.float32)
    
    try:
        while True:
            # Stage: Byte Accumulator for strict 1024-byte chunking
            data = await websocket.receive_bytes()
            incoming_buffer.extend(data)
            
            while len(incoming_buffer) >= 1024:
                chunk_bytes = incoming_buffer[:1024]
                del incoming_buffer[:1024]
                
                audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                
                # --- DYNAMIC ENROLLMENT INTERCEPT ---
                global ENROLLMENT_STATE
                if ENROLLMENT_STATE["is_active"]:
                    ENROLLMENT_STATE["buffer"].append(audio_float32)
                    
                    # If we have collected ~4.8 seconds of audio (150 chunks of 32ms)
                    if len(ENROLLMENT_STATE["buffer"]) >= 150:
                        logger.info("Processing captured enrollment audio...")
                        raw_audio = np.concatenate(ENROLLMENT_STATE["buffer"])
                        
                        # Convert to tensor and extract embedding using the warm TitaNet model
                        audio_tensor = torch.from_numpy(raw_audio).cuda().unsqueeze(0)
                        audio_len = torch.tensor([audio_tensor.shape[1]]).cuda()
                        
                        with torch.no_grad():
                            _, emb = inference_engine.speaker_model.forward(input_signal=audio_tensor, input_signal_length=audio_len)
                        
                        emb_np = emb.cpu().numpy().flatten()
                        emb_np = emb_np / np.linalg.norm(emb_np) # Normalize
                        
                        # Save to ChromaDB
                        speaker_db.add_voiceprint(ENROLLMENT_STATE["speaker_id"], emb_np)
                        logger.info(f"✅ Successfully enrolled voiceprint for: {ENROLLMENT_STATE['speaker_id']}")
                        
                        # Reset state and re-engage filters
                        ENROLLMENT_STATE["is_active"] = False
                        ENROLLMENT_STATE["buffer"] = []
                        
                    continue # Skip the rest of the loop
                # --- END ENROLLMENT INTERCEPT ---
                
                # 1. Update rolling context buffer (Overlap-Discard)
                context_buffer = np.roll(context_buffer, -512)
                context_buffer[-512:] = audio_float32
                
                # 2. Step VAD state silently
                _ = vad_engine.process_chunk(audio_float32)
                
                # 3. Infer strictly once per frame IF speech is active
                if vad_engine.is_speech_active:
                    chunk_tensor = torch.from_numpy(context_buffer).cuda().view(1, 1, CONTEXT_SIZE).to(torch.float32)
                    cleaned_outputs = inference_engine.extract_voices(chunk_tensor, active_profiles)
                else:
                    # Output silence when VAD is negative
                    cleaned_outputs = {spk: torch.zeros(1, 1, CONTEXT_SIZE, device='cuda') for spk in active_profiles.keys()}

                for spk_id, cleaned_audio in cleaned_outputs.items():
                    if cleaned_audio is not None and cleaned_audio.numel() > 0:
                        # OLA logic
                        extracted_audio_full = cleaned_audio.squeeze().cpu()
                        current_chunk = extracted_audio_full[-1024:-32] 
                        
                        if current_chunk.numel() == (512 + overlap_len):
                            if prev_chunks[spk_id] is not None:
                                blended = (prev_chunks[spk_id][-overlap_len:] * fade_out) + (current_chunk[:overlap_len] * fade_in)
                                prev_chunks[spk_id][-overlap_len:] = blended
                                current_chunk[:overlap_len] = blended
                                session_audio[spk_id].append(prev_chunks[spk_id][:512])
                                prev_chunks[spk_id] = current_chunk
                            else:
                                initial_ramp = torch.linspace(0.0, 1.0, 512)
                                current_chunk = current_chunk.clone()
                                current_chunk[:512] *= initial_ramp
                                prev_chunks[spk_id] = current_chunk
                        
                        if vad_engine.is_speech_active and cleaned_audio.abs().max() > 0.01:
                            speech_duration[spk_id] += 0.032
                    
                    if not vad_engine.is_speech_active and speech_duration[spk_id] > 0:
                        logger.info(f"[SYSTEM] Speaker '{spk_id}' segment complete. Duration: {round(speech_duration[spk_id], 4)}s.")
                        
                        # --- NEW: WHISPER ASR TRANSLATION ---
                        try:
                            segment_audio_list = session_audio[spk_id][segment_start_idx[spk_id]:]
                            if segment_audio_list:
                                segment_tensor = torch.cat(segment_audio_list, dim=-1)
                                segment_np = segment_tensor.cpu().numpy().flatten()
                                
                                segments, _ = whisper_node.transcribe(segment_np, beam_size=1)
                                text = "".join([s.text for s in segments]).strip()
                                if text:
                                    print(f"\n🗣️ [{spk_id}]: {text}\n")
                        except Exception as e:
                            logger.error(f"Whisper Transcription Error: {e}")
                            
                        segment_start_idx[spk_id] = len(session_audio[spk_id])
                        speech_duration[spk_id] = 0.0
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("Closing connection. Saving session audio per speaker...")
        for spk_id in active_profiles.keys():
            if prev_chunks[spk_id] is not None:
                session_audio[spk_id].append(prev_chunks[spk_id][:512])
                
            if session_audio[spk_id]:
                final_audio = torch.cat(session_audio[spk_id], dim=-1).unsqueeze(0)
                max_amp = final_audio.abs().max()
                if max_amp > 0.0:
                    final_audio = (final_audio / max_amp) * 0.90
                    
                output_path = f"data/test_samples/extracted_{spk_id}.wav"
                torchaudio.save(output_path, final_audio, 16000)
                logger.info(f"Successfully saved audio for {spk_id} to {output_path} ({final_audio.shape[-1]/16000:.2f}s).")
        
        vad_engine.reset()
        logger.info("WebSocket endpoint cleanup complete.")
