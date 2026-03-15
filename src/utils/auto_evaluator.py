import sys, os
sys.path.append(os.getcwd())
import torch
import torchaudio
import numpy as np
import argparse
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from src.core.speaker_db import SpeakerDB

def evaluate_audio(audio_path: str, target_speaker: str = "primary_user"):
    if not os.path.exists(audio_path):
        print("CRITIC_SCORE: 0.0 | ERROR: File not found.")
        return

    # Load audio
    audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
    
    audio_np = audio.numpy().flatten()
    
    # METRIC 1: Energy & Silence
    rms_energy = np.sqrt(np.mean(audio_np**2))
    if rms_energy < 0.001:
        print("CRITIC_SCORE: 0.0 | ERROR: Total Silence detected. Threshold likely too high.")
        return
        
    # METRIC 2: Discontinuity (Pops / Walkie-Talkie Squelch)
    # Check for massive jumps between adjacent samples
    deltas = np.abs(np.diff(audio_np))
    max_delta = np.max(deltas)
    if max_delta > 0.5:
        print(f"CRITIC_SCORE: 1.0 | ERROR: Severe audio pop detected (Max Delta: {max_delta:.2f}). Soft-fade failed.")
        return

    # METRIC 3: Speaker Identity
    print("Loading TitaNet for identity verification...")
    speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large").cuda().eval()
    
    # STRIP DIGITAL SILENCE: TitaNet embeddings are corrupted by large blocks of pure zeros.
    active_speech_mask = audio.abs() > 0.001
    clean_audio_for_eval = audio[active_speech_mask].unsqueeze(0)
    
    if clean_audio_for_eval.numel() < 16000: # Less than 1 second of audio passed
        print("CRITIC_SCORE: 1.0 | ERROR: Output is almost entirely silence. Threshold is too high.")
        return
        
    db = SpeakerDB()
    
    target_match = db.collection.get(ids=[target_speaker], include=['embeddings'])
    embeddings = target_match.get('embeddings')
    if embeddings is None or len(embeddings) == 0:
        print("ERROR: Target not found in DB.")
        return
        
    target_embedding = torch.tensor(target_match['embeddings'][0]).cuda().unsqueeze(0)
    
    # Score only the active speech
    audio_len = torch.tensor([clean_audio_for_eval.shape[1]]).cuda()
    with torch.no_grad():
        _, output_emb = speaker_model.forward(input_signal=clean_audio_for_eval.cuda(), input_signal_length=audio_len)
        
    similarity = torch.nn.functional.cosine_similarity(output_emb, target_embedding).item()
    
    # Final Report
    print(f"\n--- PIPELINE REPORT CARD ---")
    print(f"RMS Energy (Volume): {rms_energy:.4f}")
    print(f"Max Transient (Smoothness): {max_delta:.4f}")
    print(f"Target Isolation Score: {similarity:.4f} (Ideal > 0.50 for Overlap BSS)")
    
    if similarity > 0.50:
        print(f"CRITIC_SCORE: 5.0 | SUCCESS: High-quality target extraction achieved.")
    else:
        print(f"CRITIC_SCORE: {similarity * 10:.1f} | WARNING: Target isolation poor.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    args = parser.parse_args()
    evaluate_audio(args.audio_path)
