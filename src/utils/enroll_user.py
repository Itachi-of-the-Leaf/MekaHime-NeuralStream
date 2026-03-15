import torch
import torchaudio
import numpy as np
import os
import argparse
import sys

# Dynamic path injection for root execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from src.core.speaker_db import SpeakerDB

def enroll(audio_path: str, speaker_id: str = "primary_user"):
    """
    Enroll a user by extracting TitaNet-L embeddings from a WAV file.
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file {audio_path} not found.")
        return

    # Initialize SpeakerDB
    db = SpeakerDB()

    # Load TitaNet-L model
    print("Loading NeMo TitaNet-L model...")
    model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large").cuda().eval()

    # Load audio
    audio, sr = torchaudio.load(audio_path)
    
    # Ensure 16kHz
    if sr != 16000:
        print(f"Resampling from {sr}Hz to 16000Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = resampler(audio)
    
    # To mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Extract embedding
    print("Extracting embedding...")
    with torch.no_grad():
        # model.get_embedding expects path or manifest, or use forward
        # For simplicity with raw tensor:
        audio_len = torch.tensor([audio.shape[1]]).cuda()
        logits, embedding = model.forward(input_signal=audio.cuda(), input_signal_length=audio_len)
        # embedding shape: [Batch, Dim]
        embedding_np = embedding.cpu().numpy().flatten()

    # Normalize embedding (TitaNet typically returns normalized, but good to be sure for cosine)
    embedding_np = embedding_np / np.linalg.norm(embedding_np)

    # Upsert to SpeakerDB
    print(f"Upserting voiceprint for '{speaker_id}' to SpeakerDB...")
    db.add_voiceprint(speaker_id, embedding_np)
    print("Enrollment complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a user into MekaHime SpeakerDB.")
    parser.add_argument("audio_path", help="Path to the user's reference WAV file.")
    parser.add_argument("--id", default="primary_user", help="Speaker ID to store (default: primary_user).")
    args = parser.parse_args()
    
    enroll(args.audio_path, args.id)
