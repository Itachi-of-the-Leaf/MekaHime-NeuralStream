import sys, os
sys.path.append(os.getcwd())
import asyncio
import websockets
import torchaudio
import torch
import time
import os
import argparse
import sys

# Dynamic path injection for root execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

async def stream_audio(audio_path: str, uri: str = "ws://127.0.0.1:8000/ws/audio_stream"):
    """
    Simulates a live 16kHz microphone stream by sending 512-sample chunks at 32ms intervals.
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file {audio_path} not found.")
        return

    # Load audio
    audio, sr = torchaudio.load(audio_path)
    
    # Ensure 16kHz
    if sr != 16000:
        print(f"Resampling from {sr}Hz to 16000Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = resampler(audio)

    # To mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Convert to PCM16
    audio_pcm16 = (audio[0].numpy() * 32767).astype('int16')
    bytes_data = audio_pcm16.tobytes()

    chunk_size_bytes = 1024 # 512 samples * 2 bytes/sample
    
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("🟢 Connected. Starting stream...")
            
            start_time = time.time()
            chunks_sent = 0
            
            for i in range(0, len(bytes_data), chunk_size_bytes):
                chunk = bytes_data[i:i + chunk_size_bytes]
                
                # Check for last partial chunk
                if len(chunk) < chunk_size_bytes:
                    break
                    
                await websocket.send(chunk)
                chunks_sent += 1
                
                # Maintain exactly 100ms cadence (Slower than real-time for stable eval)
                # Target time for next chunk
                target_time = start_time + (chunks_sent * 0.100)
                wait_time = target_time - time.time()
                
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
            print(f"✅ Finished streaming {chunks_sent} chunks.")
            await asyncio.sleep(5) # Give the server time to clear the neural queue
            
    except Exception as e:
        print(f"❌ WebSocket error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate real-time WebSocket audio streaming.")
    parser.add_argument("audio_path", help="Path to the WAV file to stream.")
    parser.add_argument("--uri", default="ws://127.0.0.1:8000/ws/audio_stream", help="WebSocket URI.")
    args = parser.parse_args()
    
    asyncio.run(stream_audio(args.audio_path, args.uri))
