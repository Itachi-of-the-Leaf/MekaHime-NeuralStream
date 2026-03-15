import asyncio
import websockets
import numpy as np
import time

async def run_test():
    uri = "ws://127.0.0.1:8000/ws/audio_stream"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("🟢 Connected to MekaHime Streaming Core.")
            
            # Simulate 3 seconds of "speech" (16kHz * 3 seconds = 48,000 samples)
            # We use a sine wave + noise to ensure the VAD triggers
            t = np.linspace(0, 3, 48000)
            speech_sim = 0.5 * np.sin(2 * np.pi * 440 * t) + np.random.normal(0, 0.1, 48000)
            
            # Convert to PCM16 (What the server expects)
            pcm16_data = (speech_sim * 32767).astype(np.int16).tobytes()
            
            # Send in strict 1024-byte chunks (512 samples)
            chunk_size = 1024
            print("🎤 Streaming 3 seconds of simulated speech...")
            
            for i in range(0, len(pcm16_data), chunk_size):
                chunk = pcm16_data[i:i + chunk_size]
                await websocket.send(chunk)
                # Mimic real-time (32ms per chunk)
                await asyncio.sleep(0.032) 
            
            print("✅ Stream complete. Sending close frame...")
            await websocket.close(code=1000) # Explicitly close the connection
            print("Done.")

    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_test())