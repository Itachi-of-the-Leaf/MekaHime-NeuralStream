import asyncio
import websockets
import sounddevice as sd
import numpy as np
import urllib.request
import time
import sys

# Audio hardware configuration
FORMAT = np.int16
CHANNELS = 1
RATE = 16000
CHUNK = 1024  # Increased for stability in async queue

# Silence detection configuration
SILENCE_THRESHOLD = 300
SILENCE_DURATION = 3.0
MAX_SILENT_CHUNKS = int((SILENCE_DURATION * RATE) / CHUNK)

ENROLLMENT_DURATION = 6.0 # seconds

async def run_live_enroll_and_test():
    http_url = "http://127.0.0.1:8000/enroll/live_test_user"
    ws_uri = "ws://127.0.0.1:8000/ws/audio_stream"
    
    print("🚀 Initializing Live Enrollment & Test...")
    
    # Stage 1: The Trigger (HTTP POST)
    try:
        print(f"📡 Stage 1: Sending Trigger to {http_url}")
        # Using urllib.request for POST to avoid requests dependency
        req = urllib.request.Request(http_url, data=b"", method="POST")
        with urllib.request.urlopen(req) as response:
            if response.getcode() != 200:
                print(f"❌ HTTP Trigger Failed: Status {response.getcode()}")
                return
            print("✅ Stage 1: HTTP Trigger Success!")
    except Exception as e:
        print(f"❌ HTTP Trigger Failed: {e}")
        return

    # Queue to bridge sounddevice callback and async loop
    audio_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"⚠️ Mic Status: {status}")
        # Non-blocking callback: pass data to async queue
        loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy())

    # Open WebSocket for Stage 2 & 3
    try:
        async with websockets.connect(ws_uri) as websocket:
            print("🟢 Stage 2: WebSocket Connected.")
            
            # Start mic stream
            try:
                device_info = sd.query_devices(sd.default.device[0], 'input')
                print(f"🎙️ Using Audio Device: {device_info['name']}")
            except Exception:
                print("⚠️ Could not query default audio device.")

            try:
                with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype=FORMAT, 
                                 blocksize=CHUNK, callback=audio_callback):
                    
                    print(f"\n🎤 Stage 2: Speak clearly for {ENROLLMENT_DURATION} seconds to register your voiceprint...")
                    
                    start_time = time.time()
                    is_enrolling = True
                    has_spoken = False
                    silent_chunks = 0
                    stage3_start_time = None
                    
                    while True:
                        # Fetch data from queue
                        data = await audio_queue.get()
                        audio_np = data.flatten()
                        
                        # Send to server via WebSocket
                        await websocket.send(audio_np.tobytes())
                        
                        elapsed = time.time() - start_time
                        
                        # Transition from enrollment to testing
                        if is_enrolling and elapsed >= ENROLLMENT_DURATION:
                            is_enrolling = False
                            stage3_start_time = time.time()
                            print("\n✅ Stage 2: Enrollment assumed complete!")
                            print("🔊 Stage 3: Now, play Zeek's audio and talk over it!")
                        
                        # Stage 4: Silence/Timeout Kill-Switch (monitors activity after enrollment)
                        if not is_enrolling:
                            rms = np.sqrt(np.mean(audio_np.astype(np.float32)**2))
                            if rms > SILENCE_THRESHOLD:
                                has_spoken = True
                                silent_chunks = 0
                            elif has_spoken:
                                silent_chunks += 1
                            
                            # Kill switch: 3s silence OR 15s hard limit
                            if (has_spoken and silent_chunks > MAX_SILENT_CHUNKS) or (time.time() - stage3_start_time > 15.0):
                                print("\n🛑 15-second limit reached or silence detected. Stopping stream...")
                                break
            except sd.PortAudioError as e:
                print(f"❌ Hardware Audio Failed: {e}")
                return
            except Exception as e:
                print(f"❌ Hardware Audio Failed: {e}")
                return

            # Graceful closing triggers server's post-processing
            await websocket.close(code=1000)
            print("✅ Graceful Exit: WebSocket closed. Server processing triggered.")
            
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ WebSocket Failed: Connection closed unexpectedly ({e.code})")
    except Exception as e:
        print(f"❌ WebSocket Failed: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run_live_enroll_and_test())
    except KeyboardInterrupt:
        print("\n👋 Native stop by user.")
    except Exception as e:
        print(f"💥 Script Crash: {e}")
