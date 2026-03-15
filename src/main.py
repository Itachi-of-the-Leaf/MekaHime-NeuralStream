import uvicorn
import signal
import sys
from src.api.server import app, audio_buffer
from src.core.memory import RingBuffer
import src.api.server as server_mod

# Configuration
BUFFER_NAME = "/tmp/mekahime_audio_buffer.bin"
BUFFER_SIZE = 16000 * 60  # 60 seconds of 16kHz audio

def main():
    # Initialize the pre-allocated RingBuffer
    # We use a large enough size to store a buffer of audio for processing
    buf = RingBuffer(name=BUFFER_NAME, size=BUFFER_SIZE, create=True)
    
    # Inject buffer into the API layer
    server_mod.audio_buffer = buf
    
    print(f"Initialized Shared Memory Ring Buffer: {BUFFER_NAME} ({BUFFER_SIZE} samples)")

    def signal_handler(sig, frame):
        print("\nShutting down...")
        buf.close()
        buf.unlink()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run Uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()
