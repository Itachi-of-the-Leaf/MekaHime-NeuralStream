import sys, os
sys.path.append(os.getcwd())
import numpy as np
import torchaudio
import torch
import os
import argparse
import sys

# Dynamic path injection for root execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.memory import RingBuffer

from scipy.io.wavfile import write

def capture_buffer(output_path: str, buffer_name: str = "/tmp/mekahime_audio_buffer.bin", duration_sec: int = 5):
    """
    Captures the last N seconds of audio from the Shared Memory Ring Buffer and saves to WAV.
    """
    sample_rate = 16000
    num_samples = sample_rate * duration_sec
    
    print(f"Connecting to Shared Memory: {buffer_name}...")
    try:
        # Connect to existing buffer
        default_size = 16000 * 60 
        buf = RingBuffer(name=buffer_name, size=default_size, create=False)
        
        # Get the current buffer state
        raw_audio = buf.buffer.copy().astype(np.float32)
        
        # Zero-check: Don't save if the buffer is empty
        if not np.any(raw_audio):
            print("⚠️ Buffer is purely zeroes. No signal found.")
            return

        write_ptr = buf.write_ptr
        
        # Read the last N samples
        if write_ptr >= num_samples:
            extracted = raw_audio[write_ptr - num_samples : write_ptr]
        else:
            # Wrapped part
            part1 = raw_audio[-(num_samples - write_ptr):]
            part2 = raw_audio[:write_ptr]
            extracted = np.concatenate([part1, part2])
            
        # Task 15: NaN Guard & Hard Clip
        extracted = np.nan_to_num(extracted)
        extracted = np.clip(extracted, -1.0, 1.0) # Absolute hard limit
        
        # Task 14: Save raw Float32 PCM without internal normalization
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Standard Float32 PCM rendering (values should be -1.0 to 1.0)
        write(output_path, sample_rate, extracted.astype(np.float32))
        print(f"DEBUG: Overwriting {output_path} with new data.")
        print(f"✅ Successfully saved {duration_sec}s of audio to {output_path}")
            
    except FileNotFoundError:
        print(f"❌ Error: Shared memory buffer '{buffer_name}' not found.")
    except Exception as e:
        print(f"❌ Error during capture: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture audio from MekaHime Shared Memory Buffer.")
    parser.add_argument("output_path", help="Path to save the extracted WAV file.")
    parser.add_argument("--buffer", default="/tmp/mekahime_audio_buffer.bin", help="Buffer file path.")
    parser.add_argument("--duration", type=int, default=5, help="Duration in seconds to capture.")
    args = parser.parse_args()
    
    capture_buffer(args.output_path, args.buffer, args.duration)
