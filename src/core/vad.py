import torch
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)

class VADEngine:
    """
    Stateful VAD engine using Silero VAD and a Pre-Roll buffer.
    Strictly processes 512-sample chunks at 16kHz.
    """
    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.chunk_size = 512  # Strict math constraint
        
        # Load Silero VAD model
        # Load Silero VAD model via standard torch hub
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      trust_repo=True)
        self.model = model
        self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks = utils
        
        # Initialize the stateful iterator
        self.vad_iterator = self.VADIterator(self.model, threshold=self.threshold, sampling_rate=self.sampling_rate)
        
        # Pre-roll Buffer: cached 96ms (3 chunks of 512 samples)
        self.pre_roll = deque(maxlen=3)
        self.is_speech_active = False

    def process_chunk(self, chunk: np.ndarray):
        """
        Processes a single 512-sample chunk.
        Returns a list of chunks (pre-roll + current) if speech starts, 
        or the current chunk if speech is active.
        """
        if len(chunk) != self.chunk_size:
            # Reshape or pad if necessary, but strictly we expect 512
            if len(chunk) > self.chunk_size:
                chunk = chunk[:self.chunk_size]
            else:
                padding = self.chunk_size - len(chunk)
                chunk = np.pad(chunk, (0, padding), 'constant')

        tensor_chunk = torch.from_numpy(chunk)
        
        # VADIterator maintains recurrent states internally
        # It returns a dictionary when speech starts or ends
        speech_dict = self.vad_iterator(tensor_chunk, return_seconds=True)
        
        if speech_dict:
            if 'start' in speech_dict:
                logger.info(f"Speech detected at {speech_dict['start']}s")
                self.is_speech_active = True
                # Flush pre-roll buffer
                buffer_to_flush = list(self.pre_roll)
                self.pre_roll.clear()
                # Return pre-roll + current chunk
                return buffer_to_flush + [chunk]
            
            if 'end' in speech_dict:
                logger.info(f"Speech end detected at {speech_dict['end']}s")
                self.is_speech_active = False
                # VADIterator handles model.reset_states() internally when speech ends or when manually called
                return [chunk]
        
        if self.is_speech_active:
            return [chunk]
        else:
            # Cache non-speech audio in pre-roll
            self.pre_roll.append(chunk)
            return []

    def reset(self):
        self.vad_iterator.reset_states()
        self.pre_roll.clear()
        self.is_speech_active = False
