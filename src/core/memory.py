import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class RingBuffer:
    """
    Stable Ring Buffer using file-backed np.memmap.
    Ensures persistence across process lifetimes and avoids SharedMemory unlinking issues.
    """
    def __init__(self, name: str, size: int, create: bool = True):
        self.filename = name
        self.size = size
        self.dtype = np.float32
        self.item_size = np.dtype(self.dtype).itemsize
        # Task 13: Extra 8 bytes for shared write_ptr (int64)
        self.byte_size = (self.size * self.item_size) + 8

        if create:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            # Create/Initialize the file
            if not os.path.exists(self.filename) or os.path.getsize(self.filename) != self.byte_size:
                with open(self.filename, 'wb') as f:
                    f.write(b'\x00' * self.byte_size)
                logger.info(f"Created fresh memmap file: {self.filename}")
            
            self.mm = np.memmap(self.filename, dtype='uint8', mode='r+', shape=(self.byte_size,))
            # Initialize pointer to 0 if we just created it or if requested
            self._shared_ptr = np.ndarray(1, dtype=np.int64, buffer=self.mm[:8])
            # Note: We don't always reset to 0 on create=True to allow server restarts to pick up where they left off
            # unless the file was just created.
        else:
            if not os.path.exists(self.filename):
                raise FileNotFoundError(f"Memmap file {self.filename} not found.")
            self.mm = np.memmap(self.filename, dtype='uint8', mode='r+', shape=(self.byte_size,))
            self._shared_ptr = np.ndarray(1, dtype=np.int64, buffer=self.mm[:8])

        # Map the audio buffer part
        self.buffer = np.ndarray(self.size, dtype=self.dtype, buffer=self.mm[8:])

    @property
    def write_ptr(self):
        return int(self._shared_ptr[0])

    @write_ptr.setter
    def write_ptr(self, value):
        self._shared_ptr[0] = int(value)

    def write(self, data: np.ndarray):
        """
        Writes data to the ring buffer using modulo pointer arithmetic.
        """
        n = data.shape[0]
        current_ptr = self.write_ptr
        
        if n > self.size:
            data = data[-self.size:]
            n = self.size

        end_ptr = (current_ptr + n) % self.size

        if end_ptr > current_ptr:
            self.buffer[current_ptr:end_ptr] = data
        else:
            # Wrap around
            first_part = self.size - current_ptr
            self.buffer[current_ptr:] = data[:first_part]
            self.buffer[:end_ptr] = data[first_part:]

        self.write_ptr = end_ptr
        # Ensure changes are flushed to disk/OS cache
        self.mm.flush()

    def close(self):
        # Memmap doesn't have a close(), but we can flush
        if hasattr(self, 'mm'):
            self.mm.flush()
            del self.mm

    def unlink(self):
        """
        Deletes the underlying file.
        """
        try:
            if os.path.exists(self.filename):
                os.remove(self.filename)
        except Exception as e:
            logger.error(f"Failed to unlink memmap file: {e}")
