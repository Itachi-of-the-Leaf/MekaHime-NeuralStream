# MekaHime Streaming Core

Low-latency target speech extraction pipeline.

## Environment Setup

1. Activate the environment:
   ```bash
   conda activate mekahime_core
   ```

2. Set the python path (CRITICAL for imports):
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

## Running the Pipeline

1. **Start the Server**:
   ```bash
   PYTHONPATH=. uvicorn src.api.server:app --reload
   ```

2. **Run the Stream Simulator**:
   ```bash
   PYTHONPATH=. python3 src/utils/stream_simulator.py data/test_samples/noisy_overlap_mix.wav
   ```

3. **Verify Output**:
   ```bash
   PYTHONPATH=. python3 src/utils/verify_output.py output.wav
   ```
