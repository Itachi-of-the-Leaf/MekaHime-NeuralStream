#!/bin/bash
echo "Cleaning up ports..."
fuser -k 8000/tcp 2>/dev/null || true
sleep 1
echo "Starting Server..."
# Force mekahime_core environment activation in the subshell if needed
# However, the agent is already in the env.
/home/soham/miniconda3/envs/mekahime_core/bin/python3 main.py > server.log 2>&1 &
SERVER_PID=$!

# Wait for server to be fully warm
echo "Waiting for components to initialize..."
while ! grep -q "Dual-Engine Extraction fully initialized." server.log; do
  sleep 1
  # Safety break
  if ! ps -p $SERVER_PID > /dev/null; then
    echo "❌ Server crashed on startup. See server.log"
    exit 1
  fi
done

echo "Server ready. Running stream simulator..."
/home/soham/miniconda3/envs/mekahime_core/bin/python3 src/utils/stream_simulator.py data/test_samples/noisy_overlap_mix.wav

echo "Waiting for server to save OLA output..."
sleep 2

echo "Running Critic Evaluator..."
/home/soham/miniconda3/envs/mekahime_core/bin/python3 src/utils/auto_evaluator.py data/test_samples/extracted_output.wav

echo "Cleaning up..."
kill $SERVER_PID
rm -f /dev/shm/*mekahime* # Use shared memory cleanup if any
rm -f /tmp/mekahime_audio_buffer.bin # Also clean my file-backed buffer to ensure fresh run
