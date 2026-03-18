#!/bin/bash

# TRT-LLM Loading Monitor
# Tracks model loading progress and identifies bottlenecks

CONTAINER="trtllm-multinode"
LOG_FILE="/tmp/trt-llm-monitor.log"
CHECK_INTERVAL=5

echo "🔍 TRT-LLM Loading Monitor - Starting..."
echo "Checking every ${CHECK_INTERVAL}s for progress signals..."
echo ""

while true; do
  echo "=== [$(date '+%H:%M:%S')] Status Check ===" >> $LOG_FILE

  # Check if container is running
  if ! docker exec $CONTAINER true 2>/dev/null; then
    echo "❌ Container not running"
    break
  fi

  # Check process count
  proc_count=$(docker exec $CONTAINER ps aux | grep -E "trtllm-serve|flashinfer" | grep -v grep | wc -l)
  echo "[PROCESS] $proc_count processes" | tee -a $LOG_FILE

  # Check memory usage
  mem=$(docker stats --no-stream $CONTAINER 2>/dev/null | tail -1 | awk '{print $4}')
  echo "[MEMORY] $mem" | tee -a $LOG_FILE

  # Check port listening
  port_listening=$(docker exec $CONTAINER netstat -tlnp 2>/dev/null | grep 8355 | wc -l)
  if [ $port_listening -gt 0 ]; then
    echo "🟢 [PORT] 8355 is LISTENING - API READY!"
    exit 0
  fi
  echo "[PORT] 8355 not listening yet"

  # Check for recent log activity (last 3 lines with timestamps)
  echo "[LOGS]" | tee -a $LOG_FILE
  docker logs $CONTAINER 2>&1 | tail -5 | grep -E "error|Error|ERROR|ready|Ready|READY|init|Init|Loading" | head -2 | sed 's/^/  /' | tee -a $LOG_FILE

  # Check for flashinfer compilation
  if docker logs $CONTAINER 2>&1 | tail -20 | grep -q "flashinfer.jit"; then
    echo "⏳ [STAGE] Compiling flashinfer kernels (slow on GB10)" | tee -a $LOG_FILE
  fi

  # Check for NCCL initialization
  if docker logs $CONTAINER 2>&1 | tail -20 | grep -q "start MpiSession"; then
    echo "⏳ [STAGE] NCCL/MPI distributed setup" | tee -a $LOG_FILE
  fi

  # Check for weight loading
  if docker logs $CONTAINER 2>&1 | tail -20 | grep -qE "Loading|load"; then
    echo "⏳ [STAGE] Loading model weights" | tee -a $LOG_FILE
  fi

  echo "" | tee -a $LOG_FILE
  sleep $CHECK_INTERVAL
done
