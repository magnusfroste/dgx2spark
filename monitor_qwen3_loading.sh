#!/bin/bash
START_TIME=$(date +%s)
echo "Qwen3-235B loading started at $(date)"
echo "Expected time: 15-25 minutes for ARM64 kernel JIT compilation"
echo ""

# Poll every 30 seconds for up to 45 minutes
for i in {1..90}; do
    ELAPSED=$(($(date +%s) - START_TIME))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))
    
    if curl -s http://localhost:8355/v1/models > /dev/null 2>&1; then
        echo "✓ API READY at $(date) (${MINUTES}m ${SECONDS}s elapsed)"
        curl -s http://localhost:8355/v1/models | python3 -m json.tool 2>/dev/null | head -20
        echo ""
        echo "🟢 Ready to chat at http://localhost:7860"
        break
    else
        if [ $((i % 2)) -eq 0 ]; then
            echo "[${MINUTES}m ${SECONDS}s] Still loading..."
        fi
    fi
    sleep 30
done

if [ $i -eq 90 ]; then
    echo "Timeout after 45 minutes. Checking logs..."
    docker logs --tail 50 trtllm-multinode | tail -20
fi
