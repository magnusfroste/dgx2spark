#!/bin/bash

# TRT-LLM Quick Test with Qwen2.5-Coder-7B
# Purpose: Test MPI & multi-node setup BEFORE loading heavy Nemotron
# Model: Qwen2.5-Coder-7B-Instruct (7B, ~14GB, loads in ~2 minutes)
# Framework: TensorRT-LLM 1.0.0rc3

set -e

# Configuration
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
DGX1_IP="10.0.0.1"
DGX2_IP="10.0.0.2"
CONTAINER_NAME="trtllm-qwen-test"
API_PORT="8005"
HF_TOKEN="${HF_TOKEN:-}"
TP_SIZE=2        # Minimal TP for quick test

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  TRT-LLM Multi-Node TEST                   ║${NC}"
echo -e "${BLUE}║  Qwen2.5-Coder-7B (Quick Verification)     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Purpose:${NC} Verify MPI & multi-node setup works"
echo "  • Before loading massive Nemotron-120B"
echo "  • Quick model load (2 min vs 10+ min for Nemotron)"
echo "  • Test TP=2 (4 GPUs per node)"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Model:        $MODEL (7B)"
echo "  Nodes:        DGX1 ($DGX1_IP) + DGX2 ($DGX2_IP)"
echo "  TP Size:      $TP_SIZE (4 GPUs per node)"
echo "  Container:    $CONTAINER_NAME"
echo "  API Port:     $API_PORT"
echo ""

# Pre-flight checks
echo -e "${YELLOW}[1/6] Pre-flight checks...${NC}"

if ! ping -c 1 $DGX2_IP > /dev/null 2>&1; then
    echo -e "${RED}✗ Cannot ping DGX2 at $DGX2_IP${NC}"
    exit 1
fi
echo -e "${GREEN}✓ DGX2 reachable${NC}"

if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}⚠  HF_TOKEN not set (required for Qwen model)${NC}"
    echo "   Set with: export HF_TOKEN=hf_..."
    exit 1
fi
echo -e "${GREEN}✓ HF_TOKEN configured${NC}"

# Cleanup
echo -e "${YELLOW}[2/6] Cleaning up old containers...${NC}"
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$DGX2_IP "docker stop $CONTAINER_NAME 2>/dev/null || true; docker rm $CONTAINER_NAME 2>/dev/null || true" 2>/dev/null || true
sleep 2
echo -e "${GREEN}✓ Cleaned${NC}"

# Create hostfile
echo -e "${YELLOW}[3/6] Creating MPI hostfile...${NC}"
cat > /tmp/openmpi-hostfile-test << EOF
$DGX1_IP slots=1
$DGX2_IP slots=1
EOF
echo -e "${GREEN}✓ Hostfile created${NC}"
echo "Contents:"
cat /tmp/openmpi-hostfile-test | sed 's/^/  /'

# Start containers
echo -e "${YELLOW}[4/6] Starting containers on both nodes...${NC}"

docker run \
    --name $CONTAINER_NAME \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /tmp/openmpi-hostfile-test:/etc/openmpi-hostfile \
    -e HF_TOKEN="$HF_TOKEN" \
    -e NCCL_DEBUG=INFO \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_SOCKET_IFNAME=enp1s0f0np0,enp1s0f1np1 \
    -e NCCL_P2P_DISABLE=1 \
    -e OMPI_ALLOW_RUN_AS_ROOT=1 \
    -p $API_PORT:8000 \
    --ipc=host \
    --shm-size 32g \
    -d \
    nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc3 \
    sleep infinity

echo -e "${GREEN}✓ DGX1 container started${NC}"

ssh -o StrictHostKeyChecking=no root@$DGX2_IP << REMOTE_CMD
docker run \
    --name $CONTAINER_NAME \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /tmp/openmpi-hostfile-test:/etc/openmpi-hostfile \
    -e HF_TOKEN="$HF_TOKEN" \
    -e NCCL_DEBUG=INFO \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_SOCKET_IFNAME=enp1s0f0np0,enp1s0f1np1 \
    -e NCCL_P2P_DISABLE=1 \
    -e OMPI_ALLOW_RUN_AS_ROOT=1 \
    --ipc=host \
    --shm-size 32g \
    -d \
    nccr.io/nvidia/tensorrt-llm/release:1.0.0rc3 \
    sleep infinity
REMOTE_CMD

echo -e "${GREEN}✓ DGX2 container started${NC}"
sleep 3

# Copy hostfile to both containers
docker cp /tmp/openmpi-hostfile-test $CONTAINER_NAME:/etc/openmpi-hostfile
ssh -o StrictHostKeyChecking=no root@$DGX2_IP "docker cp /etc/openmpi-hostfile-test $CONTAINER_NAME:/etc/openmpi-hostfile" 2>/dev/null || true

# Launch inference
echo -e "${YELLOW}[5/6] Launching TRT-LLM with Qwen2.5-Coder...${NC}"
echo "      (First run will compile engine, ~2-3 minutes)"
echo ""

docker exec \
    -e HF_TOKEN="$HF_TOKEN" \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_SOCKET_IFNAME=enp1s0f0np0,enp1s0f1np1 \
    -e NCCL_P2P_DISABLE=1 \
    $CONTAINER_NAME bash -c '
    mpirun \
        -x HF_TOKEN \
        -x NCCL_IB_DISABLE=1 \
        -x NCCL_SOCKET_IFNAME=enp1s0f0np0,enp1s0f1np1 \
        -x NCCL_P2P_DISABLE=1 \
        -hostfile /etc/openmpi-hostfile \
        -np 2 \
        trtllm-llmapi-launch trtllm-serve '"'"'Qwen/Qwen2.5-Coder-7B-Instruct'"'"' \
            --tp_size 2 \
            --backend pytorch \
            --max_num_tokens 8192 \
            --max_batch_size 2 \
            --port 8000
' &

INFERENCE_PID=$!

# Wait for API ready
echo -e "${YELLOW}[6/6] Waiting for API to be ready...${NC}"
echo "      (Expect: startup + model download + engine build)"
echo ""

TIMEOUT=300
ELAPSED=0

while [ $ELAPSED -lt $TIMEOUT ]; do
    if curl -s http://localhost:$API_PORT/v1/models 2>/dev/null | grep -q "object"; then
        echo ""
        echo -e "${GREEN}✓ API is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 5
    ELAPSED=$((ELAPSED + 5))

    if [ $((ELAPSED % 60)) -eq 0 ]; then
        echo " ($ELAPSED/$TIMEOUT seconds)"
    fi
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo ""
    echo -e "${RED}✗ Timeout waiting for API${NC}"
    echo ""
    echo -e "${YELLOW}Debugging:${NC}"
    echo "  Check DGX1 logs:"
    echo "    docker logs $CONTAINER_NAME | tail -100"
    echo ""
    echo "  Check DGX2 logs:"
    echo "    ssh root@$DGX2_IP docker logs $CONTAINER_NAME | tail -100"
    echo ""
    echo "  Check MPI errors:"
    echo "    docker logs $CONTAINER_NAME | grep -i 'mpi\\|error\\|fail'"
    exit 1
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  TEST SUCCESSFUL!                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Next: Test inference${NC}"
echo "  curl -X POST http://localhost:$API_PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{"
echo "      \"model\": \"$MODEL\","
echo "      \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2?\"}],"
echo "      \"max_tokens\": 100"
echo "    }'"
echo ""
echo -e "${BLUE}If test succeeds:${NC}"
echo "  ✓ MPI works between nodes"
echo "  ✓ NCCL communication OK"
echo "  ✓ Model loading works"
echo "  ✓ Ready for Nemotron test"
echo ""
echo -e "${BLUE}Logs:${NC}"
echo "  DGX1: docker logs -f $CONTAINER_NAME | grep -E 'mpi|rank|ready'"
echo "  DGX2: ssh root@$DGX2_IP docker logs -f $CONTAINER_NAME | grep -E 'mpi|rank|ready'"
echo ""
echo -e "${BLUE}Stop:${NC}"
echo "  docker stop $CONTAINER_NAME"
echo "  ssh root@$DGX2_IP docker stop $CONTAINER_NAME"
