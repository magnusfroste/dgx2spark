# MPI Protocol Guide for DGX Spark Multi-Node Setup

## What is MPI?

**MPI (Message Passing Interface)** = protokoll för att få två DGX-noder att arbeta tillsammans som EN cluster.

Think of it as: **SSH-koppling fast för GPU-koordinering**

---

## How It Works (Simplified)

```
DGX1 (10.0.0.1)          DGX2 (10.0.0.2)
┌─────────────────┐      ┌─────────────────┐
│ TRT-LLM Rank 0  │◄────►│ TRT-LLM Rank 1  │
│ (Master/API)    │      │ (Worker)        │
│                 │      │                 │
│ GPU 0-7 (TP)    │      │ GPU 0-7 (TP)    │
└────────┬────────┘      └────────┬────────┘
         │                        │
         ├───────── MPI ──────────┤
         │      (QSFP Cable)      │
         │  (IPC for in-process)  │
         └────────────────────────┘
```

---

## The Three Communication Layers

### 1. **Within a Node (DGX1 or DGX2)**
```
GPU 0 ◄─► GPU 1 ◄─► GPU 2 ... GPU 7
       │
    NVLink (very fast, same PCIe bus)
    Or: Shared memory (IPC)
```
- **What:** Tensor Parallel (TP=8) splits model across 8 GPUs
- **Speed:** ~200GB/s (NVLink)
- **How:** PyTorch handles it, very fast

### 2. **Between Nodes (DGX1 ↔ DGX2)**
```
DGX1 GPUs ◄─► QSFP Cable ◄─► DGX2 GPUs
               10Gbps
```
- **What:** Pipeline Parallel or collective operations (NCCL)
- **Speed:** 10Gbps (QSFP) = ~1.2 GB/s (much slower than NVLink!)
- **How:** NCCL over TCP/IP (InfiniBand not available on GB10)

### 3. **MPI Orchestration (Control Plane)**
```
mpirun on DGX1
   │
   ├─► Rank 0 starts on DGX1
   │     └─► SSH to DGX2 port 2233
   │
   └─► Rank 1 starts on DGX2
         └─► Both sync via MPI
```
- **What:** Tells both nodes when to start/sync
- **Speed:** Not critical (control, not data)
- **How:** SSH + OpenMPI

---

## TRT-LLM Multi-Node Flow

### Step 1: Setup (Before Inference)
```bash
# On DGX1 only:
mpirun -np 2 -hostfile /etc/openmpi-hostfile \
  trtllm-llmapi-launch trtllm-serve \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --tp_size 2 \
    --backend pytorch
```

**What happens:**
1. `mpirun` starts 2 processes (np=2)
2. Process 0 (Rank 0) on DGX1
3. Process 1 (Rank 1) on DGX2
4. They connect via SSH (hostfile tells where)
5. Both load model partitions

### Step 2: Model Loading
```
DGX1 Rank 0:
  ├─ Load 1/2 of model (Layers 1-24)
  ├─ Allocate 30GB GPU memory
  └─ Wait for Rank 1

DGX2 Rank 1:
  ├─ Load 1/2 of model (Layers 25-48)
  ├─ Allocate 30GB GPU memory
  └─ Signal ready to Rank 0
```

### Step 3: Inference (Request arrives)
```
User: POST /v1/chat/completions on DGX1:8000
       │
       ▼
Rank 0 (DGX1):
  ├─ Tokenize input
  ├─ Process Layers 1-24 locally
  ├─ Send intermediate tensor to Rank 1 via QSFP
  │
  └─► Rank 1 (DGX2):
       ├─ Receive tensor from Rank 0
       ├─ Process Layers 25-48 locally
       └─ Send output tensor back to Rank 0

Rank 0 (DGX1):
  ├─ Receive result from Rank 1
  ├─ Detokenize
  └─ Return JSON response to user
```

---

## Configuration for DGX Spark

### NCCL Settings (GPU Communication)
```bash
export NCCL_DEBUG=INFO          # Show NCCL messages (debug)
export NCCL_IB_DISABLE=1        # Don't use InfiniBand (GB10 doesn't have it)
export NCCL_SOCKET_IFNAME=enp1s0f0np0,enp1s0f1np1  # Use QSFP interfaces
export NCCL_P2P_DISABLE=1       # Disable P2P (unstable on GB10)
export NCCL_NET_GDR_LEVEL=2     # GPU Direct RDMA (if supported)
```

### OpenMPI Settings (Process Coordination)
```bash
export OMPI_MCA_btl_tcp_if_include=enp1s0f0np0,enp1s0f1np1
export OMPI_MCA_orte_default_hostfile=/etc/openmpi-hostfile
export OMPI_ALLOW_RUN_AS_ROOT=1
```

### Hostfile Format
```
# /etc/openmpi-hostfile
10.0.0.1 slots=1
10.0.0.2 slots=1
```
- `slots=1` = one MPI process per node (Rank 0, Rank 1)
- Not `slots=8` (that would try to run 8 ranks per node, wrong!)

---

## Common Issues & Debugging

### Issue 1: "Cannot reach DGX2"
```bash
# Check:
ping 10.0.0.2
ssh root@10.0.0.2 "nvidia-smi"

# In container:
docker exec trtllm-container ssh -p 2233 root@10.0.0.2 "echo OK"
```

### Issue 2: NCCL Timeout
```
Error: NCCL operation timed out
```
**Reason:** Model too big, inter-node traffic congested
**Fix:**
- Reduce batch size (`--max_batch_size 1`)
- Use hybrid TP+PP instead of pure TP
- Check QSFP cable connection

### Issue 3: "MPI_Initialized error"
```
Error: MPI_Initialized returned an error code
```
**Reason:** CUDA context conflict, or SSH key missing
**Fix:**
```bash
# Ensure SSH keys exist in container
docker exec trtllm-container cat ~/.ssh/id_rsa

# Or manually create:
docker exec trtllm-container ssh-keygen -A
```

### Issue 4: "fatal error: cuda.h: No such file or directory"
```
This is the 1.2.0rc6 bug! Don't use it.
Use 1.0.0rc3 instead.
```

---

## Monitoring MPI

### During Setup
```bash
# Watch DGX1 logs:
docker logs -f trtllm-container | grep -E "mpi|rank|MPI"

# Expected output:
# mpi_rank: 0
# mpi_rank: 1
# tllm_mpi_size: 2
```

### During Inference
```bash
# Check inter-node traffic:
sar -n EDEV 1 | grep enp1s0f

# Should show non-zero TX/RX during inference
```

### NCCL Debug
```bash
# Enable detailed NCCL logging:
export NCCL_DEBUG=TRACE

# This will show:
# [NCCL] ncclAllReduce called from rank 0
# [NCCL] Launch kernel on CUDA stream
```

---

## Performance Expectations

### Qwen2.5-Coder-7B
```
Single-node (TP=8, DGX1 only):
  ├─ Throughput: ~100-150 tok/s
  ├─ Memory: 14GB
  └─ No inter-node overhead

Multi-node (TP=2, PP=2 across nodes):
  ├─ Throughput: ~80-100 tok/s (20% loss due to QSFP)
  ├─ Memory: 7GB per node
  └─ 10Gbps QSFP overhead

Multi-node (Pure TP=8):
  ├─ Throughput: ~50-80 tok/s (heavy QSFP traffic)
  ├─ Memory: 30GB per node
  └─ QSFP bottleneck at 10Gbps
```

### Nemotron-3-Super-120B (NVFP4, 60GB)
```
Single-node (Won't fit):
  ├─ Memory needed: 60GB
  └─ Not possible on single 8-GPU node

Multi-node (TP=2, PP=2):
  ├─ Throughput: ~20-30 tok/s (estimated)
  ├─ Memory: 30GB per node
  └─ Hybrid approach more stable

Multi-node (Pure TP=8, risky):
  ├─ Throughput: ~15-25 tok/s (NCCL overhead)
  ├─ Memory: 60GB per node (tight!)
  └─ High risk of GB10 multi-node crash
```

---

## Summary

| Component | What | Speed | Notes |
|-----------|------|-------|-------|
| **NVLink (within node)** | GPU↔GPU | 200GB/s | Automatic, very fast |
| **QSFP (between nodes)** | DGX↔DGX | 1.2GB/s | 10Gbps cable, bottleneck |
| **MPI (orchestration)** | Sync/startup | ~1ms | SSH-based, slow but OK for setup |
| **NCCL (collective ops)** | All-reduce, broadcast | 1.2GB/s | Over QSFP, limited by cable |

**Key Insight:** Multi-node throughput is limited by QSFP cable (10Gbps = 1.2GB/s).
- **Solution:** Use Hybrid TP+PP to minimize inter-node communication
- **Risk:** Pure TP means constant QSFP traffic → potential crashes on GB10

---

## Checklist Before Running Multi-Node

- [ ] QSFP cable physically connected
- [ ] Ping between nodes works (`ping 10.0.0.2`)
- [ ] SSH keys exchanged (SSH without password)
- [ ] Hostfile correct (`10.0.0.1 slots=1`, `10.0.0.2 slots=1`)
- [ ] NCCL envvars set (SOCKET_IFNAME, IB_DISABLE)
- [ ] Container has ssh-keygen run
- [ ] Model size fits (2 nodes × 8 GPUs × available VRAM)
- [ ] Using correct TRT-LLM version (1.0.0rc3, NOT rc6)
