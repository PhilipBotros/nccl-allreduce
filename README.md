# nccl-allreduce

Code and measurements for the blog post ["All-reduce by hand: chasing NCCL from T4s to B200s"](https://philipbotros.github.io/2026/11/18/all-reduce-by-hand.html).

Everything here runs inside one machine. The benchmarks compare hand-written all-reduce implementations against NCCL on the same harness.

## Layout

| Path | What it is |
|---|---|
| `cuda/allreduce.cu` | Two-GPU benchmark: copy then add, remote read in the add kernel, a two-GPU ring, and NCCL |
| `cuda/ring.cu` | P-GPU benchmark in one process: direct (copy all, read all), host-driven rings, the fused ring, and NCCL |
| `cuda/ring_fused.cuh` | The fused ring kernels: one kernel per GPU, per-block flags, and the LL protocol for small chunks |
| `cuda/ring_mp.cu` | One process per GPU (fork, CUDA IPC, `ncclCommInitRank`): fused ring, direct read, size dispatch, and NCCL |
| `cuda/plot.py`, `cuda/plot_ring.py` | Quick plots from the CSVs |
| `results/` | The CSVs behind the post's figures, plus NCCL debug output where it was captured |

## Building

You need the CUDA toolkit and an NCCL install. Point `NCCL` at a directory with `include/nccl.h` and `lib/libnccl.so.2`. The NCCL from the `nvidia-nccl` pip package works.

```bash
NCCL=/path/to/nccl
nvcc -O2 -arch=native -I$NCCL/include -L$NCCL/lib -l:libnccl.so.2 -o allreduce cuda/allreduce.cu
nvcc -O2 -arch=native -I$NCCL/include -L$NCCL/lib -l:libnccl.so.2 -o ring      cuda/ring.cu
nvcc -O2 -arch=native -I$NCCL/include -L$NCCL/lib -l:libnccl.so.2 -o ring_mp   cuda/ring_mp.cu
export LD_LIBRARY_PATH=$NCCL/lib
```

## Running

Each binary prints a CSV to stdout with one row per implementation and message size: median, p10 and p90 time in microseconds, plus algorithm and bus bandwidth.

```bash
# two GPUs, with and without peer access
./allreduce p2p   > results/my_2gpu_p2p.csv
./allreduce nop2p > results/my_2gpu_nop2p.csv

# P GPUs in one process: mode, GPU count, optional sizes in KB
./ring p2p 8 > results/my_ring.csv

# hold the hand-written kernels to NCCL's launch budget
RING_BLOCKS=24 RING_THREADS=512 ./ring p2p 8 > results/my_ring_b24_t512.csv

# one process per GPU
./ring_mp 8 > results/my_ring_mp.csv
```

The header comment of each source file lists its remaining options (`RING_VEC`, `RING_LL_MAX`, `RING_IMPLS`).

Every run verifies its output: GPU `r` starts with `r + 1` in every element, so each output element must equal `P(P+1)/2`.

## Results

| Files | Machine |
|---|---|
| `cuda_copy_reduce_2xT4_*.csv`, `cuda_ring_2xT4.csv` | 2x T4, PCIe 3.0, peer access |
| `cuda_ring_4xL4.csv`, `cuda_ring_4xL4_verify/` | 4x L4 (g6.12xlarge), PCIe 4.0, no peer access |
| `cuda_ring_4xA10G.csv` | 4x A10G (g5.12xlarge), PCIe 4.0, no peer access |
| `cuda_ring_4xL4_p2p/` | 4x L4, PCIe 4.0, peer access |
| `cuda_ring_8xA100*.csv`, `cuda_ring_8xA100_verify/`, `cuda_ring_A100_subsets.csv` | 8x A100 (p4d.24xlarge), NVLink |
| `cuda_ring_8xH100*/` | 8x H100, NVLink with in-switch reduction |
| `cuda_ring_8xB200*/` | 8x B200, NVLink with in-switch reduction |

## Caveats

- Direct read has no flags. The benchmark makes every input ready before timing and keeps buffers alive until all readers finish, while NCCL does that coordination inside its call.
- The LL protocol relies on each 8-byte data and flag pair arriving together, which the PTX memory model does not guarantee for vector stores.
- A waiting block spins on its SM, so the fused ring needs a grid small enough for every block to be resident at once.
