// Two-GPU all-reduce in CUDA. A: peer copy + local add (three ways of issuing
// the copies). B: the add kernel reads the other GPU's input directly.
// nvcc -O2 -o allreduce allreduce.cu && ./allreduce [p2p|nop2p] [size_kb]
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include <nccl.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e)); exit(1); } } while (0)
#define NCK(x) do { ncclResult_t r = (x); if (r != ncclSuccess) { \
  printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, ncclGetErrorString(r)); exit(1); } } while (0)

__global__ void add(const float* a, const float* b, float* c, int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

// Kernel variants for the remote-load experiment. Each thread's b[] load is a
// PCIe read request, so what matters is how many bytes are in flight per SM.
//
// Grid-stride: a fixed grid that exactly fills the GPU (one resident block
// per slot), each thread looping over elements. Same 4-byte requests, but no
// block launch/retire churn and a known number of threads in flight.
__global__ void add_stride(const float* a, const float* b, float* c, int N) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += gridDim.x * blockDim.x) {
    c[i] = a[i] + b[i];
  }
}

// float4: one 16-byte request per thread instead of four 4-byte ones. N must
// be a multiple of 4 (every size in the sweep is) and the pointers 16-byte
// aligned (cudaMalloc guarantees 256).
__global__ void add4(const float4* a, const float4* b, float4* c, int N4) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N4) {
    float4 x = a[i], y = b[i];
    c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
  }
}

__global__ void add4_stride(const float4* a, const float4* b, float4* c, int N4) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N4; i += gridDim.x * blockDim.x) {
    float4 x = a[i], y = b[i];
    c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
  }
}

// in[d], out[d], recv[d] live on GPU d. s[d] and ev[d] were created on GPU d.
float *in[2], *out[2], *recv[2];
cudaStream_t s[2];
cudaEvent_t ev[2];
ncclComm_t comm[2];

const int numThreads = 256;
int fullGrid;  // blocks that exactly fill the GPU at numThreads per block; set in main

// Serialised copies: each cudaMemcpyPeer is ordered after all other work on both
// GPUs it touches (it need not block the host), so the two copies run one after
// the other and the link only ever carries one direction at a time.
void copy_sync(int N) {
  cudaMemcpyPeer(recv[1], 1, in[0], 0, N * sizeof(float));
  cudaMemcpyPeer(recv[0], 0, in[1], 1, N * sizeof(float));
  int numBlocks = (N + numThreads - 1) / numThreads;
  cudaSetDevice(0);
  add<<<numBlocks, numThreads>>>(in[0], recv[0], out[0], N);
  cudaSetDevice(1);
  add<<<numBlocks, numThreads>>>(in[1], recv[1], out[1], N);
}

// Pull: the copy that fills recv[d] is enqueued on s[d], the destination
// GPU's stream, so the add on s[d] is ordered after it by the stream alone.
// The destination GPU's copy engine does the DMA: a read of remote memory.
void copy_pull(int N) {
  cudaMemcpyPeerAsync(recv[0], 0, in[1], 1, N * sizeof(float), s[0]);
  cudaMemcpyPeerAsync(recv[1], 1, in[0], 0, N * sizeof(float), s[1]);
  int numBlocks = (N + numThreads - 1) / numThreads;
  cudaSetDevice(0);
  add<<<numBlocks, numThreads, 0, s[0]>>>(in[0], recv[0], out[0], N);
  cudaSetDevice(1);
  add<<<numBlocks, numThreads, 0, s[1]>>>(in[1], recv[1], out[1], N);
}

// Push: each GPU's stream sends its own input to the other GPU, so the
// source GPU's copy engine does the DMA: a write to remote memory. The add on
// GPU d now depends on work in the other stream, so it waits on an event
// recorded after that copy. Events work across devices.
void copy_push(int N) {
  cudaMemcpyPeerAsync(recv[1], 1, in[0], 0, N * sizeof(float), s[0]);
  cudaEventRecord(ev[0], s[0]);
  cudaMemcpyPeerAsync(recv[0], 0, in[1], 1, N * sizeof(float), s[1]);
  cudaEventRecord(ev[1], s[1]);
  int numBlocks = (N + numThreads - 1) / numThreads;
  cudaStreamWaitEvent(s[0], ev[1], 0);
  cudaSetDevice(0);
  add<<<numBlocks, numThreads, 0, s[0]>>>(in[0], recv[0], out[0], N);
  cudaStreamWaitEvent(s[1], ev[0], 0);
  cudaSetDevice(1);
  add<<<numBlocks, numThreads, 0, s[1]>>>(in[1], recv[1], out[1], N);
}

// B: no copy at all. Each GPU's add kernel reads the other GPU's input
// directly; every b[idx] load crosses PCIe as a small read request. Needs peer
// access, otherwise the first remote load faults and kills the context.
void remote_reduce(int N) {
  int numBlocks = (N + numThreads - 1) / numThreads;
  cudaSetDevice(0);
  add<<<numBlocks, numThreads, 0, s[0]>>>(in[0], in[1], out[0], N);
  cudaSetDevice(1);
  add<<<numBlocks, numThreads, 0, s[1]>>>(in[1], in[0], out[1], N);
}

void remote_stride(int N) {
  cudaSetDevice(0);
  add_stride<<<fullGrid, numThreads, 0, s[0]>>>(in[0], in[1], out[0], N);
  cudaSetDevice(1);
  add_stride<<<fullGrid, numThreads, 0, s[1]>>>(in[1], in[0], out[1], N);
}

void remote_float4(int N) {
  int N4 = N / 4;
  int numBlocks = (N4 + numThreads - 1) / numThreads;
  cudaSetDevice(0);
  add4<<<numBlocks, numThreads, 0, s[0]>>>((const float4*)in[0], (const float4*)in[1], (float4*)out[0], N4);
  cudaSetDevice(1);
  add4<<<numBlocks, numThreads, 0, s[1]>>>((const float4*)in[1], (const float4*)in[0], (float4*)out[1], N4);
}

void remote_float4_stride(int N) {
  int N4 = N / 4;
  cudaSetDevice(0);
  add4_stride<<<fullGrid, numThreads, 0, s[0]>>>((const float4*)in[0], (const float4*)in[1], (float4*)out[0], N4);
  cudaSetDevice(1);
  add4_stride<<<fullGrid, numThreads, 0, s[1]>>>((const float4*)in[1], (const float4*)in[0], (float4*)out[1], N4);
}

// One-way: only GPU 0 reads remotely. GPU 1 adds from recv[1], which main
// filled with in[0] once at setup (inputs never change, so it stays valid).
// The link carries a single GPU's remote reads, to tell a per-GPU limit on
// outstanding reads from a shared cap on total traffic.
void remote_oneway(int N) {
  int numBlocks = (N + numThreads - 1) / numThreads;
  cudaSetDevice(0);
  add<<<numBlocks, numThreads, 0, s[0]>>>(in[0], in[1], out[0], N);
  cudaSetDevice(1);
  add<<<numBlocks, numThreads, 0, s[1]>>>(in[1], recv[1], out[1], N);
}

// Reduce-scatter: GPU 0 and GPU 1 compute half of the elements of the sum each.
// Then we do an all-gather to send each GPUs half to the other GPU.
void reduce_scatter(int N) {
  int h = N/2;
  int numBlocks = (h + numThreads - 1) / numThreads;
  cudaSetDevice(0);
  add<<<numBlocks, numThreads, 0, s[0]>>>(in[0], in[1], out[0], h);
  cudaSetDevice(1);
  int n1 = N - h;
  numBlocks = (n1 + numThreads - 1) / numThreads;
  add<<<numBlocks, numThreads, 0, s[1]>>>(in[1] + h, in[0] + h, out[1] + h, n1);

  // All gather the results now by using arrays offset
  cudaMemcpyPeerAsync(out[0] + h, 0, out[1] + h, 1, n1 * sizeof(float), s[1]);
  cudaMemcpyPeerAsync(out[1], 1, out[0], 0, h * sizeof(float), s[0]);
}

void nccl_all_reduce(int N) {
  NCK(ncclGroupStart());
  for (int d = 0; d < 2; d++) {
    CK(cudaSetDevice(d));
    NCK(ncclAllReduce(in[d], out[d], N, ncclFloat, ncclSum, comm[d], s[d]));
  }
  NCK(ncclGroupEnd());
}

void sync_both() {
  for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaDeviceSynchronize()); }
}

struct Impl { const char* name; void (*fn)(int); bool needs_p2p; };
Impl impls[] = {{"copy_sync", copy_sync, false}, {"copy_pull", copy_pull, false},
                {"copy_push", copy_push, false}, {"remote", remote_reduce, true},
                {"remote_stride", remote_stride, true}, {"remote_float4", remote_float4, true},
                {"remote_float4_stride", remote_float4_stride, true},
                {"remote_oneway", remote_oneway, true},
                {"reduce_scatter", reduce_scatter, true},
                {"nccl", nccl_all_reduce, false}};

int main(int argc, char** argv) {
  bool want_p2p = !(argc > 1 && strcmp(argv[1], "nop2p") == 0);
  const int MAXN = 64 << 20;  // 256 MB of floats, the largest size in the sweep
  std::vector<int> sizes_kb = {4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144};
  if (argc > 2) sizes_kb = {atoi(argv[2])};  // ./allreduce p2p 262144 -> one size, for profiling

  int gpu_count;
  CK(cudaGetDeviceCount(&gpu_count));
  if (gpu_count < 2) { printf("need at least 2 GPUs, found %d\n", gpu_count); return 1; }

  // Check if peer2peer is possible
  int can01, can10;
  CK(cudaDeviceCanAccessPeer(&can01, 0, 1));
  CK(cudaDeviceCanAccessPeer(&can10, 1, 0));
  bool p2p = want_p2p && can01 && can10;
  if (p2p) {
    CK(cudaSetDevice(0)); CK(cudaDeviceEnablePeerAccess(1, 0));   // 0 may now touch 1's memory
    CK(cudaSetDevice(1)); CK(cudaDeviceEnablePeerAccess(0, 0));   // and vice versa
  }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int devs[2] = {0, 1};
  NCK(ncclCommInitAll(comm, 2, devs));
  fullGrid = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / numThreads);
  fprintf(stderr, "%s x2, %d SMs, full grid %d blocks, can access peer 0->1: %d, 1->0: %d, peer access %s\n",
          prop.name, prop.multiProcessorCount, fullGrid, can01, can10,
          p2p ? "enabled" : "disabled (copies stage through host)");

  for (int d = 0; d < 2; d++) {
    CK(cudaSetDevice(d));
    CK(cudaStreamCreateWithFlags(&s[d], cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&ev[d], cudaEventDisableTiming));
    CK(cudaMalloc(&in[d], MAXN * sizeof(float)));
    CK(cudaMalloc(&out[d], MAXN * sizeof(float)));
    CK(cudaMalloc(&recv[d], MAXN * sizeof(float)));
  }
  // GPU 0 holds all 1s, GPU 1 all 3s, so every output element must be 4.
  std::vector<float> h(MAXN);
  std::fill(h.begin(), h.end(), 1.0f);
  CK(cudaMemcpy(in[0], h.data(), MAXN * sizeof(float), cudaMemcpyHostToDevice));
  std::fill(h.begin(), h.end(), 3.0f);
  CK(cudaMemcpy(in[1], h.data(), MAXN * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpyPeer(recv[1], 1, in[0], 0, MAXN * sizeof(float)));  // for remote_oneway

  printf("impl,gpu,p2p,bytes,iters,median_us,p10_us,p90_us,GBps,status\n");
  for (int kb : sizes_kb) {
    int N = kb * 1024 / sizeof(float);
    size_t bytes = N * sizeof(float);
    for (Impl& impl : impls) {
      if (impl.needs_p2p && !p2p) continue;
      // correctness: fill outputs with NaN first so unwritten elements show up
      for (int d = 0; d < 2; d++) { CK(cudaSetDevice(d)); CK(cudaMemset(out[d], 0xFF, bytes)); }
      impl.fn(N);
      sync_both();
      int bad = 0;
      for (int d = 0; d < 2; d++) {
        CK(cudaMemcpy(h.data(), out[d], bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; i++) bad += (h[i] != 4.0f);
      }

      // timing: host clock around the whole collective, both GPUs drained
      int iters = bytes >= (64u << 20) ? 20 : 100;
      for (int i = 0; i < 3; i++) { impl.fn(N); sync_both(); }
      std::vector<double> t(iters);
      for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::steady_clock::now();
        impl.fn(N);
        sync_both();
        t[i] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
      }
      std::sort(t.begin(), t.end());
      double med = t[iters / 2];
      printf("%s,%s,%d,%zu,%d,%.1f,%.1f,%.1f,%.3f,%s\n", impl.name, prop.name, p2p, bytes, iters,
             med, t[iters / 10], t[iters * 9 / 10], bytes / med / 1e3, bad ? "WRONG" : "ok");
    }
  }
  for (int d = 0; d < 2; d++) NCK(ncclCommDestroy(comm[d]));
  printf("# last error: %s\n", cudaGetErrorString(cudaGetLastError()));
  return 0;
}
