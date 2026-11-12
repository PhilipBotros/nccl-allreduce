// All-reduce across P GPUs in one process, five ways on one harness:
//   copy_all     every GPU pulls every other input with cudaMemcpyPeerAsync, then sums P buffers
//   remote_all   every GPU's kernel reads every other input directly over the link
//   ring         ring reduce-scatter + all-gather, both phases as peer copies (pull, events)
//   ring_remote  same ring, but the reduce-scatter add reads the neighbour's chunk directly
//   ring_fused   the same ring in ONE kernel per GPU: blocks synchronise with their neighbours
//                through flags in peer memory, so the host issues one launch per GPU per collective
//   nccl         ncclAllReduce on the same streams
// The ring is what NCCL runs: P chunks, P-1 steps per phase, each GPU talks only to its
// neighbour, and 2(P-1)/P * N bytes cross the link per GPU instead of (P-1) * N.
//
//   ./ring [p2p|nop2p] [ngpus] [size_kb[,size_kb...]]      e.g. ./ring p2p 8 > results/x.csv
//   Buffers are sized for the largest size requested, so a 1 GB point needs 9 GB per GPU at P=8.
//   RING_BLOCKS=24 caps every hand-written kernel at 24 blocks per GPU (grid-stride), to compare
//   against NCCL at the SM budget NCCL actually uses; RING_THREADS=512 sets the block size;
//   RING_VEC=0 uses the plain one-float-per-thread kernels instead of the float4, 4x-unrolled ones.
//   Impl names get a suffix in the CSV for anything non-default, e.g. ring_remote_b24_t512.

#include <cuda_runtime.h>
#include <nccl.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e)); exit(1); } } while (0)
#define NCK(x) do { ncclResult_t rc_ = (x); if (rc_ != ncclSuccess) { \
  printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, ncclGetErrorString(rc_)); exit(1); } } while (0)

#include <cstdint>
#include <string>
#include <thread>

const int MAXP = 8;
int numThreads = 512;   // RING_THREADS (NCCL's block size)
bool useVec = true;     // RING_VEC

struct Ptrs { const float* p[MAXP]; };  // passed by value so a kernel can read up to 8 inputs

// Position-dependent integer inputs: GPU r holds r+1 + (i*mul + r*add) % mod at element i, so a
// kernel that reads the wrong element or the wrong chunk produces a wrong sum, and all sums are
// small integers, exact in fp32 in any order. Two patterns: one checked before timing, one after.
__global__ void fill(float* a, int r, int n, int mul, int add, int mod) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x)
    a[i] = (float)(r + 1 + (int)(((long)i * mul + (long)r * add) % mod));
}

// Grid-stride kernels: correct for any block count, so the grid can be capped (RING_BLOCKS).
__global__ void add(const float* a, const float* b, float* c, int n) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x)
    c[i] = a[i] + b[i];
}

__device__ __forceinline__ float4 f4add(float4 x, float4 y) {
  return make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
}

// float4 loads, four independent loads per pointer in flight per thread before the first store:
// what it takes to pull link bandwidth with few SMs, which is how NCCL's kernels are written.
__global__ void add4(const float4* __restrict__ a, const float4* __restrict__ b, float4* __restrict__ c, int n4) {
  int stride = gridDim.x * blockDim.x;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  for (; i + 3 * stride < n4; i += 4 * stride) {
    float4 a0 = a[i], a1 = a[i + stride], a2 = a[i + 2 * stride], a3 = a[i + 3 * stride];
    float4 b0 = b[i], b1 = b[i + stride], b2 = b[i + 2 * stride], b3 = b[i + 3 * stride];
    c[i] = f4add(a0, b0); c[i + stride] = f4add(a1, b1);
    c[i + 2 * stride] = f4add(a2, b2); c[i + 3 * stride] = f4add(a3, b3);
  }
  for (; i < n4; i += stride) c[i] = f4add(a[i], b[i]);
}

__global__ void sum_all(Ptrs in, float* out, int P, int n) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    float v = 0.f;
#pragma unroll
    for (int p = 0; p < MAXP; p++)   // unrolled so the struct param is indexed by constants and
      if (p < P) v += in.p[p][i];    // stays in param space instead of spilling to local memory
    out[i] = v;
  }
}

__global__ void sum_all4(Ptrs in, float4* out, int P, int n4) {
  int stride = gridDim.x * blockDim.x;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  for (; i + stride < n4; i += 2 * stride) {   // two elements per iteration, all loads issued first
    float4 v0 = make_float4(0.f, 0.f, 0.f, 0.f), v1 = v0;
    float4 x0[MAXP], x1[MAXP];
#pragma unroll
    for (int p = 0; p < MAXP; p++)
      if (p < P) { x0[p] = ((const float4*)in.p[p])[i]; x1[p] = ((const float4*)in.p[p])[i + stride]; }
#pragma unroll
    for (int p = 0; p < MAXP; p++)
      if (p < P) { v0 = f4add(v0, x0[p]); v1 = f4add(v1, x1[p]); }
    out[i] = v0; out[i + stride] = v1;
  }
  for (; i < n4; i += stride) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int p = 0; p < MAXP; p++)
      if (p < P) v = f4add(v, ((const float4*)in.p[p])[i]);
    out[i] = v;
  }
}

int P;                                      // number of GPUs in the collective
float *in[MAXP], *out[MAXP], *recv[MAXP];   // recv[r] holds (P-1) slots of MAXN floats
cudaStream_t s[MAXP];
cudaEvent_t ev[MAXP][2 * MAXP];             // one per (gpu, ring step), never reused in a call
ncclComm_t comm[MAXP];

int maxBlocks = 0;                          // RING_BLOCKS: 0 = one thread per element
inline int blocks(int n) {
  int b = (n + numThreads - 1) / numThreads;
  return maxBlocks > 0 && b > maxBlocks ? maxBlocks : b;
}
inline bool aligned16(const void* p) { return ((uintptr_t)p & 15) == 0; }

// Every launch goes through these so the kernel flavour, block size and block cap apply everywhere.
void launch_add(const float* a, const float* b, float* c, int n, cudaStream_t st) {
  if (useVec && n % 4 == 0 && aligned16(a) && aligned16(b) && aligned16(c))
    add4<<<blocks(n / 4), numThreads, 0, st>>>((const float4*)a, (const float4*)b, (float4*)c, n / 4);
  else
    add<<<blocks(n), numThreads, 0, st>>>(a, b, c, n);
}
void launch_sum_all(Ptrs ptrs, float* out, int n, cudaStream_t st) {
  bool ok = useVec && n % 4 == 0 && aligned16(out);
  for (int p = 0; p < P && ok; p++) ok = aligned16(ptrs.p[p]);
  if (ok) sum_all4<<<blocks(n / 4), numThreads, 0, st>>>(ptrs, (float4*)out, P, n / 4);
  else sum_all<<<blocks(n), numThreads, 0, st>>>(ptrs, out, P, n);
}
inline long off(int c, int N) { return (long)N * c / P; }            // chunk c starts here
inline int len(int c, int N) { return (int)(off(c + 1, N) - off(c, N)); }

// A generalised: pull all P-1 inputs, sum locally. (P-1)*N bytes in per GPU.
void copy_all(int N) {
  for (int r = 0; r < P; r++) {
    CK(cudaSetDevice(r));
    Ptrs ptrs; ptrs.p[0] = in[r];
    int k = 1;
    for (int p = 0; p < P; p++) {
      if (p == r) continue;
      float* slot = recv[r] + (long)(k - 1) * N;
      CK(cudaMemcpyPeerAsync(slot, r, in[p], p, (size_t)N * sizeof(float), s[r]));
      ptrs.p[k++] = slot;
    }
    launch_sum_all(ptrs, out[r], N, s[r]);
  }
}

// B generalised: one kernel per GPU reads all P inputs, P-1 of them remote.
void remote_all(int N) {
  for (int r = 0; r < P; r++) {
    CK(cudaSetDevice(r));
    Ptrs ptrs; ptrs.p[0] = in[r];
    int k = 1;
    for (int p = 0; p < P; p++) if (p != r) ptrs.p[k++] = in[p];
    launch_sum_all(ptrs, out[r], N, s[r]);
  }
}

// C generalised: ring. GPU r receives from r-1 and sends to r+1.
// Reduce-scatter step st: GPU r adds chunk (r-st-1) from its neighbour into its own copy;
// after P-1 steps GPU r holds the fully reduced chunk (r+1).
// All-gather step st: GPU r copies chunk (r-st) from its neighbour, which just completed it.
// Copies are pulls on the receiver's stream, so each step waits on an event the neighbour
// recorded after its previous step. The accumulator is out[], so no initial in->out copy:
// step 0 reads the neighbour's raw input and later steps read its partial sum.
void ring_impl(int N, bool remote_rs) {
  for (int st = 0; st < P - 1; st++) {
    for (int r = 0; r < P; r++) {
      int src = (r - 1 + P) % P;
      int c = ((r - st - 1) % P + P) % P;
      long o = off(c, N); int n = len(c, N);
      const float* from = (st == 0 ? in[src] : out[src]) + o;
      CK(cudaSetDevice(r));
      if (st > 0) CK(cudaStreamWaitEvent(s[r], ev[src][st - 1], 0));
      if (remote_rs) {
        launch_add(in[r] + o, from, out[r] + o, n, s[r]);
      } else {
        CK(cudaMemcpyPeerAsync(recv[r], r, from, src, (size_t)n * sizeof(float), s[r]));
        launch_add(in[r] + o, recv[r], out[r] + o, n, s[r]);
      }
      CK(cudaEventRecord(ev[r][st], s[r]));
    }
  }
  for (int st = 0; st < P - 1; st++) {
    for (int r = 0; r < P; r++) {
      int src = (r - 1 + P) % P;
      int c = ((r - st) % P + P) % P;
      long o = off(c, N); int n = len(c, N);
      int nxt = (r + 1) % P;
      CK(cudaSetDevice(r));
      CK(cudaStreamWaitEvent(s[r], ev[src][P - 2 + st], 0));
      if (st > 0) CK(cudaStreamWaitEvent(s[r], ev[nxt][st], 0));  // nxt has read this chunk's partial sum
      CK(cudaMemcpyPeerAsync(out[r] + o, r, out[src] + o, src, (size_t)n * sizeof(float), s[r]));
      if (st < P - 2) CK(cudaEventRecord(ev[r][P - 1 + st], s[r]));
    }
  }
}
void ring(int N) { ring_impl(N, false); }
void ring_remote(int N) { ring_impl(N, true); }

// Fused ring: device code in ring_fused.cuh, shared with the multi-process harness ring_mp.cu.
#include "ring_fused.cuh"
unsigned* flags[MAXP];
unsigned fusedCalls = 0;
int fusedBlocks = 24;   // NCCL's channel count on an A100 node; RING_BLOCKS overrides; must be <= resident blocks since blocks spin on each other
bool fusedMT = false;   // RING_MT=1: launch the P kernels from P host threads instead of one
int pipeK = 1;   // measured: sub-chunk pipelining only adds flag traffic on NVLink; 1 is best
uint4* llbuf[MAXP];
int LLCAP = 262144;     // RING_LL_MAX: floats per chunk up to which the LL path is used (0 = never)
const int LLSLOTS = 2 * MAXP - 2;

void ring_fused(int N) {
  int chunk = (N + P - 1) / P;
  if (LLCAP > 0 && chunk <= LLCAP && N % (2 * P) == 0) {
    LLArgs a;
    for (int r = 0; r < P; r++) { a.in[r] = in[r]; a.out[r] = out[r]; a.ll[r] = llbuf[r]; a.flag[r] = flags[r]; }
    unsigned base = fusedCalls * (2 * P - 2) * pipeK;
    fusedCalls++;
    for (int r = 0; r < P; r++) {
      CK(cudaSetDevice(r));
      ring_fused_ll_kernel<<<fusedBlocks, numThreads, 0, s[r]>>>(a, r, P, N, base, LLCAP / 2, pipeK);
    }
    return;
  }
  FusedArgs a;
  for (int r = 0; r < P; r++) { a.in[r] = in[r]; a.out[r] = out[r]; a.flag[r] = flags[r]; }
  unsigned base = fusedCalls * (2 * P - 2) * pipeK;
  fusedCalls++;
  if (fusedMT) {
    std::vector<std::thread> th;
    for (int r = 0; r < P; r++)
      th.emplace_back([=] { CK(cudaSetDevice(r)); ring_fused_kernel<<<fusedBlocks, numThreads, 0, s[r]>>>(a, r, P, N, base, pipeK); });
    for (auto& t : th) t.join();
    return;
  }
  for (int r = 0; r < P; r++) {
    CK(cudaSetDevice(r));
    ring_fused_kernel<<<fusedBlocks, numThreads, 0, s[r]>>>(a, r, P, N, base, pipeK);
  }
}

void nccl_all_reduce(int N) {
  NCK(ncclGroupStart());
  for (int r = 0; r < P; r++) {
    CK(cudaSetDevice(r));
    NCK(ncclAllReduce(in[r], out[r], N, ncclFloat, ncclSum, comm[r], s[r]));
  }
  NCK(ncclGroupEnd());
}

void sync_all() {
  for (int r = 0; r < P; r++) { CK(cudaSetDevice(r)); CK(cudaDeviceSynchronize()); }
}

struct Impl { const char* name; void (*fn)(int); bool needs_p2p; };
Impl impls[] = {{"copy_all", copy_all, false}, {"remote_all", remote_all, true},
                {"ring", ring, false}, {"ring_remote", ring_remote, true},
                {"ring_fused", ring_fused, true}, {"nccl", nccl_all_reduce, false}};

int main(int argc, char** argv) {
  bool want_p2p = !(argc > 1 && strcmp(argv[1], "nop2p") == 0);
  int want_P = argc > 2 ? atoi(argv[2]) : 0;
  std::vector<int> sizes_kb = {4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144};
  if (argc > 3) {  // comma-separated list of sizes in KB
    sizes_kb.clear();
    for (char* tok = strtok(argv[3], ","); tok; tok = strtok(nullptr, ",")) sizes_kb.push_back(atoi(tok));
  }
  const int MAXN = *std::max_element(sizes_kb.begin(), sizes_kb.end()) * 1024 / sizeof(float);

  if (getenv("RING_BLOCKS")) maxBlocks = fusedBlocks = atoi(getenv("RING_BLOCKS"));
  if (getenv("RING_THREADS")) numThreads = atoi(getenv("RING_THREADS"));
  if (getenv("RING_VEC")) useVec = atoi(getenv("RING_VEC")) != 0;
  if (getenv("RING_MT")) fusedMT = atoi(getenv("RING_MT")) != 0;
  if (getenv("RING_LL_MAX")) LLCAP = atoi(getenv("RING_LL_MAX"));
  std::string only = getenv("RING_IMPLS") ? getenv("RING_IMPLS") : "";   // comma list: run only these
  char suffix[32] = "";
  if (maxBlocks > 0) snprintf(suffix + strlen(suffix), 8, "_b%d", maxBlocks);
  if (numThreads != 512) snprintf(suffix + strlen(suffix), 8, "_t%d", numThreads);
  if (!useVec) snprintf(suffix + strlen(suffix), 8, "_scalar");
  if (fusedMT) snprintf(suffix + strlen(suffix), 8, "_mt");
  if (LLCAP != 262144) snprintf(suffix + strlen(suffix), 12, "_ll%d", LLCAP);
  if (getenv("RING_PIPE")) pipeK = atoi(getenv("RING_PIPE"));
  if (pipeK != 1) snprintf(suffix + strlen(suffix), 8, "_k%d", pipeK);
  if (FUSED_UNROLL != 4) snprintf(suffix + strlen(suffix), 8, "_u%d", (int)FUSED_UNROLL);

  int gpu_count;
  CK(cudaGetDeviceCount(&gpu_count));
  P = want_P > 0 ? want_P : std::min(gpu_count, MAXP);
  if (P < 2 || P > gpu_count || P > MAXP) { printf("need 2..%d GPUs, have %d, asked %d\n", MAXP, gpu_count, P); return 1; }

  bool p2p = want_p2p;
  for (int r = 0; r < P && p2p; r++)
    for (int p = 0; p < P; p++) {
      if (p == r) continue;
      int can; CK(cudaDeviceCanAccessPeer(&can, r, p));
      if (!can) p2p = false;
    }
  if (p2p)
    for (int r = 0; r < P; r++) {
      CK(cudaSetDevice(r));
      for (int p = 0; p < P; p++) if (p != r) CK(cudaDeviceEnablePeerAccess(p, 0));
    }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  if (fusedBlocks > prop.multiProcessorCount) fusedBlocks = prop.multiProcessorCount;  // all blocks must be resident
  fprintf(stderr, "%s x%d of %d, %d SMs, fused grid %d x %d, unroll %d, peer access %s\n", prop.name, P, gpu_count,
          prop.multiProcessorCount, fusedBlocks, numThreads, (int)FUSED_UNROLL,
          p2p ? "enabled" : (want_p2p ? "unavailable, disabled" : "disabled"));

  std::vector<int> devs(P);
  for (int r = 0; r < P; r++) devs[r] = r;
  NCK(ncclCommInitAll(comm, P, devs.data()));

  for (int r = 0; r < P; r++) {
    CK(cudaSetDevice(r));
    CK(cudaStreamCreateWithFlags(&s[r], cudaStreamNonBlocking));
    for (int i = 0; i < 2 * MAXP; i++) CK(cudaEventCreateWithFlags(&ev[r][i], cudaEventDisableTiming));
    CK(cudaMalloc(&in[r], (size_t)MAXN * sizeof(float)));
    CK(cudaMalloc(&out[r], (size_t)MAXN * sizeof(float)));
    CK(cudaMalloc(&recv[r], (size_t)(P - 1) * MAXN * sizeof(float)));
    CK(cudaMalloc(&flags[r], 1024 * sizeof(unsigned)));
    CK(cudaMemset(flags[r], 0, 1024 * sizeof(unsigned)));
    size_t llbytes = (size_t)LLSLOTS * (LLCAP > 0 ? LLCAP / 2 : 1) * sizeof(uint4);
    CK(cudaMalloc(&llbuf[r], llbytes));
    CK(cudaMemset(llbuf[r], 0, llbytes));
  }
  sync_all();
  std::vector<float> h(MAXN);
  const int PAT[2][3] = {{7, 13, 1021}, {11, 17, 1019}};   // mul, add, mod
  auto fill_inputs = [&](int N, int pat) {
    for (int r = 0; r < P; r++) { CK(cudaSetDevice(r)); fill<<<blocks(N), numThreads>>>(in[r], r, N, PAT[pat][0], PAT[pat][1], PAT[pat][2]); }
    for (int r = 0; r < P; r++) { CK(cudaSetDevice(r)); CK(cudaMemset(out[r], 0xFF, (size_t)N * sizeof(float))); }  // NaN
    sync_all();   // fills and memsets are on the legacy stream, which our non-blocking streams ignore
  };
  auto verify = [&](int N, int pat) -> long {
    std::vector<float> want(N);
    for (long i = 0; i < N; i++) {
      int v = 0;
      for (int r = 0; r < P; r++) v += r + 1 + (int)((i * PAT[pat][0] + (long)r * PAT[pat][1]) % PAT[pat][2]);
      want[i] = (float)v;
    }
    long bad = 0;
    for (int r = 0; r < P; r++) {
      CK(cudaSetDevice(r));
      CK(cudaMemcpy(h.data(), out[r], (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
      for (int i = 0; i < N; i++) bad += (h[i] != want[i]);
    }
    return bad;
  };

  printf("impl,gpu,ngpus,p2p,bytes,iters,median_us,p10_us,p90_us,algbw_GBps,busbw_GBps,status\n");
  for (int kb : sizes_kb) {
    int N = kb * 1024 / sizeof(float);
    size_t bytes = (size_t)N * sizeof(float);
    for (Impl& impl : impls) {
      if (impl.needs_p2p && !p2p) continue;
      if (!only.empty() && ("," + only + ",").find("," + std::string(impl.name) + ",") == std::string::npos) continue;
      fill_inputs(N, 0);
      impl.fn(N);
      sync_all();
      long bad = verify(N, 0);                       // first call, pattern 0
      for (int w = 0; w < 3; w++) { impl.fn(N); sync_all(); }
      int iters = bytes >= (64u << 20) ? 20 : 100;
      std::vector<double> t(iters);
      for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::steady_clock::now();
        impl.fn(N);
        sync_all();
        t[i] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
      }
      fill_inputs(N, 1);                             // after the timed loop: new inputs, NaN outputs, check again
      impl.fn(N);
      sync_all();
      bad += verify(N, 1);
      if (getenv("RING_SELFTEST"))                   // the checker must reject the output against the other pattern
        fprintf(stderr, "selftest %s %zu: mismatches vs wrong pattern = %ld of %ld (must be > 0)\n", impl.name, bytes, verify(N, 0), (long)N * P);
      std::sort(t.begin(), t.end());
      double med = t[iters / 2];
      double algbw = bytes / med / 1e3;                 // bytes per GPU / latency, GB/s
      double busbw = algbw * 2.0 * (P - 1) / P;         // NCCL's bus bandwidth convention
      printf("%s%s,%s,%d,%d,%zu,%d,%.1f,%.1f,%.1f,%.3f,%.3f,%s\n", impl.name, suffix, prop.name, P, p2p, bytes, iters,
             med, t[iters / 10], t[iters * 9 / 10], algbw, busbw, bad ? "WRONG" : "ok");
      fflush(stdout);
    }
  }
  for (int r = 0; r < P; r++) NCK(ncclCommDestroy(comm[r]));
  printf("# last error: %s\n", cudaGetErrorString(cudaGetLastError()));
  return 0;
}
