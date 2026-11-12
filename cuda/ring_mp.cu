// All-reduce with one process per GPU, the layout training actually runs in. The parent forks P
// children before touching CUDA; child r owns GPU r. The children exchange CUDA IPC handles for
// their buffers and the NCCL id through an anonymous shared mmap, and synchronise with a shared-
// memory barrier. Two implementations, the same kernels as ring.cu:
//   ring_fused   one kernel per process, per-block flags / LL words in peer memory (ring_fused.cuh)
//   direct       every GPU reads all P inputs over the link and sums them: one hop, no
//                synchronisation at all, (P-1)N bytes per GPU, so only sensible for small messages
//   auto         direct while (P-1) * bytes <= 8 MB, the fused ring above: the dispatch a real
//                implementation would do (NCCL's tree-vs-ring switch is the same idea)
//   nccl         ncclAllReduce on a communicator from ncclCommInitRank
// Each rank times its own call from a shared barrier to its own stream sync; the collective's
// latency is the max over ranks per iteration, then the median over iterations.
//
//   ./ring_mp [ngpus] [size_kb[,size_kb...]]      env: RING_BLOCKS, RING_THREADS, RING_LL_MAX, RING_IMPLS

#include <cuda_runtime.h>
#include <nccl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("rank %d %s:%d %s -> %s\n", rank, __FILE__, __LINE__, #x, cudaGetErrorString(e)); exit(1); } } while (0)
#define NCK(x) do { ncclResult_t rc_ = (x); if (rc_ != ncclSuccess) { \
  printf("rank %d %s:%d %s -> %s\n", rank, __FILE__, __LINE__, #x, ncclGetErrorString(rc_)); exit(1); } } while (0)

const int MAXP = 8;
int P, rank;
__device__ __forceinline__ float4 f4add(float4 x, float4 y) {
  return make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
}
#include "ring_fused.cuh"

struct Ptrs { const float* p[MAXP]; };
__global__ void sum_all4(Ptrs in, float4* out, int P, int n4) {
  int stride = gridDim.x * blockDim.x;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  for (; i + stride < n4; i += 2 * stride) {   // two elements per iteration, all loads issued first
    float4 v0 = make_float4(0.f, 0.f, 0.f, 0.f), v1 = v0;
    float4 x0[MAXP], x1[MAXP];
#pragma unroll
    for (int p = 0; p < MAXP; p++)
      if (p < P) { x0[p] = __ldcg((const float4*)in.p[p] + i); x1[p] = __ldcg((const float4*)in.p[p] + i + stride); }
#pragma unroll
    for (int p = 0; p < MAXP; p++)
      if (p < P) { v0 = f4add(v0, x0[p]); v1 = f4add(v1, x1[p]); }
    out[i] = v0; out[i + stride] = v1;
  }
  for (; i < n4; i += stride) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int p = 0; p < MAXP; p++)
      if (p < P) v = f4add(v, __ldcg((const float4*)in.p[p] + i));
    out[i] = v;
  }
}

__global__ void fill(float* a, int r, int n, int mul, int add, int mod) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x)
    a[i] = (float)(r + 1 + (int)(((long)i * mul + (long)r * add) % mod));
}

// Shared between the P processes (anonymous MAP_SHARED mmap created before the fork).
struct Shared {
  cudaIpcMemHandle_t hin[MAXP], hout[MAXP], hflag[MAXP], hll[MAXP];
  ncclUniqueId id;
  std::atomic<int> bar_count, bar_gen;
  double t[MAXP];
  long bad[MAXP];
};
Shared* sh;

void barrier() {
  int gen = sh->bar_gen.load();
  if (sh->bar_count.fetch_add(1) == P - 1) { sh->bar_count.store(0); sh->bar_gen.store(gen + 1); }
  else while (sh->bar_gen.load() == gen) {}
}

// Per-rank state. in/out/flags/ll hold this rank's own buffers at [rank] and IPC mappings of the
// others' elsewhere, so the kernels see the same P-entry tables they do in ring.cu.
float *in[MAXP], *out[MAXP];
unsigned* flags[MAXP];
uint4* llbuf[MAXP];
cudaStream_t s;
ncclComm_t comm;
int numThreads = 512, fusedBlocks = 24, LLCAP = 262144;
const int LLSLOTS = 2 * MAXP - 2;
unsigned fusedCalls = 0;
inline int blocks(int n) { return (n + numThreads - 1) / numThreads; }

void ring_fused(int N) {
  int chunk = (N + P - 1) / P;
  unsigned base = fusedCalls * (2 * P - 2);   // pipeK = 1
  fusedCalls++;
  if (LLCAP > 0 && chunk <= LLCAP && N % (2 * P) == 0) {
    LLArgs a;
    for (int r = 0; r < P; r++) { a.in[r] = in[r]; a.out[r] = out[r]; a.ll[r] = llbuf[r]; a.flag[r] = flags[r]; }
    ring_fused_ll_kernel<<<fusedBlocks, numThreads, 0, s>>>(a, rank, P, N, base, LLCAP / 2, 1);
  } else {
    FusedArgs a;
    for (int r = 0; r < P; r++) { a.in[r] = in[r]; a.out[r] = out[r]; a.flag[r] = flags[r]; }
    ring_fused_kernel<<<fusedBlocks, numThreads, 0, s>>>(a, rank, P, N, base, 1);
  }
}

void direct(int N) {   // N is a multiple of 4 for every size in the sweep
  Ptrs ptrs;
  for (int p = 0; p < P; p++) ptrs.p[p] = in[p];
  sum_all4<<<fusedBlocks, numThreads, 0, s>>>(ptrs, (float4*)out[rank], P, N / 4);
}

void nccl_all_reduce(int N) {
  NCK(ncclAllReduce(in[rank], out[rank], N, ncclFloat, ncclSum, comm, s));
}

void auto_dispatch(int N) {
  if ((long)(P - 1) * N * sizeof(float) <= (8l << 20)) direct(N); else ring_fused(N);
}

struct Impl { const char* name; void (*fn)(int); bool needs_p2p; };
Impl impls[] = {{"ring_fused", ring_fused, true}, {"direct", direct, true}, {"auto", auto_dispatch, true},
                {"nccl", nccl_all_reduce, false}};
bool p2p = true;   // every pair can access each other's memory; without it only NCCL runs

int child(int MAXN, const std::vector<int>& sizes_kb, const std::string& only) {
  CK(cudaSetDevice(rank));
  for (int r = 0; r < P; r++) {
    if (r == rank) continue;
    int can; CK(cudaDeviceCanAccessPeer(&can, rank, r));
    if (!can) { p2p = false; continue; }
    cudaError_t pe = cudaDeviceEnablePeerAccess(r, 0);
    if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) CK(pe);
  }
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  CK(cudaMalloc(&in[rank], (size_t)MAXN * sizeof(float)));
  CK(cudaMalloc(&out[rank], (size_t)MAXN * sizeof(float)));
  CK(cudaMalloc(&flags[rank], 1024 * sizeof(unsigned)));
  CK(cudaMemset(flags[rank], 0, 1024 * sizeof(unsigned)));
  size_t llbytes = (size_t)LLSLOTS * (LLCAP > 0 ? LLCAP / 2 : 1) * sizeof(uint4);
  CK(cudaMalloc(&llbuf[rank], llbytes));
  CK(cudaMemset(llbuf[rank], 0, llbytes));
  CK(cudaIpcGetMemHandle(&sh->hin[rank], in[rank]));
  CK(cudaIpcGetMemHandle(&sh->hout[rank], out[rank]));
  CK(cudaIpcGetMemHandle(&sh->hflag[rank], flags[rank]));
  CK(cudaIpcGetMemHandle(&sh->hll[rank], llbuf[rank]));
  if (rank == 0) NCK(ncclGetUniqueId(&sh->id));
  barrier();
  for (int r = 0; r < P && p2p; r++) {   // IPC mappings of peer memory need peer access
    if (r == rank) continue;
    CK(cudaIpcOpenMemHandle((void**)&in[r], sh->hin[r], cudaIpcMemLazyEnablePeerAccess));
    CK(cudaIpcOpenMemHandle((void**)&out[r], sh->hout[r], cudaIpcMemLazyEnablePeerAccess));
    CK(cudaIpcOpenMemHandle((void**)&flags[r], sh->hflag[r], cudaIpcMemLazyEnablePeerAccess));
    CK(cudaIpcOpenMemHandle((void**)&llbuf[r], sh->hll[r], cudaIpcMemLazyEnablePeerAccess));
  }
  NCK(ncclCommInitRank(&comm, P, sh->id, rank));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, rank));
  if (fusedBlocks > prop.multiProcessorCount) fusedBlocks = prop.multiProcessorCount;
  barrier();
  if (rank == 0) fprintf(stderr, "%s x%d, one process per GPU, buffers via CUDA IPC, fused ring %dx%d, peer access %s\n",
                         prop.name, P, fusedBlocks, numThreads, p2p ? "on" : "unavailable (NCCL only)");

  std::vector<float> h(MAXN), want(MAXN);
  const int PAT[2][3] = {{7, 13, 1021}, {11, 17, 1019}};
  auto fill_inputs = [&](int N, int pat) {
    fill<<<blocks(N), numThreads, 0, s>>>(in[rank], rank, N, PAT[pat][0], PAT[pat][1], PAT[pat][2]);
    CK(cudaMemsetAsync(out[rank], 0xFF, (size_t)N * sizeof(float), s));  // NaN
    CK(cudaStreamSynchronize(s));
    barrier();   // nobody starts until every rank's inputs are in place
  };
  auto verify = [&](int N, int pat) -> long {
    for (long i = 0; i < N; i++) {
      int v = 0;
      for (int r = 0; r < P; r++) v += r + 1 + (int)((i * PAT[pat][0] + (long)r * PAT[pat][1]) % PAT[pat][2]);
      want[i] = (float)v;
    }
    CK(cudaMemcpy(h.data(), out[rank], (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    long bad = 0;
    for (int i = 0; i < N; i++) bad += (h[i] != want[i]);
    return bad;
  };
  auto run_once = [&](Impl& impl, int N) -> double {   // barrier, own call, own sync, own time
    barrier();
    auto t0 = std::chrono::steady_clock::now();
    impl.fn(N);
    CK(cudaStreamSynchronize(s));
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  };

  if (rank == 0) printf("impl,gpu,ngpus,p2p,bytes,iters,median_us,p10_us,p90_us,algbw_GBps,busbw_GBps,status\n");
  for (int kb : sizes_kb) {
    int N = kb * 1024 / sizeof(float);
    size_t bytes = (size_t)N * sizeof(float);
    for (Impl& impl : impls) {
      if (impl.needs_p2p && !p2p) continue;
      if (!only.empty() && ("," + only + ",").find("," + std::string(impl.name) + ",") == std::string::npos) continue;
      fill_inputs(N, 0);
      run_once(impl, N);
      sh->bad[rank] = verify(N, 0);
      for (int w = 0; w < 3; w++) run_once(impl, N);
      int iters = bytes >= (64u << 20) ? 20 : 100;
      std::vector<double> t(iters);
      for (int i = 0; i < iters; i++) {
        sh->t[rank] = run_once(impl, N);
        barrier();                                  // all ranks' times for this iteration are in
        if (rank == 0) { double m = 0; for (int r = 0; r < P; r++) m = std::max(m, sh->t[r]); t[i] = m; }
        barrier();                                  // rank 0 has read them before anyone overwrites
      }
      fill_inputs(N, 1);
      run_once(impl, N);
      sh->bad[rank] += verify(N, 1);
      barrier();
      if (rank == 0) {
        long bad = 0; for (int r = 0; r < P; r++) bad += sh->bad[r];
        std::sort(t.begin(), t.end());
        double med = t[iters / 2], algbw = bytes / med / 1e3, busbw = algbw * 2.0 * (P - 1) / P;
        printf("%s,%s,%d,%d,%zu,%d,%.1f,%.1f,%.1f,%.3f,%.3f,%s\n", impl.name, prop.name, P, p2p, bytes, iters,
               med, t[iters / 10], t[iters * 9 / 10], algbw, busbw, bad ? "WRONG" : "ok");
        fflush(stdout);
      }
      barrier();
    }
  }
  NCK(ncclCommDestroy(comm));
  return 0;
}

int main(int argc, char** argv) {
  P = argc > 1 ? atoi(argv[1]) : 8;
  std::vector<int> sizes_kb = {4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144};
  if (argc > 2) {
    sizes_kb.clear();
    for (char* tok = strtok(argv[2], ","); tok; tok = strtok(nullptr, ",")) sizes_kb.push_back(atoi(tok));
  }
  const int MAXN = *std::max_element(sizes_kb.begin(), sizes_kb.end()) * 1024 / sizeof(float);
  if (getenv("RING_BLOCKS")) fusedBlocks = atoi(getenv("RING_BLOCKS"));
  if (getenv("RING_THREADS")) numThreads = atoi(getenv("RING_THREADS"));
  if (getenv("RING_LL_MAX")) LLCAP = atoi(getenv("RING_LL_MAX"));
  std::string only = getenv("RING_IMPLS") ? getenv("RING_IMPLS") : "";
  if (P < 2 || P > MAXP) { printf("ngpus must be 2..%d\n", MAXP); return 1; }

  // The parent must not touch CUDA before forking: a CUDA context does not survive fork().
  sh = (Shared*)mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (sh == MAP_FAILED) { perror("mmap"); return 1; }
  memset((void*)sh, 0, sizeof(Shared));
  new (&sh->bar_count) std::atomic<int>(0);
  new (&sh->bar_gen) std::atomic<int>(0);
  std::vector<pid_t> pids;
  for (int r = 0; r < P; r++) {
    pid_t pid = fork();
    if (pid == 0) { rank = r; exit(child(MAXN, sizes_kb, only)); }
    pids.push_back(pid);
  }
  int worst = 0;
  for (pid_t pid : pids) { int st = 0; waitpid(pid, &st, 0); worst = std::max(worst, WIFEXITED(st) ? WEXITSTATUS(st) : 1); }
  return worst;
}
