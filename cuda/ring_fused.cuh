// Device code for the fused ring all-reduce, shared by ring.cu (one process, P GPUs) and
// ring_mp.cu (one process per GPU, buffers mapped with CUDA IPC). The including file defines
// MAXP and f4add() first. Two kernels: ring_fused_kernel (per-block flags in peer memory, for
// large chunks) and ring_fused_ll_kernel (16-byte {data,flag} words, for chunks up to LLCAP).
#pragma once

// Fused ring: one kernel per GPU does every step. Block b works on the same element slice on every
// GPU, so block b only ever depends on block b of its neighbours, and signals through one flag per
// block that counts completed steps (monotonic across collectives, base = calls so far * 2(P-1)).
// Remote data is read with __ldcg so a stale L1 line from the previous collective can't be served.
struct FusedArgs { const float* in[MAXP]; float* out[MAXP]; unsigned* flag[MAXP]; };

// Block-level signal / wait on a flag in (possibly remote) device memory. The barrier makes every
// thread's writes visible to thread 0, whose system-scope release store then publishes them; the
// reader's acquire load plus barrier does the reverse. One fence per block per step, not one per thread.
__device__ __forceinline__ void signal_step(unsigned* f, unsigned v) {   // all threads of the block
  __syncthreads();
  if (threadIdx.x == 0) asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(f), "r"(v) : "memory");
}
__device__ __forceinline__ void wait_step(const unsigned* f, unsigned v) {
  if (threadIdx.x == 0) {
    unsigned cur;
    do { asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(cur) : "l"(f) : "memory"); } while (cur < v);
  }
  __syncthreads();
}
// FUSED_UNROLL independent 16-byte loads per pointer are in flight per thread before the first
// store (default 4, the A100 tuning; -DFUSED_UNROLL=8 doubles the bytes in flight at a fixed grid).
#ifndef FUSED_UNROLL
#define FUSED_UNROLL 4
#endif
__device__ __forceinline__ void add_slice(const float* __restrict__ mine, const float* __restrict__ from,
                                          float* __restrict__ dst, int n, int t0, int stride) {
  constexpr int U = FUSED_UNROLL;
  if ((n & 3) == 0) {
    const float4* a4 = (const float4*)mine; const float4* b4 = (const float4*)from; float4* c4 = (float4*)dst;
    int n4 = n >> 2, i = t0;
    for (; i + (U - 1) * stride < n4; i += U * stride) {
      float4 a[U], b[U];
#pragma unroll
      for (int u = 0; u < U; u++) a[u] = a4[i + u * stride];
#pragma unroll
      for (int u = 0; u < U; u++) b[u] = __ldcg(b4 + i + u * stride);
#pragma unroll
      for (int u = 0; u < U; u++) c4[i + u * stride] = f4add(a[u], b[u]);
    }
    for (; i < n4; i += stride) c4[i] = f4add(a4[i], __ldcg(b4 + i));
  } else {
    for (int i = t0; i < n; i += stride) dst[i] = mine[i] + __ldcg(from + i);
  }
}
__device__ __forceinline__ void copy_slice(const float* __restrict__ from, float* __restrict__ dst, int n, int t0, int stride) {
  constexpr int U = FUSED_UNROLL;
  if ((n & 3) == 0) {
    const float4* b4 = (const float4*)from; float4* c4 = (float4*)dst;
    int n4 = n >> 2, i = t0;
    for (; i + (U - 1) * stride < n4; i += U * stride) {
      float4 b[U];
#pragma unroll
      for (int u = 0; u < U; u++) b[u] = __ldcg(b4 + i + u * stride);
#pragma unroll
      for (int u = 0; u < U; u++) c4[i + u * stride] = b[u];
    }
    for (; i < n4; i += stride) c4[i] = __ldcg(b4 + i);
  } else {
    for (int i = t0; i < n; i += stride) dst[i] = __ldcg(from + i);
  }
}

// Each step is split into K sub-chunks with their own flag count, so the neighbour can start on
// sub-chunk 0 while this GPU is still working on sub-chunk K-1 (RING_PIPE, NCCL's steps-per-chunk).
// Flags count sub-steps: base = calls so far * 2(P-1)*K.
__global__ void ring_fused_kernel(FusedArgs a, int me, int P, int N, unsigned base, int K) {
  const int b = blockIdx.x, src = (me - 1 + P) % P, nxt = (me + 1) % P;
  const int stride = gridDim.x * blockDim.x, t0 = threadIdx.x + b * blockDim.x;
  unsigned* myflag = a.flag[me] + b;
  const unsigned* srcflag = a.flag[src] + b;
  const unsigned* nxtflag = a.flag[nxt] + b;
  wait_step(nxtflag, base);                 // the GPU that reads my buffers has finished the previous call
  for (int st = 0; st < P - 1; st++) {      // reduce-scatter: add the neighbour's chunk into mine
    int c = ((me - st - 1) % P + P) % P;
    long o = (long)N * c / P; int n = (int)((long)N * (c + 1) / P - o);
    const float* from = (st == 0 ? a.in[src] : a.out[src]) + o;
    for (int k = 0; k < K; k++) {
      long so = (long)n * k / K; int sn = (int)((long)n * (k + 1) / K - so);
      if (st > 0) wait_step(srcflag, base + (st - 1) * K + k + 1);
      add_slice(a.in[me] + o + so, from + so, a.out[me] + o + so, sn, t0, stride);
      signal_step(myflag, base + st * K + k + 1);
    }
  }
  for (int st = 0; st < P - 1; st++) {      // all-gather: copy the neighbour's finished chunk
    int c = ((me - st) % P + P) % P;
    long o = (long)N * c / P; int n = (int)((long)N * (c + 1) / P - o);
    for (int k = 0; k < K; k++) {
      long so = (long)n * k / K; int sn = (int)((long)n * (k + 1) / K - so);
      wait_step(srcflag, base + (P - 2 + st) * K + k + 1);
      if (st > 0) wait_step(nxtflag, base + st * K + k + 1);   // nxt has read this sub-chunk's partial sum
      copy_slice(a.out[src] + o + so, a.out[me] + o + so, sn, t0, stride);
      signal_step(myflag, base + (P - 1 + st) * K + k + 1);
    }
  }
}

// Small-message path, NCCL's "LL" idea: the sender writes 16-byte words {data, flag, data, flag}
// into a slot in the receiver's memory, so the flag travels with the data in one atomic store and the
// receiver just polls the words it needs. No fences, no separate flags, no block synchronisation; the
// price is 50% payload, so it is only used while a chunk is at most LLCAP floats. One slot per step.
struct LLArgs { const float* in[MAXP]; float* out[MAXP]; uint4* ll[MAXP]; unsigned* flag[MAXP]; };

// 16-byte volatile stores/loads as PTX: CUDA's vector types cannot be assigned through a volatile
// pointer, and the store must be one instruction so data and flag land together.
__device__ __forceinline__ void st_v4(uint4* p, unsigned x, unsigned y, unsigned z, unsigned w) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
__device__ __forceinline__ uint4 ld_v4(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void ll_send(uint4* slot, const float* __restrict__ src, int n2, unsigned flag, int t0, int stride) {
  const float2* s2 = (const float2*)src;
  for (int i = t0; i < n2; i += stride) {
    float2 v = s2[i];
    st_v4(slot + i, __float_as_uint(v.x), flag, __float_as_uint(v.y), flag);
  }
}
__device__ __forceinline__ float2 ll_recv(const uint4* slot, int i, unsigned flag) {
  uint4 w;
  do { w = ld_v4(slot + i); } while (w.y != flag || w.w != flag);
  return make_float2(__uint_as_float(w.x), __uint_as_float(w.z));
}

__global__ void ring_fused_ll_kernel(LLArgs a, int me, int P, int N, unsigned base, int cap2, int K) {
  const int b = blockIdx.x, nxt = (me + 1) % P;
  const int stride = gridDim.x * blockDim.x, t0 = threadIdx.x + b * blockDim.x;
  wait_step(a.flag[nxt] + b, base);         // nxt has finished the previous call: its slots may be reused
  for (int st = 0; st < P - 1; st++) {      // reduce-scatter
    int cs = ((me - st) % P + P) % P, cr = ((me - st - 1) % P + P) % P;
    long os = (long)N * cs / P, orr = (long)N * cr / P;
    int n2 = (int)((long)N * (cr + 1) / P - orr) / 2;
    const float* sendp = (st == 0 ? a.in[me] : a.out[me]) + os;
    ll_send(a.ll[nxt] + (long)st * cap2, sendp, (int)((long)N * (cs + 1) / P - os) / 2, base + st + 1, t0, stride);
    const uint4* slot = a.ll[me] + (long)st * cap2;
    const float2* mine = (const float2*)(a.in[me] + orr); float2* dst = (float2*)(a.out[me] + orr);
    for (int i = t0; i < n2; i += stride) {
      float2 r = ll_recv(slot, i, base + st + 1), m = mine[i];
      dst[i] = make_float2(m.x + r.x, m.y + r.y);
    }
  }
  for (int st = 0; st < P - 1; st++) {      // all-gather
    int cs = ((me + 1 - st) % P + P) % P, cr = ((me - st) % P + P) % P;
    long os = (long)N * cs / P, orr = (long)N * cr / P;
    int n2 = (int)((long)N * (cr + 1) / P - orr) / 2;
    ll_send(a.ll[nxt] + (long)(P - 1 + st) * cap2, a.out[me] + os, (int)((long)N * (cs + 1) / P - os) / 2, base + P + st, t0, stride);
    const uint4* slot = a.ll[me] + (long)(P - 1 + st) * cap2;
    float2* dst = (float2*)(a.out[me] + orr);
    for (int i = t0; i < n2; i += stride) dst[i] = ll_recv(slot, i, base + P + st);
  }
  signal_step(a.flag[me] + b, base + (2 * P - 2) * K);   // done: lets nxt's next call reuse my slots
}

