#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace bif {

// Collapse a vector load to a scalar so the accumulation cannot be elided.
__device__ __forceinline__ float reduce_vec(float  v) { return v; }
__device__ __forceinline__ float reduce_vec(float2 v) { return v.x + v.y; }
__device__ __forceinline__ float reduce_vec(float4 v) { return v.x + v.y + v.z + v.w; }

// Streaming-read microbenchmark kernel.
//
//   V    : access vector type (float / float2 / float4) => 4 / 8 / 16 bytes
//   ILP  : independent in-flight loads issued per thread per iteration
//
// The grid is launched as a single wave of `resident_threads` threads so every
// launched warp is simultaneously resident; `resident_threads` is the
// Little's-Law concurrency knob. Each thread runs `iters` iterations, each
// issuing ILP independent loads spaced `stride` vector-elements apart, wrapping
// inside a power-of-two buffer of `n_vec` elements so any stride stays in bounds.
//
// Useful bytes moved (host-side accounting):
//     iters * resident_threads * ILP * sizeof(V)
// Bytes in flight (occupancy-curve x-axis):
//     resident_threads * ILP * sizeof(V)
template <typename V, int ILP>
__global__ void __launch_bounds__(256)
stream_read(const V* __restrict__ in,
            float* __restrict__ sink,
            size_t   n_vec,           // power of two
            uint32_t stride,
            uint32_t iters,
            size_t   resident_threads,
            float    poison)
{
    const size_t tid  = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t mask = n_vec - 1;
    size_t base = tid * ILP * (size_t)stride;
    float acc = 0.0f;

    for (uint32_t it = 0; it < iters; ++it) {
        V vals[ILP];
        #pragma unroll
        for (int i = 0; i < ILP; ++i) {
            const size_t idx = (base + (size_t)i * stride) & mask;
            vals[i] = in[idx];
        }
        #pragma unroll
        for (int i = 0; i < ILP; ++i) acc += reduce_vec(vals[i]);
        base = (base + resident_threads * ILP * (size_t)stride) & mask;
    }

    // Never true at runtime (`poison` is a host-chosen sentinel); defeats DCE.
    if (acc == poison) sink[tid] = acc;
}

} // namespace bif
