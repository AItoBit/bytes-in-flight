// bytes-in-flight -- HBM bandwidth ladder microbenchmark.
//
// Sweeps stride, access vector width, resident warps, and per-thread ILP; writes
// one CSV row per configuration. See README.md for the schema and
// docs/methodology.md for the model.

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kernels.cuh"

#define CUDA_OK(x)                                                              \
    do {                                                                       \
        cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(e_), __FILE__, __LINE__);              \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

static std::vector<uint32_t> parse_list(const char* s) {
    std::vector<uint32_t> v;
    std::string cur;
    for (const char* p = s;; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!cur.empty()) {
                v.push_back((uint32_t)strtoul(cur.c_str(), nullptr, 10));
                cur.clear();
            }
            if (*p == '\0') break;
        } else {
            cur += *p;
        }
    }
    return v;
}

static size_t pow2_le(size_t x) {
    size_t p = 1;
    while (p * 2 <= x) p *= 2;
    return p;
}

__global__ void fill_kernel(float* p, size_t n, float val) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        p[i] = val;
}

template <typename V, int ILP>
static float run_once(const V* in, float* sink, size_t n_vec, uint32_t stride,
                      uint32_t iters, size_t threads, int repeats) {
    const int block = 256;
    const unsigned grid = (unsigned)((threads + block - 1) / block);
    const float poison = -3.14159e30f;

    cudaEvent_t a, b;
    CUDA_OK(cudaEventCreate(&a));
    CUDA_OK(cudaEventCreate(&b));

    for (int w = 0; w < 3; ++w)  // warmup
        bif::stream_read<V, ILP>
            <<<grid, block>>>(in, sink, n_vec, stride, iters, threads, poison);
    CUDA_OK(cudaDeviceSynchronize());

    float best = 1e30f;
    for (int r = 0; r < repeats; ++r) {
        CUDA_OK(cudaEventRecord(a));
        bif::stream_read<V, ILP>
            <<<grid, block>>>(in, sink, n_vec, stride, iters, threads, poison);
        CUDA_OK(cudaEventRecord(b));
        CUDA_OK(cudaEventSynchronize(b));
        float ms = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&ms, a, b));
        best = std::min(best, ms);
    }
    CUDA_OK(cudaGetLastError());
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return best;
}

template <typename V>
static float dispatch_ilp(int ilp, const V* in, float* sink, size_t n_vec,
                          uint32_t stride, uint32_t iters, size_t threads,
                          int repeats) {
    switch (ilp) {
        case 1: return run_once<V, 1>(in, sink, n_vec, stride, iters, threads, repeats);
        case 2: return run_once<V, 2>(in, sink, n_vec, stride, iters, threads, repeats);
        case 4: return run_once<V, 4>(in, sink, n_vec, stride, iters, threads, repeats);
        case 8: return run_once<V, 8>(in, sink, n_vec, stride, iters, threads, repeats);
        default: fprintf(stderr, "ilp must be one of 1,2,4,8\n"); exit(1);
    }
}

static float dispatch(int vec_bytes, int ilp, const void* in, float* sink,
                      size_t n_vec, uint32_t stride, uint32_t iters,
                      size_t threads, int repeats) {
    switch (vec_bytes) {
        case 4:
            return dispatch_ilp<float>(ilp, (const float*)in, sink, n_vec,
                                       stride, iters, threads, repeats);
        case 8:
            return dispatch_ilp<float2>(ilp, (const float2*)in, sink, n_vec,
                                        stride, iters, threads, repeats);
        case 16:
            return dispatch_ilp<float4>(ilp, (const float4*)in, sink, n_vec,
                                        stride, iters, threads, repeats);
        default:
            fprintf(stderr, "vec must be 4, 8 or 16 bytes\n");
            exit(1);
    }
}

int main(int argc, char** argv) {
    int gpu = 0;
    size_t footprint = (size_t)1 << 31;  // 2 GiB
    uint32_t iters = 512;
    int repeats = 20;
    double spec_override = 0.0;
    const char* csv = "data/run.csv";
    std::vector<uint32_t> vecs = {16}, strides = {1}, warp_list = {0}, ilps = {4};

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* n) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", n); exit(1); }
            return argv[++i];
        };
        if (!strcmp(argv[i], "--gpu")) gpu = atoi(need("--gpu"));
        else if (!strcmp(argv[i], "--footprint-bytes")) footprint = strtoull(need("--footprint-bytes"), nullptr, 10);
        else if (!strcmp(argv[i], "--iters")) iters = (uint32_t)strtoul(need("--iters"), nullptr, 10);
        else if (!strcmp(argv[i], "--repeats")) repeats = atoi(need("--repeats"));
        else if (!strcmp(argv[i], "--spec-gbps")) spec_override = atof(need("--spec-gbps"));
        else if (!strcmp(argv[i], "--csv")) csv = need("--csv");
        else if (!strcmp(argv[i], "--vec")) vecs = parse_list(need("--vec"));
        else if (!strcmp(argv[i], "--stride")) strides = parse_list(need("--stride"));
        else if (!strcmp(argv[i], "--warps")) warp_list = parse_list(need("--warps"));
        else if (!strcmp(argv[i], "--ilp")) ilps = parse_list(need("--ilp"));
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("usage: bif [--gpu N] [--footprint-bytes B] [--iters N] "
                   "[--repeats N] [--spec-gbps X] [--csv PATH]\n"
                   "           [--vec 4,8,16] [--stride L1,L2,...] "
                   "[--warps W1,W2,... | 0] [--ilp 1,2,4,8]\n"
                   "see README.md for details\n");
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    CUDA_OK(cudaSetDevice(gpu));
    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, gpu));

    int mem_clk_khz = 0, bus_bits = 0;
    CUDA_OK(cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, gpu));
    CUDA_OK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, gpu));
    // DDR: two transfers per memory clock.
    double spec_bps = 2.0 * ((double)mem_clk_khz * 1e3) * (bus_bits / 8.0);
    double spec_gbps = spec_override > 0.0 ? spec_override : spec_bps / 1e9;

    size_t n_float = pow2_le(footprint / sizeof(float));
    size_t buf_bytes = n_float * sizeof(float);

    size_t max_resident =
        (size_t)prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;

    if (warp_list.size() == 1 && warp_list[0] == 0) {
        warp_list.clear();
        for (size_t w = 1; w * 32 <= max_resident; w *= 2)
            warp_list.push_back((uint32_t)w);
        if (warp_list.empty() || (size_t)warp_list.back() * 32 != max_resident)
            warp_list.push_back((uint32_t)(max_resident / 32));
    }

    fprintf(stderr, "device %d: %s\n", gpu, prop.name);
    fprintf(stderr, "  SMs=%d  maxThreads/SM=%d  bus=%d-bit  memClk=%.0f MHz\n",
            prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor, bus_bits,
            mem_clk_khz / 1000.0);
    fprintf(stderr, "  peak spec = %.1f GB/s%s\n", spec_gbps,
            spec_override > 0.0 ? " (override)" : "");
    fprintf(stderr, "  buffer = %.2f GiB (%zu floats)\n",
            buf_bytes / 1073741824.0, n_float);
    fprintf(stderr, "  max resident threads = %zu (%zu warps)\n", max_resident,
            max_resident / 32);

    float* buf = nullptr;
    CUDA_OK(cudaMalloc(&buf, buf_bytes));
    {
        int b = 256;
        int g = (int)std::min<size_t>((n_float + b - 1) / b, 65535);
        fill_kernel<<<g, b>>>(buf, n_float, 1.0f);
        CUDA_OK(cudaDeviceSynchronize());
    }

    size_t max_threads = 0;
    for (uint32_t w : warp_list)
        max_threads = std::max(max_threads, (((size_t)w * 32 + 255) / 256) * 256);
    float* sink = nullptr;
    CUDA_OK(cudaMalloc(&sink, max_threads * sizeof(float)));

    FILE* probe = fopen(csv, "r");
    bool existed = probe != nullptr;
    if (probe) fclose(probe);
    FILE* out = fopen(csv, "a");
    if (!out) { fprintf(stderr, "cannot open %s for append\n", csv); return 1; }
    if (!existed)
        fprintf(out,
                "gpu_name,vec_bytes,stride,resident_warps,ilp,footprint_bytes,"
                "iters,bytes_moved,time_ms,achieved_gbps,spec_gbps,pct_of_spec,"
                "bytes_in_flight\n");

    for (uint32_t vb : vecs)
        for (uint32_t st : strides)
            for (uint32_t wl : warp_list)
                for (uint32_t il : ilps) {
                    size_t n_vec = n_float / (vb / 4);  // buffer as V-vectors
                    size_t threads = (((size_t)wl * 32 + 255) / 256) * 256;
                    if (threads > max_resident)
                        fprintf(stderr,
                                "  [warn] %zu threads > %zu resident: multi-wave, "
                                "occupancy-curve x-axis is approximate\n",
                                threads, max_resident);

                    float ms = dispatch((int)vb, (int)il, buf, sink, n_vec, st,
                                        iters, threads, repeats);

                    double bytes_moved =
                        (double)iters * (double)threads * il * vb;
                    double gbps = bytes_moved / (ms * 1e-3) / 1e9;
                    double bif_bytes = (double)threads * il * vb;
                    double pct = spec_gbps > 0.0 ? 100.0 * gbps / spec_gbps : 0.0;

                    fprintf(out,
                            "\"%s\",%u,%u,%zu,%u,%zu,%u,%.0f,%.5f,%.3f,%.3f,"
                            "%.2f,%.0f\n",
                            prop.name, vb, st, threads / 32, il, footprint, iters,
                            bytes_moved, ms, gbps, spec_gbps, pct, bif_bytes);
                    fflush(out);

                    fprintf(stderr,
                            "  vec=%2u stride=%4u warps=%5zu ilp=%u -> %9.1f GB/s "
                            "(%.1f%% spec)\n",
                            vb, st, threads / 32, il, gbps, pct);
                }

    fclose(out);
    cudaFree(buf);
    cudaFree(sink);
    return 0;
}
