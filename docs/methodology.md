# Methodology

## What is being measured

`bif` runs one CUDA kernel, `stream_read<V, ILP>`, that does nothing but issue
independent global loads and accumulate them. Four knobs are swept:

| knob | mechanism | what it exposes |
|---|---|---|
| **stride** | vector elements between a thread's consecutive loads | DRAM burst / coalescing efficiency |
| **vector width** (`V`) | `float` / `float2` / `float4` = 4 / 8 / 16 B per load | wide-access efficiency, LSU throughput |
| **resident warps** | size of the single launched wave | Little's-Law concurrency |
| **ILP** | independent loads per thread per iteration (`#pragma unroll`) | bytes in flight per warp |

## Peak spec

```
spec_GBps = 2 * mem_clock_Hz * (bus_width_bits / 8) / 1e9
```

`mem_clock_Hz` and `bus_width_bits` come from
`cudaDeviceGetAttribute(cudaDevAttrMemoryClockRate | cudaDevAttrGlobalMemoryBusWidth)`.
The factor 2 is the DDR double-pump. On HBM parts the driver already reports the
per-pin data rate as the "memory clock", so this yields the datasheet aggregate
(e.g. H100 SXM ~3.35 TB/s, H200 ~4.8 TB/s). If the computed value disagrees with
the datasheet for your board, pass `--spec-gbps` explicitly; every `pct_of_spec`
is then relative to that.

## The resident-warp model

The grid is launched as **one wave**: `grid.x * 256 == resident_threads`, capped
so that `resident_threads <= SMs * maxThreadsPerSM`. Under that cap every
launched warp is resident at once, so "warps launched" == "warps in flight". The
default `--warps 0` expands to `1, 2, 4, ... , full_occupancy`.

Beyond the cap the launch becomes multi-wave; `bif` still runs but prints a
`[warn]` and the occupancy-curve x-axis becomes approximate. At very high `ILP`
(8) with `float4`, register pressure can also push the real resident count below
the launched count — keep `ILP <= 4` for the canonical curve.

## Byte accounting

Useful bytes are counted host-side, independent of stride:

```
bytes_moved     = iters * resident_threads * ILP * sizeof(V)
achieved_GBps   = bytes_moved / kernel_time
bytes_in_flight = resident_threads * ILP * sizeof(V)
```

For `stride > 1` the DRAM still moves whole bursts, so `achieved_GBps` drops
below spec — that gap *is* the stride-efficiency result.

## Little's Law

For a pipeline, `throughput = concurrency / latency`. For memory:

```
achieved_BW(C) = C / latency          for small C
achieved_BW(C) -> BW_max              once the memory system saturates
```

`plot_littles_law.py` fits the saturating form

```
BW(C) = BW_max * C / (C + C_knee)
```

by least squares on `1/BW` vs `1/C` (it is linear there), then reports

```
latency = C_knee / BW_max        # bytes / (bytes/ns) = ns
```

`C_knee` is how many bytes you must keep in flight to sit at the knee;
translate it back to occupancy with `warps = C_knee / (32 * ILP * sizeof(V))`.

## Caveats

- **L2 reuse.** The kernel wraps inside a power-of-two buffer. At full occupancy
  the defaults (2 GiB buffer, `iters=512`) make one sweep ~= one buffer pass, so
  reuse is minimal; at low occupancy the sweep is shorter and purely cold. Keep
  the buffer many times the L2 size (H100 L2 = 50 MB, H200 = 60 MB). Raise
  `--footprint-bytes` / `--iters` if a run looks implausibly fast.
- **Clock throttling.** Lock clocks: `nvidia-smi -pm 1` then
  `nvidia-smi -lgc <clock>` (`nvidia-smi -q -d SUPPORTED_CLOCKS` lists them).
  Watch for thermal/power capping on long sweeps.
- **ECC.** Enabled ECC costs a few percent of HBM bandwidth. Report its state
  (`nvidia-smi -q -d ECC`) alongside results.
- **Windows.** Use the GPU in **TCC** mode (`nvidia-smi -dm TCC`) where possible;
  under **WDDM** the OS scheduler adds launch jitter — raise `--repeats`.
- **Timing.** `bif` reports the *minimum* kernel time over `--repeats` (best
  case). Increase `--repeats` on noisy systems.
- **Read-only.** This measures the read path. A write / copy variant would show a
  different ceiling; not included here.

## References

- J. D. C. Little, "A Proof for the Queuing Formula L = λW," *Operations
  Research*, 1961.
- V. Volkov, "Understanding Latency Hiding on GPUs," PhD thesis, UC Berkeley,
  2016.
- NVIDIA H100 / H200 datasheets for HBM3 / HBM3e aggregate bandwidth.
