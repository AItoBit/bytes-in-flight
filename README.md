# bytes-in-flight

An **HBM bandwidth ladder**: a CUDA microbenchmark that measures *achievable*
HBM3 / HBM3e bandwidth versus datasheet spec while sweeping

- **stride** — vector elements between consecutive per-thread accesses (DRAM burst / coalescing efficiency)
- **vector width** — 4 B (`float`), 8 B (`float2`), 16 B (`float4`) per access
- **in-flight warps** — resident warps launched as a single wave (the Little's-Law concurrency knob)
- **ILP** — independent loads issued per thread per iteration (more bytes in flight per warp)

and publishes the **Little's-Law occupancy curve**: achieved bandwidth as a
function of bytes in flight, with the saturation knee and the implied memory
latency marked.

> **Little's Law for a memory pipeline:** `achieved_BW = bytes_in_flight / latency`.
> Bandwidth climbs almost linearly with concurrency until the memory system
> saturates at `BW_max`; the knee tells you how many bytes you must keep in
> flight (occupancy x ILP x access width) to reach peak, and
> `latency = knee_bytes / BW_max`.

## Requirements

- NVIDIA GPU — tuned for HBM parts (Hopper `sm_90`, Blackwell `sm_100`), runs on any arch
- CUDA Toolkit 12.x or 13.x (`nvcc`)
- Python 3.9+ for analysis: `pip install -r analysis/requirements.txt`

## Build

```bash
make                 # -arch=native (needs a visible GPU at build time)
make ARCH=sm_90      # Hopper: H100 / H200 (HBM3 / HBM3e)
make ARCH=sm_100     # Blackwell
```

Windows, from a CUDA-enabled *x64 Native Tools* prompt:

```bat
nvcc -O3 -std=c++17 -arch=sm_90 src\main.cu -o bin\bif.exe
```

## Run

Pin clocks first so the ladder is reproducible:

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc <base_graphics_clock>   # see: nvidia-smi -q -d SUPPORTED_CLOCKS
```

Full sweep (writes `data/run.csv`):

```bash
bash scripts/run_sweep.sh
# or on Windows
pwsh scripts/run_sweep.ps1
```

Single point:

```bash
./bin/bif --gpu 0 --footprint-bytes 2147483648 --iters 512 \
          --vec 16 --stride 1 --warps 1024 --ilp 4 --repeats 20 --csv data/run.csv
```

`--warps 0` expands to a power-of-two warp ladder up to full device occupancy.

## Analyze

```bash
python analysis/plot_ladder.py       data/run.csv   # data/ladder_stride.png, data/ladder_vec.png
python analysis/plot_littles_law.py  data/run.csv   # data/littles_law.png  (+ latency estimate)
python analysis/report.py            data/run.csv   # data/REPORT.md
```

## Try the pipeline without a GPU

```bash
python scripts/synthesize_sample.py                 # data/sample_synthetic.csv  (SYNTHETIC)
python analysis/plot_littles_law.py data/sample_synthetic.csv
```

`data/sample_synthetic.csv` comes from a toy Little's-Law model, **not hardware**.
Never cite it as a measurement.

## CSV schema

| column | meaning |
|---|---|
| `gpu_name` | `cudaDeviceProp::name` |
| `vec_bytes` | 4 / 8 / 16 — bytes per access |
| `stride` | vector elements between a thread's consecutive loads |
| `resident_warps` | warps launched in the single wave |
| `ilp` | independent loads per thread per iteration |
| `footprint_bytes` | requested buffer size (rounded down to a power of two) |
| `iters` | loop iterations per thread |
| `bytes_moved` | useful bytes read = `iters x threads x ilp x vec_bytes` |
| `time_ms` | best (min) kernel time over `--repeats` |
| `achieved_gbps` | `bytes_moved / time` |
| `spec_gbps` | computed peak (`2 x memClk x busWidth / 8`) or `--spec-gbps` override |
| `pct_of_spec` | `100 x achieved / spec` |
| `bytes_in_flight` | `threads x ilp x vec_bytes` — the occupancy-curve x-axis |

See [docs/methodology.md](docs/methodology.md) for the resident-warp model, how
peak spec is derived, the Little's-Law fit, and the measurement caveats
(L2 reuse, ECC, clock throttling, WDDM vs TCC).

## License

MIT — see [LICENSE](LICENSE).
