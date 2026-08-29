#!/usr/bin/env python3
"""Publish the Little's-Law occupancy curve: achieved BW vs bytes in flight.

    python analysis/plot_littles_law.py [data/run.csv]

Little's Law for a memory pipeline:  achieved_BW = bytes_in_flight / latency.
Bandwidth rises ~linearly with concurrency, then saturates at BW_max. We fit the
saturating form

    BW(C) = BW_max * C / (C + C_knee)

which is linear in 1/C for 1/BW:

    1/BW = 1/BW_max + (C_knee / BW_max) * (1/C)

and report BW_max, the knee concurrency C_knee, and the implied latency

    latency = C_knee / BW_max        (bytes / (bytes/ns) = ns)

Writes data/littles_law.png.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(path: str) -> None:
    df = pd.read_csv(path, comment="#")
    d = df[df["stride"] == 1].copy()
    if d.empty:
        sys.exit("no stride==1 rows to fit the occupancy curve")
    d = d.sort_values("bytes_in_flight")

    C = d["bytes_in_flight"].to_numpy(float)      # bytes
    BW = d["achieved_gbps"].to_numpy(float)       # GB/s == bytes/ns
    spec = float(d["spec_gbps"].iloc[0])

    # Least-squares fit of 1/BW against 1/C.
    x = 1.0 / C
    y = 1.0 / BW
    A = np.vstack([np.ones_like(x), x]).T
    (b0, b1), *_ = np.linalg.lstsq(A, y, rcond=None)
    bw_max = 1.0 / b0
    c_knee = b1 * bw_max
    latency_ns = c_knee / bw_max

    grid = np.linspace(C.min(), C.max(), 240)
    model = bw_max * grid / (grid + c_knee)

    os.makedirs("data", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(C / 1024.0, BW, s=28, zorder=3, label="measured")
    ax.plot(grid / 1024.0, model, color="C1", zorder=2,
            label=f"fit  BW_max={bw_max:.0f} GB/s  knee={c_knee / 1024.0:.0f} KiB")
    ax.axhline(spec, ls="--", color="k", lw=1, label=f"spec {spec:.0f} GB/s")
    ax.axhline(bw_max, ls=":", color="C1", lw=1)
    ax.axvline(c_knee / 1024.0, ls=":", color="C1", lw=1)
    ax.set_xlabel("bytes in flight  (KiB)   =   resident_threads x ILP x access_bytes")
    ax.set_ylabel("achieved bandwidth (GB/s)")
    ax.set_title("Little's-Law occupancy curve  (stride 1)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.text(0.5, 0.005,
             f"implied memory latency  =  knee / BW_max  =  {latency_ns:.0f} ns",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig("data/littles_law.png", dpi=130)

    print(f"BW_max          ~= {bw_max:.0f} GB/s  ({100 * bw_max / spec:.0f}% of spec)")
    print(f"knee concurrency ~= {c_knee / 1024.0:.0f} KiB in flight")
    print(f"implied latency  ~= {latency_ns:.0f} ns")
    print("wrote data/littles_law.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/run.csv")
