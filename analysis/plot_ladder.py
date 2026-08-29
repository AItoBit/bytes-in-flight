#!/usr/bin/env python3
"""Plot the HBM bandwidth ladder: achieved GB/s vs stride, and vs access width.

    python analysis/plot_ladder.py [data/run.csv]

Writes data/ladder_stride.png and data/ladder_vec.png.
"""
import os
import sys

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(path: str) -> None:
    df = pd.read_csv(path, comment="#")
    spec = float(df["spec_gbps"].iloc[0])
    os.makedirs("data", exist_ok=True)

    # Reference point: highest occupancy present, and its most common ILP.
    wl = int(df["resident_warps"].max())
    il = int(df.loc[df["resident_warps"] == wl, "ilp"].mode().iloc[0])

    # ---- bandwidth vs stride, one line per access width -------------------
    s = df[(df["resident_warps"] == wl) & (df["ilp"] == il)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for vb, g in sorted(s.groupby("vec_bytes")):
        g = g.sort_values("stride")
        ax.plot(g["stride"], g["achieved_gbps"], marker="o",
                label=f"{int(vb)} B/access")
    ax.axhline(spec, ls="--", color="k", lw=1, label=f"spec {spec:.0f} GB/s")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("stride (vector elements between a thread's loads)")
    ax.set_ylabel("achieved bandwidth (GB/s)")
    ax.set_title(f"HBM bandwidth vs stride   (warps={wl}, ILP={il})")
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("data/ladder_stride.png", dpi=130)
    print("wrote data/ladder_stride.png")

    # ---- bandwidth vs access width at stride 1 ---------------------------
    v = (df[(df["stride"] == 1) & (df["resident_warps"] == wl)]
         .groupby("vec_bytes", as_index=False)["achieved_gbps"].max()
         .sort_values("vec_bytes"))
    fig, ax = plt.subplots(figsize=(5, 4))
    xs = [str(int(x)) for x in v["vec_bytes"]]
    ax.bar(xs, v["achieved_gbps"])
    ax.axhline(spec, ls="--", color="k", lw=1, label=f"spec {spec:.0f} GB/s")
    for i, val in enumerate(v["achieved_gbps"]):
        ax.text(i, val, f"{val:.0f}\n{100 * val / spec:.0f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("bytes per access")
    ax.set_ylabel("achieved bandwidth (GB/s)")
    ax.set_title("HBM bandwidth vs access width (stride 1)")
    ax.set_ylim(0, max(spec, v["achieved_gbps"].max()) * 1.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig("data/ladder_vec.png", dpi=130)
    print("wrote data/ladder_vec.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/run.csv")
