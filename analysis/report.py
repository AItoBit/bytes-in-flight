#!/usr/bin/env python3
"""Assemble data/REPORT.md from a sweep CSV plus any generated PNGs.

    python analysis/report.py [data/run.csv]
"""
import os
import sys

import pandas as pd


def main(path: str) -> None:
    df = pd.read_csv(path, comment="#")
    spec = float(df["spec_gbps"].iloc[0])
    name = str(df["gpu_name"].iloc[0])
    wl = int(df["resident_warps"].max())
    peak = df.loc[df["achieved_gbps"].idxmax()]

    out = ["# bytes-in-flight report -- " + name, ""]
    out.append(f"- datasheet / spec peak: **{spec:.0f} GB/s**")
    out.append(
        f"- best achieved: **{peak.achieved_gbps:.0f} GB/s** "
        f"({100 * peak.achieved_gbps / spec:.1f}% of spec) at "
        f"vec={int(peak.vec_bytes)} B, stride={int(peak.stride)}, "
        f"warps={int(peak.resident_warps)}, ILP={int(peak.ilp)}"
    )
    out.append("")

    st = df[df["resident_warps"] == wl].sort_values(["vec_bytes", "stride"])
    out.append(f"## Stride ladder (warps={wl})")
    out.append("")
    out.append("| vec B | stride | GB/s | % spec |")
    out.append("|---:|---:|---:|---:|")
    for _, r in st.iterrows():
        out.append(
            f"| {int(r.vec_bytes)} | {int(r.stride)} | "
            f"{r.achieved_gbps:.0f} | {r.pct_of_spec:.1f} |"
        )
    out.append("")

    for img in ("ladder_stride.png", "ladder_vec.png", "littles_law.png"):
        if os.path.exists(os.path.join("data", img)):
            out.append(f"![{img}]({img})")
            out.append("")

    dest = os.path.join("data", "REPORT.md")
    with open(dest, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("wrote", dest)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/run.csv")
