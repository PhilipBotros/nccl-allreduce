"""Plot the CUDA all-reduce sweeps in results/cuda_*.csv into docs/.

    python3 cuda/plot.py results/cuda_copy_reduce_2xT4_p2p.csv results/cuda_copy_reduce_2xT4_nop2p.csv

Colour is the implementation, line style is peer access on (solid) or off (dashed).
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
REF = "#9b9a95"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT, "xtick.color": TEXT_2,
    "ytick.color": TEXT_2, "text.color": TEXT, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})

PCIE3_X16_GBPS = 15.75  # PCIe 3.0 x16 raw spec, one direction; not a measurement
IMPL_LABEL = {"copy_sync": "copy, sync", "copy_pull": "copy, async pull", "copy_push": "copy, async push",
              "remote": "remote loads, no copy",
              "reduce_scatter": "reduce-scatter + all-gather", "nccl": "NCCL"}
IMPL_ORDER = ["copy_sync", "copy_pull", "copy_push", "remote", "reduce_scatter", "nccl"]


def load(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(l for l in f if not l.startswith("#"))]
    for r in rows:
        for k in ("bytes", "median_us", "p10_us", "p90_us", "GBps"):
            r[k] = float(r[k])
    return rows


def size_label(b):
    return f"{int(b // 1024)} KB" if b < 1 << 20 else f"{int(b // (1 << 20))} MB"


def main(paths):
    rows = [r for p in paths for r in load(p)]
    gpu = rows[0]["gpu"]
    sizes = sorted({r["bytes"] for r in rows})
    series = []  # (label, colour, linestyle, rows) in fixed order
    for impl, color in zip(IMPL_ORDER, SERIES):
        for p2p, ls, tag in (("1", "-", "peer access on"), ("0", "--", "peer access off")):
            rs = sorted((r for r in rows if r["impl"] == impl and r["p2p"] == p2p), key=lambda r: r["bytes"])
            if rs:
                series.append((f"{IMPL_LABEL.get(impl, impl)}, {tag}", color, ls, rs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.suptitle(f"Two-GPU all-reduce on 2x {gpu}: three hand-written variants vs NCCL, peer access on (solid) / off (dashed)",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")

    for label, color, ls, rs in series:
        x = [r["bytes"] for r in rs]
        ax1.plot(x, [r["median_us"] for r in rs], color=color, ls=ls, lw=2, marker="o", ms=5, label=label)
        ax2.plot(x, [r["GBps"] for r in rs], color=color, ls=ls, lw=2, marker="o", ms=5, label=label)

    # direct labels: fastest and slowest at the small end (latency), all at the large end (bandwidth)
    small = sorted(series, key=lambda s: s[3][0]["median_us"])
    for label, color, ls, rs in (small[0], small[-1]):
        ax1.annotate(f"{rs[0]['median_us']:.0f} us", (rs[0]["bytes"], rs[0]["median_us"]),
                     textcoords="offset points", xytext=(-7, 0), ha="right", va="center",
                     fontsize=8, color=TEXT_2)
    big = sorted(series, key=lambda s: s[3][-1]["GBps"])
    for i, (label, color, ls, rs) in enumerate(big):
        ax2.annotate(f"{rs[-1]['GBps']:.2f}", (rs[-1]["bytes"], rs[-1]["GBps"]),
                     textcoords="offset points", xytext=(8, -14 + 7 * i), ha="left", va="center",
                     fontsize=8, color=TEXT_2)

    ax1.set_xscale("log", base=2); ax1.set_yscale("log")
    ax1.set_xticks(sizes); ax1.set_xticklabels([size_label(s) for s in sizes], rotation=45, ha="right")
    ax1.set_xlim(sizes[0] * 0.5, sizes[-1] * 1.5)
    ax1.set_xlabel("Tensor size per GPU (fp32)")
    ax1.set_ylabel("All-reduce latency (us, median)")
    ax1.set_title("Latency", loc="left", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, handlelength=3.5)

    ax2.axhline(PCIE3_X16_GBPS, color=REF, lw=1.2, ls="--")
    ax2.text(sizes[0], PCIE3_X16_GBPS, f"  PCIe 3.0 x16 spec, one direction: {PCIE3_X16_GBPS} GB/s",
             color=TEXT_2, va="bottom", fontsize=9)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(sizes); ax2.set_xticklabels([size_label(s) for s in sizes], rotation=45, ha="right")
    ax2.set_xlim(sizes[0] * 0.7, sizes[-1] * 2.4)
    ax2.set_ylim(0, PCIE3_X16_GBPS * 1.15)
    ax2.set_xlabel("Tensor size per GPU (fp32)")
    ax2.set_ylabel("Algorithm bandwidth, bytes / latency (GB/s)")
    ax2.set_title("Bandwidth", loc="left", fontsize=10)
    ax2.legend(loc="center left", fontsize=8, handlelength=3.5)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs",
                       f"cuda_copy_reduce_2x{gpu.replace('Tesla ', '')}.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
