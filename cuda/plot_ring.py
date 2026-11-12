"""Plot the P-GPU ring sweeps (cuda/ring.cu) as one column per machine.

    python3 cuda/plot_ring.py results/cuda_ring_2xT4.csv results/cuda_ring_4xL4.csv results/cuda_ring_8xA100.csv

Colour is the implementation; each column is one machine (GPU x count). The bandwidth row is
NCCL's bus bandwidth, algbw * 2(P-1)/P, so columns with different P are comparable.
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

IMPL_ORDER = ["copy_all", "remote_all", "ring", "ring_remote", "nccl", "ring_fused"]
IMPL_LABEL = {"copy_all": "copy all inputs + sum", "remote_all": "read all inputs remotely + sum",
              "ring": "ring, copies", "ring_remote": "ring, remote-read reduce", "nccl": "NCCL",
              "ring_fused": "ring, fused kernel (LL + flags)"}
# one-direction link spec per GPU, for the dashed reference line; not a measurement
LINK = {"Tesla T4": ("PCIe 3.0 x16", 15.75), "NVIDIA L4": ("PCIe 4.0 x16", 31.5), "NVIDIA A10G": ("PCIe 4.0 x16", 31.5),
        "NVIDIA A100-SXM4-40GB": ("NVLink 3, 12 links", 300.0), "NVIDIA A100-SXM4-80GB": ("NVLink 3, 12 links", 300.0)}


def load(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(l for l in f if not l.startswith("#"))]
    for r in rows:
        for k in ("bytes", "median_us", "algbw_GBps", "busbw_GBps"):
            r[k] = float(r[k])
        r["ngpus"] = int(r["ngpus"])
    return rows


def size_label(b):
    return f"{int(b // 1024)} KB" if b < 1 << 20 else f"{int(b // (1 << 20))} MB"


def spread(vals, min_gap):
    """Nudge label positions apart so none are closer than min_gap; keeps order."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    pos = [vals[i] for i in order]
    for _ in range(50):
        moved = False
        for a in range(1, len(pos)):
            if pos[a] - pos[a - 1] < min_gap:
                mid = (pos[a] + pos[a - 1]) / 2
                pos[a - 1], pos[a] = mid - min_gap / 2, mid + min_gap / 2
                moved = True
        if not moved:
            break
    out = [0.0] * len(vals)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


def main(paths):
    machines = [load(p) for p in paths]
    M = len(machines)
    fig, axes = plt.subplots(2, M, figsize=(4.4 * M + 1.2, 8.6), squeeze=False)
    fig.suptitle("All-reduce across P GPUs in one process:\nhand-written variants vs NCCL" if M == 1 else
                 "All-reduce across P GPUs in one process: hand-written variants vs NCCL",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    handles = {}
    for col, rows in enumerate(machines):
        gpu, P = rows[0]["gpu"], rows[0]["ngpus"]
        sizes = sorted({r["bytes"] for r in rows})
        link = LINK.get(gpu)
        title = f"{P}x {gpu.replace('NVIDIA ', '').replace('-SXM4', '')}" + (f", {link[0]}" if link else "")
        if rows[0]["p2p"] == "0":
            title += ", no peer access"
        for row, key, ylabel in ((0, "median_us", "latency (us, median)"), (1, "busbw_GBps", "bus bandwidth (GB/s)")):
            ax = axes[row][col]
            ends = []
            for impl, color in zip(IMPL_ORDER, SERIES):
                rs = sorted((r for r in rows if r["impl"] == impl), key=lambda r: r["bytes"])
                if not rs:
                    continue
                (h,) = ax.plot([r["bytes"] for r in rs], [r[key] for r in rs], color=color, lw=2,
                               marker="o", ms=4, markeredgecolor=SURFACE, markeredgewidth=1)
                handles.setdefault(impl, h)
                ends.append((rs[-1]["bytes"], rs[-1][key]))
            if row == 1 and ends:
                if link:
                    ax.axhline(link[1], color=REF, lw=1, ls="--")
                ax.set_ylim(bottom=0)
                ys = spread([v for _, v in ends], 0.035 * ax.get_ylim()[1])
                for (x, v), y in zip(ends, ys):
                    ax.annotate(f"{v:.1f}", (x, y), xytext=(6, 0), textcoords="offset points",
                                va="center", fontsize=8, color=TEXT_2, annotation_clip=False)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sizes)
            ax.set_xticklabels([size_label(b) for b in sizes], rotation=45, ha="right")
            ax.set_xlabel("Tensor size per GPU (fp32)")
            if row == 0:
                ax.set_yscale("log")
                ax.set_title(title, loc="left", fontsize=11)
                lo = min(r["median_us"] for r in rows if r["bytes"] == sizes[0])
                ax.annotate(f"{lo:.0f} us", (sizes[0], lo), xytext=(-4, -12), textcoords="offset points",
                            ha="left", fontsize=8, color=TEXT_2)
            else:
                if link:
                    ax.annotate(f"{link[0]} spec, one direction: {link[1]:g} GB/s", (sizes[0], link[1]),
                                xytext=(2, 4), textcoords="offset points", fontsize=8, color=TEXT_2)
            if col == 0:
                ax.set_ylabel(ylabel)
    fig.legend([handles[i] for i in IMPL_ORDER if i in handles], [IMPL_LABEL[i] for i in IMPL_ORDER if i in handles],
               loc="lower center", ncol=3 if M == 1 else 6, bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "cuda_ring.png")
    fig.savefig(out, dpi=150)
    print("Wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
