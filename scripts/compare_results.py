"""
Compare several experiment results in one chart and one table.

    python -m scripts.compare_results \\
        beam_experiment_small_gpu:"EfficientCube small" \\
        beam_experiment_large:"EfficientCube large" \\
        deepcubea_large:"DeepCubeA large"

Each argument = <folder name under results/>:<label>. Only compare results from the same
hardware (e.g. all on a Colab GPU), because thinking time depends on the hardware.

Output: results/comparison/comparison.png and comparison.csv
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLORS = ["#A9B3C2", "#2453E8", "#E8762D", "#14A050", "#8E44AD"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="folder_name:label")
    p.add_argument("--out", default="comparison")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})

    out_dir = ROOT / "results" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, kociemba = [], None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))

    for i, run in enumerate(args.runs):
        name, _, label = run.partition(":")
        label = label or name
        with open(ROOT / "results" / name / "summary.csv", newline="") as f:
            summary = list(csv.DictReader(f))
        configs = [s for s in summary if s["beam"]]
        if kociemba is None:  # every run uses the same 100 test cubes
            kociemba = next((s for s in summary if s["method"] == "Kociemba"), None)
        color = COLORS[i % len(COLORS)]

        ok = [s for s in configs if s["mean_length"]]
        t = [float(s["mean_time_sec"]) for s in ok]
        ax1.plot(t, [float(s["mean_length"]) for s in ok], "o-", color=color, label=label)
        for s, x in zip(ok, t):
            ax1.annotate(s["beam"], (x, float(s["mean_length"])), textcoords="offset points",
                         xytext=(4, 5), fontsize=7, color=color)
        ax2.plot([float(s["mean_time_sec"]) for s in configs],
                 [float(s["solved_pct"]) for s in configs], "o-", color=color, label=label)
        for s in configs:
            rows.append({"method": label, "setting": s["beam"], **{k: s[k] for k in (
                "solved_pct", "mean_length", "median_length", "gap_vs_kociemba",
                "mean_time_sec", "mean_nodes")}})

    if kociemba:
        kl = float(kociemba["mean_length"])
        ax1.axhline(kl, color="#667085", ls="--", lw=1, label=f"Kociemba ({kl})")
        rows.append({"method": "Kociemba", "setting": "", "solved_pct": 100,
                     "mean_length": kl, "median_length": kociemba["median_length"],
                     "gap_vs_kociemba": 0, "mean_time_sec": kociemba["mean_time_sec"],
                     "mean_nodes": ""})

    ax1.set_xscale("log"); ax1.set_xlabel("Mean thinking time (seconds, log)")
    ax1.set_ylabel("Mean solution length (HTM)"); ax1.set_title("Solution quality vs time")
    ax1.legend(frameon=False, fontsize=8)
    ax2.set_xscale("log"); ax2.set_xlabel("Mean thinking time (seconds, log)")
    ax2.set_ylabel("Solved (%)"); ax2.set_ylim(0, 105); ax2.set_title("Success rate vs time")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png")

    with open(out_dir / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"{'Method':<24}{'Setting':>8}{'Solved':>9}{'Moves':>9}{'Time':>11}")
    for r in rows:
        print(f"{r['method']:<24}{r['setting']:>8}{float(r['solved_pct']):>8.0f}%"
              f"{str(r['mean_length'] or '-'):>9}{float(r['mean_time_sec']):>9.3f} s")
    print(f"\nSaved in {out_dir}")


if __name__ == "__main__":
    main()
