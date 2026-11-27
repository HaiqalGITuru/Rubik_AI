"""
Beam width experiment: how search width relates to success rate,
solution length and thinking time.

Examples (from the project root):
    python -m scripts.beam_experiment                          # default: 100 cubes, beam 256..16384
    python -m scripts.beam_experiment --n 30                   # quick test
    python -m scripts.beam_experiment --beams 256 1024 4096    # without 16384
    python -m scripts.beam_experiment --plot-only              # only redraw the charts

Can be stopped with Ctrl+C at any time. Run the same command again to
resume: cubes that were already tested are not repeated.

Output:
    results/<name>/raw.csv       result per cube per beam
    results/<name>/summary.csv   summary per beam (+ Kociemba)
    results/<name>/*.png         charts for the report
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch

from rubik import Cube
from rubik.benchmark import load_or_create_benchmark
from rubik.model import load_model
from rubik.deepcubea import weighted_astar
from rubik.solver import beam_search, compress_moves

ROOT = Path(__file__).resolve().parent.parent
RAW_FIELDS = ["beam", "cube_id", "solved", "length_htm", "length_qtm", "time_sec", "nodes", "kociemba_length"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/cubenet.pt")
    p.add_argument("--n", type=int, default=100, help="number of test cubes")
    p.add_argument("--beams", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    p.add_argument("--max-depth", type=int, default=40)
    p.add_argument("--name", default="beam_experiment", help="output folder name under results/")
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--device", default="cpu", help="cpu / cuda / auto")
    p.add_argument("--solver", default="beam", choices=["beam", "astar"],
                   help="beam = EfficientCube (beam search), astar = DeepCubeA (batch weighted A*)")
    p.add_argument("--weight", type=float, default=0.6, help="weight of g in A* (DeepCubeA paper: 0.6)")
    p.add_argument("--plot-only", action="store_true")
    return p.parse_args()


def read_raw(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run(args, bench, out_dir: Path):
    raw_path = out_dir / "raw.csv"
    done = {(int(r["beam"]), int(r["cube_id"])) for r in read_raw(raw_path)}
    if not raw_path.exists():
        with open(raw_path, "w", newline="") as f:
            csv.DictWriter(f, RAW_FIELDS).writeheader()

    torch.set_num_threads(args.threads)
    model = load_model(args.model, device=args.device)
    cubes = bench["cubes"][: args.n]

    if args.solver == "astar":
        solve = lambda cube, width: weighted_astar(model, cube, batch_size=width, weight=args.weight)
    else:
        solve = lambda cube, width: beam_search(model, cube, beam_width=width, max_depth=args.max_depth)

    # Warm-up: the first PyTorch call is always slower; keep it out of the timing
    warm = Cube().apply_sequence("R U F")
    solve(warm, 64)

    for beam in sorted(args.beams):
        todo = [c for c in cubes if (beam, c["id"]) not in done]
        if not todo:
            print(f"{label_prefix(args)} {beam:>6}: already done, skipped")
            continue
        print(f"\n{label_prefix(args)} {beam:>6}: testing {len(todo)} cubes")
        t_start = time.perf_counter()
        for k, item in enumerate(todo, 1):
            cube = Cube().apply_sequence(item["scramble"])
            res = solve(cube, beam)
            htm = compress_moves(res.moves) if res.solved else []
            if res.solved:
                assert cube.copy().apply_sequence(htm).is_solved(), "invalid solution!"
            row = {
                "beam": beam, "cube_id": item["id"], "solved": int(res.solved),
                "length_htm": len(htm) if res.solved else "", "length_qtm": len(res.moves) if res.solved else "",
                "time_sec": f"{res.time_sec:.4f}", "nodes": res.nodes,
                "kociemba_length": item.get("kociemba_length", ""),
            }
            with open(raw_path, "a", newline="") as f:
                csv.DictWriter(f, RAW_FIELDS).writerow(row)

            elapsed = time.perf_counter() - t_start
            eta = elapsed / k * (len(todo) - k)
            status = f"{len(htm):>2} moves" if res.solved else "FAILED   "
            print(f"  [{k:>3}/{len(todo)}] cube #{item['id']:<3} {status} {res.time_sec:6.2f} s"
                  f"  | ~{eta / 60:.1f} min left")


def label_prefix(args) -> str:
    return "DeepCubeA N=" if args.solver == "astar" else "AI beam"


def summarize(raw: list[dict], bench: dict, n: int, prefix: str = "AI beam") -> list[dict]:
    rows = []
    for beam in sorted({int(r["beam"]) for r in raw}):
        rs = [r for r in raw if int(r["beam"]) == beam]
        ok = [r for r in rs if r["solved"] == "1"]
        lengths = np.array([int(r["length_htm"]) for r in ok]) if ok else np.array([])
        gaps = np.array([int(r["length_htm"]) - int(r["kociemba_length"]) for r in ok if r["kociemba_length"]])
        times = np.array([float(r["time_sec"]) for r in rs])
        rows.append({
            "method": f"{prefix}{beam}" if prefix.endswith("=") else f"{prefix} {beam}", "beam": beam, "cubes": len(rs),
            "solved_pct": round(100 * len(ok) / len(rs), 1),
            "mean_length": round(lengths.mean(), 2) if len(lengths) else "",
            "median_length": float(np.median(lengths)) if len(lengths) else "",
            "min_length": int(lengths.min()) if len(lengths) else "",
            "max_length": int(lengths.max()) if len(lengths) else "",
            "gap_vs_kociemba": round(gaps.mean(), 2) if len(gaps) else "",
            "mean_time_sec": round(times.mean(), 3),
            "max_time_sec": round(times.max(), 3),
            "mean_nodes": int(np.mean([int(r["nodes"]) for r in rs])),
        })
    kc = [c for c in bench["cubes"][:n] if "kociemba_length" in c]
    if kc:
        kl = np.array([c["kociemba_length"] for c in kc])
        rows.append({
            "method": "Kociemba", "beam": "", "cubes": len(kc), "solved_pct": 100.0,
            "mean_length": round(kl.mean(), 2), "median_length": float(np.median(kl)),
            "min_length": int(kl.min()), "max_length": int(kl.max()), "gap_vs_kociemba": 0,
            "mean_time_sec": round(np.mean([c["kociemba_ms"] for c in kc]) / 1000, 4),
            "max_time_sec": round(max(c["kociemba_ms"] for c in kc) / 1000, 4),
            "mean_nodes": "",
        })
    return rows


def plot(raw: list[dict], summary: list[dict], out_dir: Path, xlabel: str = "Beam width"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    blue, grey = "#2453E8", "#8A93A3"
    ai = [s for s in summary if s["beam"] != ""]
    kc = next((s for s in summary if s["method"] == "Kociemba"), None)
    beams = [s["beam"] for s in ai]
    labels = [f"{b:,}" for b in beams]
    short = "N" if "A*" in xlabel else "beam"
    x = np.arange(len(beams))

    # 1) Three-panel overview
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    ax.bar(x, [s["solved_pct"] for s in ai], color=blue)
    ax.set_xticks(x, labels); ax.set_ylim(0, 105)
    ax.set_title("Success rate"); ax.set_ylabel("%"); ax.set_xlabel(xlabel)
    for i, s in enumerate(ai):
        ax.text(i, s["solved_pct"] + 1.5, f"{s['solved_pct']:.0f}%", ha="center", fontsize=9)

    ax = axes[1]
    ax.plot(x, [s["mean_length"] or np.nan for s in ai], "o-", color=blue, label="AI")
    if kc:
        ax.axhline(kc["mean_length"], color=grey, ls="--", label=f"Kociemba ({kc['mean_length']})")
    ax.set_xticks(x, labels); ax.set_title("Mean solution length"); ax.set_ylabel("moves (HTM)")
    ax.set_xlabel(xlabel); ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(x, [s["mean_time_sec"] for s in ai], "o-", color=blue)
    ax.set_yscale("log"); ax.set_xticks(x, labels)
    ax.set_title("Mean thinking time"); ax.set_ylabel("seconds (log scale)"); ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(out_dir / "overview.png")
    plt.close(fig)

    # 2) Trade-off: time vs solution length
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([s["mean_time_sec"] for s in ai], [s["mean_length"] or np.nan for s in ai], "o-", color=blue)
    for s, lab in zip(ai, labels):
        ax.annotate(f"{short} {lab}", (s["mean_time_sec"], s["mean_length"] or np.nan),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    if kc:
        ax.scatter([kc["mean_time_sec"]], [kc["mean_length"]], color=grey, zorder=3)
        ax.annotate("Kociemba", (kc["mean_time_sec"], kc["mean_length"]),
                    textcoords="offset points", xytext=(6, -12), fontsize=8, color=grey)
    ax.set_xscale("log"); ax.set_xlabel("Mean thinking time (seconds, log)")
    ax.set_ylabel("Mean solution length (HTM)")
    ax.set_title("Speed vs solution quality trade-off")
    fig.tight_layout(); fig.savefig(out_dir / "tradeoff.png"); plt.close(fig)

    # 3) Solution length distribution per beam
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [[int(r["length_htm"]) for r in raw if int(r["beam"]) == b and r["solved"] == "1"] for b in beams]
    if kc:
        data.append([int(r["kociemba_length"]) for r in raw
                     if int(r["beam"]) == beams[0] and r["kociemba_length"]])
    box = ax.boxplot([d if d else [np.nan] for d in data], patch_artist=True, widths=0.55,
                     medianprops={"color": "#18202C", "lw": 1.5})
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(grey if kc and i == len(data) - 1 else "#C9D3E6")
    ax.set_xticks(range(1, len(data) + 1), [f"{short}\n{l}" for l in labels] + (["Kociemba"] if kc else []))
    ax.set_ylabel("Solution length (HTM)"); ax.set_title("Solution length distribution")
    fig.tight_layout(); fig.savefig(out_dir / "length_distribution.png"); plt.close(fig)


def main():
    args = parse_args()
    out_dir = ROOT / "results" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    bench = load_or_create_benchmark(100 if args.n <= 100 else args.n)

    if not args.plot_only:
        try:
            run(args, bench, out_dir)
        except KeyboardInterrupt:
            print("\nStopped. Run the same command to resume.")

    raw = [r for r in read_raw(out_dir / "raw.csv") if int(r["cube_id"]) < args.n]
    if not raw:
        print("No results yet.")
        return
    summary = summarize(raw, bench, args.n, label_prefix(args))
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    plot(raw, summary, out_dir, "A* batch size (N)" if args.solver == "astar" else "Beam width")

    print("\n================== SUMMARY ==================")
    print(f"{'Method':<18}{'Solved':>9}{'Moves':>9}{'vs Koc.':>9}{'Time':>11}")
    for s in summary:
        length = f"{s['mean_length']}" if s["mean_length"] != "" else "-"
        gap = f"{s['gap_vs_kociemba']:+}" if s["gap_vs_kociemba"] != "" else "-"
        print(f"{s['method']:<18}{s['solved_pct']:>8}%{length:>9}{gap:>9}{s['mean_time_sec']:>9} s")
    print(f"\nResults saved in {out_dir}")


if __name__ == "__main__":
    main()