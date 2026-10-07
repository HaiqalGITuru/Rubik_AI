"""
Evaluasi AI kubus 4x4 pada berbagai tingkat kesulitan (panjang scramble).

Tidak ada solver pembanding semudah Kociemba untuk 4x4, jadi yang diukur adalah
tingkat keberhasilan dan panjang solusi dibanding panjang scramble.

    python -m scripts.eval_4x4 --model models/cubenet_4x4.pt
    python -m scripts.eval_4x4 --model models/cubenet_4x4.pt --beams 256 1024 --n 20
    python -m scripts.eval_4x4 --plot-only

Bisa dihentikan (Ctrl+C) dan dilanjutkan dengan perintah yang sama.
Hasil: results/<nama>/raw.csv, summary.csv, ringkasan_4x4.png
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from rubik.model import load_model
from rubik.nxn import CubeN
from rubik.solver import beam_search, compress_moves

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["beam", "depth", "cube_id", "solved", "length_htm", "scramble_htm", "time_sec"]


def load_benchmark(depths, n, seed=4444) -> dict:
    path = ROOT / "benchmarks" / "benchmark_4x4.json"
    if path.exists():
        bench = json.loads(path.read_text())
        if all(str(d) in bench["cubes"] and len(bench["cubes"][str(d)]) >= n for d in depths):
            return bench
    rng = np.random.default_rng(seed)
    bench = {"seed": seed, "cubes": {}}
    for d in sorted(set(depths) | {5, 10, 15, 20, 25, 30, 40}):
        items = []
        for i in range(max(n, 20)):
            moves = CubeN(4).scramble(d, rng=rng)
            items.append({"id": i, "scramble": " ".join(moves), "scramble_htm": len(compress_moves(moves))})
        bench["cubes"][str(d)] = items
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(bench, indent=1))
    print(f"Set kubus uji 4x4 dibuat: {path}")
    return bench


def read_raw(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run(args, bench, out):
    raw_path = out / "raw.csv"
    done = {(int(r["beam"]), int(r["depth"]), int(r["cube_id"])) for r in read_raw(raw_path)}
    if not raw_path.exists():
        with open(raw_path, "w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()
    torch.set_num_threads(os.cpu_count())
    model = load_model(args.model, device=args.device)
    beam_search(model, CubeN(4).apply_sequence("U R F"), beam_width=16, max_depth=5)  # pemanasan

    for beam in args.beams:
        for d in args.depths:
            todo = [c for c in bench["cubes"][str(d)][: args.n] if (beam, d, c["id"]) not in done]
            if not todo:
                continue
            print(f"\nBeam {beam}, scramble {d} gerakan: {len(todo)} kubus")
            for item in todo:
                cube = CubeN(4).apply_sequence(item["scramble"])
                res = beam_search(model, cube, beam_width=beam, max_depth=args.max_depth)
                sol = compress_moves(res.moves) if res.solved else []
                if res.solved:
                    assert cube.copy().apply_sequence(sol).is_solved(), "solusi tidak valid!"
                with open(raw_path, "a", newline="") as f:
                    csv.DictWriter(f, FIELDS).writerow({
                        "beam": beam, "depth": d, "cube_id": item["id"], "solved": int(res.solved),
                        "length_htm": len(sol) if res.solved else "", "scramble_htm": item["scramble_htm"],
                        "time_sec": f"{res.time_sec:.4f}"})
                status = f"{len(sol):>2} langkah" if res.solved else "GAGAL    "
                print(f"  kubus #{item['id']:<3} {status} (scramble {item['scramble_htm']:>2}) {res.time_sec:6.2f} dtk")


def summarize_and_plot(out, depths, n):
    raw = [r for r in read_raw(out / "raw.csv") if int(r["depth"]) in depths and int(r["cube_id"]) < n]
    rows = []
    for beam in sorted({int(r["beam"]) for r in raw}):
        for d in depths:
            rs = [r for r in raw if int(r["beam"]) == beam and int(r["depth"]) == d]
            if not rs:
                continue
            ok = [r for r in rs if r["solved"] == "1"]
            rows.append({
                "beam": beam, "scramble_gerakan": d, "kubus": len(rs),
                "berhasil_persen": round(100 * len(ok) / len(rs), 1),
                "langkah_rata2": round(np.mean([int(r["length_htm"]) for r in ok]), 2) if ok else "",
                "scramble_htm_rata2": round(np.mean([int(r["scramble_htm"]) for r in rs]), 2),
                "waktu_rata2_dtk": round(np.mean([float(r["time_sec"]) for r in rs]), 3),
            })
    if not rows:
        print("Belum ada hasil.")
        return
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#A9B3C2", "#6B85E8", "#2453E8", "#14318F"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, beam in enumerate(sorted({r["beam"] for r in rows})):
        rb = [r for r in rows if r["beam"] == beam]
        c = colors[i % len(colors)]
        a1.plot([r["scramble_gerakan"] for r in rb], [r["berhasil_persen"] for r in rb], "o-", color=c, label=f"beam {beam}")
        ok = [r for r in rb if r["langkah_rata2"] != ""]
        a2.plot([r["scramble_gerakan"] for r in ok], [r["langkah_rata2"] for r in ok], "o-", color=c, label=f"AI beam {beam}")
    ref = [r for r in rows if r["beam"] == rows[0]["beam"]]
    a2.plot([r["scramble_gerakan"] for r in ref], [r["scramble_htm_rata2"] for r in ref], "--", color="#667085",
            label="panjang scramble")
    a1.set_ylim(0, 105); a1.set_xlabel("Panjang scramble (gerakan seperempat)"); a1.set_ylabel("Berhasil (%)")
    a1.set_title("Kubus 4x4: tingkat keberhasilan"); a1.legend(frameon=False)
    a2.set_xlabel("Panjang scramble (gerakan seperempat)"); a2.set_ylabel("Langkah (HTM)")
    a2.set_title("Kubus 4x4: panjang solusi"); a2.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "ringkasan_4x4.png")

    print(f"\n{'Beam':>6}{'Scramble':>10}{'Berhasil':>10}{'Langkah':>9}{'(scramble)':>11}{'Waktu':>10}")
    for r in rows:
        print(f"{r['beam']:>6}{r['scramble_gerakan']:>10}{r['berhasil_persen']:>9}%{str(r['langkah_rata2'] or '-'):>9}"
              f"{r['scramble_htm_rata2']:>11}{r['waktu_rata2_dtk']:>8} dtk")
    print(f"\nHasil tersimpan di {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/cubenet_4x4.pt")
    p.add_argument("--beams", type=int, nargs="+", default=[1024])
    p.add_argument("--depths", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30, 40])
    p.add_argument("--n", type=int, default=20, help="kubus per tingkat kesulitan")
    p.add_argument("--max-depth", type=int, default=60)
    p.add_argument("--device", default="cpu")
    p.add_argument("--name", default="eval_4x4")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()

    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    bench = load_benchmark(args.depths, args.n)
    if not args.plot_only:
        try:
            run(args, bench, out)
        except KeyboardInterrupt:
            print("\nDihentikan. Jalankan perintah yang sama untuk melanjutkan.")
    summarize_and_plot(out, args.depths, args.n)


if __name__ == "__main__":
    main()
