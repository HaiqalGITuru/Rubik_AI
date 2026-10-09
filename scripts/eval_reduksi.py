"""
Evaluasi metode reduksi 4x4 (AI reduksi + paritas + AI 3x3).

Dua set kubus uji:
    - "depth": set yang sama dengan eval_4x4 (benchmarks/benchmark_4x4.json), supaya bisa
               dibandingkan langsung dengan AI 4x4 satu tahap.
    - "wca"  : scramble penuh gaya kompetisi, 40 gerakan WCA (benchmarks/benchmark_4x4_wca.json).

    python -m scripts.eval_reduksi --reduce-model models/reduksi_4x4.pt
    python -m scripts.eval_reduksi --sets wca --n 50 --beams 1024 4096
    python -m scripts.eval_reduksi --summary-only

Bisa dihentikan (Ctrl+C) dan dilanjutkan dengan perintah yang sama.
Hasil: results/<nama>/raw.csv dan summary.csv
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from rubik.model import load_model
from rubik.nxn import CubeN
from rubik.reduction import solve_reduction
from rubik.solver import compress_moves
from scripts.eval_4x4 import load_benchmark
from scripts.solve_4x4_reduksi import wca_scramble
from scripts.solve_ai_4x4 import to_wca

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["set", "beam", "depth", "cube_id", "solved", "stage_failed", "len_reduce", "len_parity", "len_3x3",
          "len_total", "oll_parity", "pll_parity", "time_reduce", "time_3x3"]


def load_wca_benchmark(n: int, seed: int = 4040) -> dict:
    path = ROOT / "benchmarks" / "benchmark_4x4_wca.json"
    if path.exists():
        bench = json.loads(path.read_text())
        if len(bench["cubes"]) >= n:
            return bench
    rng = np.random.default_rng(seed)
    bench = {"seed": seed, "length": 40,
             "cubes": [{"id": i, "scramble": " ".join(wca_scramble(40, rng))} for i in range(max(n, 100))]}
    path.write_text(json.dumps(bench, indent=1))
    print(f"Set kubus uji WCA dibuat: {path}")
    return bench


def read_raw(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reduce-model", default="models/reduksi_4x4.pt")
    p.add_argument("--model3", default="models/cubenet_large_infer.pt")
    p.add_argument("--sets", nargs="+", default=["depth", "wca"], choices=["depth", "wca"])
    p.add_argument("--beams", type=int, nargs="+", default=[1024])
    p.add_argument("--beam3", type=int, default=1024)
    p.add_argument("--depths", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    p.add_argument("--n", type=int, default=20, help="kubus per tingkat (set depth) / total (set wca)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--name", default="eval_reduksi")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()

    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw.csv"

    jobs = []  # (set, depth, id, scramble)
    if "depth" in args.sets:
        bench = load_benchmark(args.depths, args.n)
        for d in args.depths:
            jobs += [("depth", d, c["id"], c["scramble"]) for c in bench["cubes"][str(d)][: args.n]]
    if "wca" in args.sets:
        jobs += [("wca", 40, c["id"], c["scramble"]) for c in load_wca_benchmark(args.n)["cubes"][: args.n]]

    if not args.summary_only:
        done = {(r["set"], int(r["beam"]), int(r["depth"]), int(r["cube_id"])) for r in read_raw(raw_path)}
        if not raw_path.exists():
            with open(raw_path, "w", newline="") as f:
                csv.DictWriter(f, FIELDS).writeheader()
        torch.set_num_threads(os.cpu_count())
        reduce_model = load_model(args.reduce_model, device=args.device)
        model3 = load_model(args.model3, device=args.device)
        try:
            for beam in args.beams:
                for s, d, i, scr in jobs:
                    if (s, beam, d, i) in done:
                        continue
                    cube = CubeN(4).apply_sequence(scr)
                    res = solve_reduction(reduce_model, model3, cube, beam_reduce=beam, beam_3x3=args.beam3)
                    t1, t2, t3 = to_wca(res.reduce_moves), to_wca(res.parity_moves), res.solve3_moves
                    total = compress_moves(t1 + t2 + t3)
                    stage = "" if res.solved else ("reduksi" if "reduksi" in res.message else "3x3")
                    with open(raw_path, "a", newline="") as f:
                        csv.DictWriter(f, FIELDS).writerow({
                            "set": s, "beam": beam, "depth": d, "cube_id": i, "solved": int(res.solved),
                            "stage_failed": stage, "len_reduce": len(t1), "len_parity": len(t2),
                            "len_3x3": len(t3), "len_total": len(total) if res.solved else "",
                            "oll_parity": int(res.parity[0]), "pll_parity": int(res.parity[1]),
                            "time_reduce": f"{res.time_reduce:.3f}", "time_3x3": f"{res.time_3x3:.3f}"})
                    label = f"{s} {d:>2}" if s == "depth" else "wca 40"
                    status = f"{len(total):>3} gerakan WCA" if res.solved else f"GAGAL di tahap {stage}"
                    print(f"beam {beam:<5} {label} #{i:<3} {status:<24} "
                          f"(reduksi {len(t1)}, paritas {len(t2)}, 3x3 {len(t3)}) {res.time_sec:6.1f} dtk",
                          flush=True)
        except KeyboardInterrupt:
            print("\nDihentikan. Jalankan perintah yang sama untuk melanjutkan.")

    # Ringkasan
    raw = read_raw(raw_path)
    rows = []
    keys = sorted({(r["set"], int(r["beam"]), int(r["depth"])) for r in raw})
    for s, beam, d in keys:
        rs = [r for r in raw if (r["set"], int(r["beam"]), int(r["depth"])) == (s, beam, d)]
        ok = [r for r in rs if r["solved"] == "1"]
        mean = lambda k: round(float(np.mean([float(r[k]) for r in ok])), 1) if ok else ""
        rows.append({"set": s, "beam": beam, "scramble": d, "kubus": len(rs),
                     "berhasil_persen": round(100 * len(ok) / len(rs), 1),
                     "reduksi_rata2": mean("len_reduce"), "paritas_rata2": mean("len_parity"),
                     "3x3_rata2": mean("len_3x3"), "total_wca_rata2": mean("len_total"),
                     "waktu_rata2_dtk": round(float(np.mean([float(r["time_reduce"]) + float(r["time_3x3"])
                                                             for r in rs])), 2)})
    if not rows:
        print("Belum ada hasil.")
        return
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'Set':<6}{'Beam':>6}{'Scramble':>10}{'Berhasil':>10}{'Reduksi':>9}{'Paritas':>9}{'3x3':>6}"
          f"{'Total':>7}{'Waktu':>9}")
    for r in rows:
        print(f"{r['set']:<6}{r['beam']:>6}{r['scramble']:>10}{r['berhasil_persen']:>9}%"
              f"{str(r['reduksi_rata2']):>9}{str(r['paritas_rata2']):>9}{str(r['3x3_rata2']):>6}"
              f"{str(r['total_wca_rata2']):>7}{r['waktu_rata2_dtk']:>7} dtk")
    print(f"\nHasil tersimpan di {out}")


if __name__ == "__main__":
    main()
