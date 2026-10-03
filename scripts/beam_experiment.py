"""
Eksperimen lebar beam: hubungan antara lebar pencarian, tingkat keberhasilan,
panjang solusi, dan waktu berpikir.

Contoh (dari folder root proyek):
    python -m scripts.beam_experiment                          # default: 100 kubus, beam 256..16384
    python -m scripts.beam_experiment --n 30                   # uji cepat
    python -m scripts.beam_experiment --beams 256 1024 4096    # tanpa 16384
    python -m scripts.beam_experiment --plot-only              # buat ulang grafik saja

Bisa dihentikan dengan Ctrl+C kapan saja. Jalankan perintah yang sama lagi untuk
melanjutkan: kubus yang sudah diuji tidak diulang.

Hasil:
    results/<nama>/raw.csv       hasil per kubus per beam
    results/<nama>/summary.csv   ringkasan per beam (+ Kociemba)
    results/<nama>/*.png         grafik untuk laporan
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
from rubik.solver import beam_search, compress_moves

ROOT = Path(__file__).resolve().parent.parent
RAW_FIELDS = ["beam", "cube_id", "solved", "length_htm", "length_qtm", "time_sec", "nodes", "kociemba_length"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/cubenet.pt")
    p.add_argument("--n", type=int, default=100, help="jumlah kubus uji")
    p.add_argument("--beams", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    p.add_argument("--max-depth", type=int, default=40)
    p.add_argument("--name", default="beam_experiment", help="nama folder hasil di results/")
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--device", default="cpu", help="cpu / cuda / auto")
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

    # Pemanasan: panggilan pertama PyTorch selalu lebih lambat, jangan sampai ikut terukur
    warm = Cube().apply_sequence(cubes[0]["scramble"])
    beam_search(model, warm, beam_width=64, max_depth=5)

    for beam in sorted(args.beams):
        todo = [c for c in cubes if (beam, c["id"]) not in done]
        if not todo:
            print(f"Beam {beam:>6}: sudah selesai, dilewati")
            continue
        print(f"\nBeam {beam:>6}: menguji {len(todo)} kubus")
        t_start = time.perf_counter()
        for k, item in enumerate(todo, 1):
            cube = Cube().apply_sequence(item["scramble"])
            res = beam_search(model, cube, beam_width=beam, max_depth=args.max_depth)
            htm = compress_moves(res.moves) if res.solved else []
            if res.solved:
                assert cube.copy().apply_sequence(htm).is_solved(), "solusi tidak valid!"
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
            status = f"{len(htm):>2} langkah" if res.solved else "GAGAL    "
            print(f"  [{k:>3}/{len(todo)}] kubus #{item['id']:<3} {status} {res.time_sec:6.2f} dtk"
                  f"  | sisa ~{eta / 60:.1f} mnt")


def summarize(raw: list[dict], bench: dict, n: int) -> list[dict]:
    rows = []
    for beam in sorted({int(r["beam"]) for r in raw}):
        rs = [r for r in raw if int(r["beam"]) == beam]
        ok = [r for r in rs if r["solved"] == "1"]
        lengths = np.array([int(r["length_htm"]) for r in ok]) if ok else np.array([])
        gaps = np.array([int(r["length_htm"]) - int(r["kociemba_length"]) for r in ok if r["kociemba_length"]])
        times = np.array([float(r["time_sec"]) for r in rs])
        rows.append({
            "metode": f"AI beam {beam}", "beam": beam, "kubus": len(rs),
            "berhasil_persen": round(100 * len(ok) / len(rs), 1),
            "langkah_rata2": round(lengths.mean(), 2) if len(lengths) else "",
            "langkah_median": float(np.median(lengths)) if len(lengths) else "",
            "langkah_min": int(lengths.min()) if len(lengths) else "",
            "langkah_maks": int(lengths.max()) if len(lengths) else "",
            "selisih_vs_kociemba": round(gaps.mean(), 2) if len(gaps) else "",
            "waktu_rata2_dtk": round(times.mean(), 3),
            "waktu_maks_dtk": round(times.max(), 3),
            "posisi_dievaluasi_rata2": int(np.mean([int(r["nodes"]) for r in rs])),
        })
    kc = [c for c in bench["cubes"][:n] if "kociemba_length" in c]
    if kc:
        kl = np.array([c["kociemba_length"] for c in kc])
        rows.append({
            "metode": "Kociemba", "beam": "", "kubus": len(kc), "berhasil_persen": 100.0,
            "langkah_rata2": round(kl.mean(), 2), "langkah_median": float(np.median(kl)),
            "langkah_min": int(kl.min()), "langkah_maks": int(kl.max()), "selisih_vs_kociemba": 0,
            "waktu_rata2_dtk": round(np.mean([c["kociemba_ms"] for c in kc]) / 1000, 4),
            "waktu_maks_dtk": round(max(c["kociemba_ms"] for c in kc) / 1000, 4),
            "posisi_dievaluasi_rata2": "",
        })
    return rows


def plot(raw: list[dict], summary: list[dict], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    blue, grey = "#2453E8", "#8A93A3"
    ai = [s for s in summary if s["metode"].startswith("AI")]
    kc = next((s for s in summary if s["metode"] == "Kociemba"), None)
    beams = [s["beam"] for s in ai]
    labels = [f"{b:,}".replace(",", ".") for b in beams]
    x = np.arange(len(beams))

    # 1) Ringkasan tiga panel
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    ax.bar(x, [s["berhasil_persen"] for s in ai], color=blue)
    ax.set_xticks(x, labels); ax.set_ylim(0, 105)
    ax.set_title("Tingkat keberhasilan"); ax.set_ylabel("%"); ax.set_xlabel("Lebar beam")
    for i, s in enumerate(ai):
        ax.text(i, s["berhasil_persen"] + 1.5, f"{s['berhasil_persen']:.0f}%", ha="center", fontsize=9)

    ax = axes[1]
    ax.plot(x, [s["langkah_rata2"] or np.nan for s in ai], "o-", color=blue, label="AI")
    if kc:
        ax.axhline(kc["langkah_rata2"], color=grey, ls="--", label=f"Kociemba ({kc['langkah_rata2']})")
    ax.set_xticks(x, labels); ax.set_title("Rata-rata panjang solusi"); ax.set_ylabel("langkah (HTM)")
    ax.set_xlabel("Lebar beam"); ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(x, [s["waktu_rata2_dtk"] for s in ai], "o-", color=blue)
    ax.set_yscale("log"); ax.set_xticks(x, labels)
    ax.set_title("Rata-rata waktu berpikir"); ax.set_ylabel("detik (skala log)"); ax.set_xlabel("Lebar beam")
    fig.tight_layout()
    fig.savefig(out_dir / "ringkasan.png")
    plt.close(fig)

    # 2) Trade-off: waktu vs panjang solusi
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([s["waktu_rata2_dtk"] for s in ai], [s["langkah_rata2"] or np.nan for s in ai], "o-", color=blue)
    for s, lab in zip(ai, labels):
        ax.annotate(f"beam {lab}", (s["waktu_rata2_dtk"], s["langkah_rata2"] or np.nan),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    if kc:
        ax.scatter([kc["waktu_rata2_dtk"]], [kc["langkah_rata2"]], color=grey, zorder=3)
        ax.annotate("Kociemba", (kc["waktu_rata2_dtk"], kc["langkah_rata2"]),
                    textcoords="offset points", xytext=(6, -12), fontsize=8, color=grey)
    ax.set_xscale("log"); ax.set_xlabel("Waktu berpikir rata-rata (detik, log)")
    ax.set_ylabel("Panjang solusi rata-rata (HTM)")
    ax.set_title("Trade-off kecepatan vs kualitas solusi")
    fig.tight_layout(); fig.savefig(out_dir / "tradeoff.png"); plt.close(fig)

    # 3) Distribusi panjang solusi per beam
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [[int(r["length_htm"]) for r in raw if int(r["beam"]) == b and r["solved"] == "1"] for b in beams]
    if kc:
        data.append([int(r["kociemba_length"]) for r in raw
                     if int(r["beam"]) == beams[0] and r["kociemba_length"]])
    box = ax.boxplot([d if d else [np.nan] for d in data], patch_artist=True, widths=0.55,
                     medianprops={"color": "#18202C", "lw": 1.5})
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(grey if kc and i == len(data) - 1 else "#C9D3E6")
    ax.set_xticks(range(1, len(data) + 1), [f"beam\n{l}" for l in labels] + (["Kociemba"] if kc else []))
    ax.set_ylabel("Panjang solusi (HTM)"); ax.set_title("Distribusi panjang solusi")
    fig.tight_layout(); fig.savefig(out_dir / "distribusi_langkah.png"); plt.close(fig)


def main():
    args = parse_args()
    out_dir = ROOT / "results" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    bench = load_or_create_benchmark(100 if args.n <= 100 else args.n)

    if not args.plot_only:
        try:
            run(args, bench, out_dir)
        except KeyboardInterrupt:
            print("\nDihentikan. Jalankan perintah yang sama untuk melanjutkan.")

    raw = [r for r in read_raw(out_dir / "raw.csv") if int(r["cube_id"]) < args.n]
    if not raw:
        print("Belum ada hasil.")
        return
    summary = summarize(raw, bench, args.n)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    plot(raw, summary, out_dir)

    print("\n================= RINGKASAN =================")
    print(f"{'Metode':<16}{'Berhasil':>9}{'Langkah':>9}{'vs Koc.':>9}{'Waktu':>11}")
    for s in summary:
        lang = f"{s['langkah_rata2']}" if s["langkah_rata2"] != "" else "-"
        gap = f"{s['selisih_vs_kociemba']:+}" if s["selisih_vs_kociemba"] != "" else "-"
        print(f"{s['metode']:<16}{s['berhasil_persen']:>8}%{lang:>9}{gap:>9}{s['waktu_rata2_dtk']:>9} dtk")
    print(f"\nHasil tersimpan di {out_dir}")


if __name__ == "__main__":
    main()