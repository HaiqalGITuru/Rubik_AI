"""
Bandingkan beberapa hasil eksperimen dalam satu grafik dan satu tabel.

    python -m scripts.compare_results \\
        beam_experiment_small_gpu:"EfficientCube kecil" \\
        beam_experiment_large:"EfficientCube besar" \\
        deepcubea_large:"DeepCubeA besar"

Setiap argumen = <nama folder di results/>:<label>. Bandingkan hanya hasil dari perangkat
yang sama (misalnya semua GPU Colab), karena waktu berpikir bergantung pada perangkat keras.

Hasil: results/perbandingan/perbandingan.png dan perbandingan.csv
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLORS = ["#A9B3C2", "#2453E8", "#E8762D", "#14A050", "#8E44AD"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="nama_folder:label")
    p.add_argument("--out", default="perbandingan")
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
        if kociemba is None:  # semua hasil memakai 100 kubus uji yang sama
            kociemba = next((s for s in summary if s["metode"] == "Kociemba"), None)
        color = COLORS[i % len(COLORS)]

        ok = [s for s in configs if s["langkah_rata2"]]
        t = [float(s["waktu_rata2_dtk"]) for s in ok]
        ax1.plot(t, [float(s["langkah_rata2"]) for s in ok], "o-", color=color, label=label)
        for s, x in zip(ok, t):
            ax1.annotate(s["beam"], (x, float(s["langkah_rata2"])), textcoords="offset points",
                         xytext=(4, 5), fontsize=7, color=color)
        ax2.plot([float(s["waktu_rata2_dtk"]) for s in configs],
                 [float(s["berhasil_persen"]) for s in configs], "o-", color=color, label=label)
        for s in configs:
            rows.append({"metode": label, "pengaturan": s["beam"], **{k: s[k] for k in (
                "berhasil_persen", "langkah_rata2", "langkah_median", "selisih_vs_kociemba",
                "waktu_rata2_dtk", "posisi_dievaluasi_rata2")}})

    if kociemba:
        kl = float(kociemba["langkah_rata2"])
        ax1.axhline(kl, color="#667085", ls="--", lw=1, label=f"Kociemba ({kl})")
        rows.append({"metode": "Kociemba", "pengaturan": "", "berhasil_persen": 100,
                     "langkah_rata2": kl, "langkah_median": kociemba["langkah_median"],
                     "selisih_vs_kociemba": 0, "waktu_rata2_dtk": kociemba["waktu_rata2_dtk"],
                     "posisi_dievaluasi_rata2": ""})

    ax1.set_xscale("log"); ax1.set_xlabel("Waktu berpikir rata-rata (detik, log)")
    ax1.set_ylabel("Panjang solusi rata-rata (HTM)"); ax1.set_title("Kualitas solusi vs waktu")
    ax1.legend(frameon=False, fontsize=8)
    ax2.set_xscale("log"); ax2.set_xlabel("Waktu berpikir rata-rata (detik, log)")
    ax2.set_ylabel("Berhasil (%)"); ax2.set_ylim(0, 105); ax2.set_title("Tingkat keberhasilan vs waktu")
    fig.tight_layout()
    fig.savefig(out_dir / "perbandingan.png")

    with open(out_dir / "perbandingan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"{'Metode':<24}{'Atur':>7}{'Berhasil':>10}{'Langkah':>9}{'Waktu':>11}")
    for r in rows:
        print(f"{r['metode']:<24}{r['pengaturan']:>7}{float(r['berhasil_persen']):>9.0f}%"
              f"{str(r['langkah_rata2'] or '-'):>9}{float(r['waktu_rata2_dtk']):>9.3f} dtk")
    print(f"\nTersimpan di {out_dir}")


if __name__ == "__main__":
    main()
