# Rubik AI — Solver Kubus Rubik 3x3 dengan Deep Learning

## Struktur Proyek

```
rubik-ai/
├── rubik/              # Paket utama: simulator kubus
│   ├── __init__.py
│   ├── moves.py        # Definisi & permutasi 18 gerakan (dihitung dari geometri 3D)
│   ├── cube.py         # Kelas Cube: apply, scramble, is_solved, one-hot, render
│   ├── data.py         # Generator data training (self-supervised)
│   ├── model.py        # CubeNet (MLP + residual block, PyTorch)
│   └── solver.py       # Beam search yang dipandu jaringan saraf
├── tests/
│   └── test_cube.py    # Unit test simulator (47 test)
├── scripts/
│   ├── demo.py         # Demo simulator + pembanding Kociemba
│   ├── solve.py        # Solusi Kociemba untuk kubus fisik
│   ├── train.py        # Training model
│   ├── evaluate.py     # Evaluasi AI vs Kociemba
│   └── solve_ai.py     # Solusi AI untuk kubus fisik
├── data/               # (nanti) data hasil generate
├── models/             # (nanti) bobot model hasil training
├── notebooks/          # (opsional) eksperimen Jupyter
├── .vscode/settings.json
├── conftest.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

Jika `kociemba` gagal di-install di Windows (butuh compiler C), lewati saja dulu.
Simulator tetap jalan; test yang memakai Kociemba otomatis di-skip.

## Menjalankan

```bash
pytest -v               # jalankan semua test
python -m scripts.demo  # demo simulator
```

## Contoh Pemakaian

```python
from rubik import Cube

cube = Cube()
cube.apply_sequence("R U R' U'")
print(cube.render())
print(cube.is_solved())        # False

scramble = cube.scramble(20)   # acak 20 langkah
x = cube.to_one_hot()          # input jaringan saraf, shape (324,)
```

## Notasi
- `U R F D L B` : putar wajah searah jarum jam (dilihat dari wajah itu)
- `U'` : berlawanan jarum jam, `U2` : setengah putaran

## Roadmap
1. [x] Simulator kubus + unit test
2. [x] Generator data (scramble acak)
3. [x] Model jaringan saraf (PyTorch)
4. [x] Solver: beam search
5. [ ] Training penuh & evaluasi vs Kociemba

## Training & Evaluasi

```bash
python -m scripts.train --steps 2000        # uji cepat (~10-20 menit di CPU)
python -m scripts.train                     # training penuh (20.000 langkah)
python -m scripts.train --resume            # lanjutkan setelah Ctrl+C
python -m scripts.evaluate                  # AI vs Kociemba
python -m scripts.solve_ai "R U R' U'"      # coba dengan kubus fisik
```

Log training tersimpan di `models/cubenet_log.csv` (bisa dibuat grafik untuk laporan).

## Website (lokal)

```bash
pip install fastapi uvicorn
python -m web.server        # browser terbuka otomatis di http://127.0.0.1:8000
```

Server hanya bisa diakses dari laptop sendiri (127.0.0.1). Tekan Ctrl+C untuk berhenti.

## Training model besar (Google Colab)

Buka `notebooks/train_colab.ipynb` di Google Colab (File → Upload notebook), aktifkan GPU T4,
lalu jalankan sel dari atas. Checkpoint tersimpan di Google Drive dan training bisa dilanjutkan
setelah Colab terputus. Model ringan hasil ekspor (`cubenet_large_infer.pt`, ~29 MB) dipakai di laptop:

```bash
python -m web.server --model models/cubenet_large_infer.pt
python -m scripts.beam_experiment --model models/cubenet_large_infer.pt --name beam_experiment_large_cpu
```

## DeepCubeA (pendekatan kedua)

`rubik/deepcubea.py` berisi DeepCubeA versi diperkecil: value network (perkiraan jarak ke kondisi
selesai) yang dilatih dengan Deep Approximate Value Iteration, dan solver batch weighted A*.

```bash
python -m scripts.train_deepcubea --steps 3000                          # uji cepat di CPU
python -m scripts.beam_experiment --model models/deepcubea_large_infer.pt --solver astar --beams 100 1000 --name deepcubea_cpu
python -m scripts.compare_results beam_experiment_large:"EfficientCube besar" deepcubea_large:"DeepCubeA besar"
```

Training model besar: `notebooks/train_deepcubea_colab.ipynb` di Google Colab.

## Kubus 4x4 (eksplorasi)

`rubik/nxn.py` berisi simulator kubus N x N generik (terverifikasi identik dengan simulator 3x3).
Pada 4x4 tidak ada pusat yang tetap, jadi **sudut kuning-biru-oranye dikunci di bawah-belakang-kiri**
dan AI memakai 18 gerakan lapisan dari sisi U, R, F (notasi SiGN: U, 2U, 3U, R, 2R, ...).

```bash
python -m scripts.train --puzzle 4 --max-depth 30 --save models/cubenet_4x4.pt   # training
python -m scripts.eval_4x4 --model models/cubenet_4x4.pt --beams 256 1024        # evaluasi per tingkat kesulitan
python -m scripts.solve_ai_4x4 --depth 15                                         # coba dengan kubus fisik
```