# Rubik AI — Solver Kubus Rubik 3x3 dengan Deep Learning

## Struktur Proyek

```
rubik-ai/
├── rubik/              # Paket utama: simulator kubus
│   ├── __init__.py
│   ├── moves.py        # Definisi & permutasi 18 gerakan (dihitung dari geometri 3D)
│   └── cube.py         # Kelas Cube: apply, scramble, is_solved, one-hot, render
├── tests/
│   └── test_cube.py    # Unit test simulator (47 test)
├── scripts/
│   └── demo.py         # Demo simulator + pembanding Kociemba
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
2. [ ] Generator data (scramble acak)
3. [ ] Model jaringan saraf (PyTorch)
4. [ ] Solver: beam search
5. [ ] Evaluasi vs Kociemba
