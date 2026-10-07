"""
Server lokal untuk website Rubik AI (kubus 3x3 dan 4x4).

Jalankan dari folder root proyek:
    python -m web.server
    python -m web.server --model models/cubenet_large_infer.pt      # model 3x3 lain
    python -m web.server --model4 models/cubenet_4x4.pt             # model 4x4 (default)
lalu buka http://127.0.0.1:8000 (browser terbuka otomatis).

Server hanya mendengarkan di 127.0.0.1, jadi hanya bisa diakses dari laptop Anda sendiri.
"""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rubik import Cube, parse_sequence
from rubik.data import QUARTER_PERMS
from rubik.model import CubeNet
from rubik.moves import QUARTER_MOVES
from rubik.nxn import CubeN, get_puzzle
from rubik.solver import beam_search, compress_moves

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"

# Jumlah neuron yang ditampilkan per lapisan tersembunyi (sampel dari ribuan neuron asli)
SHOW_NEURONS = [20, 18, 18, 18]

try:
    import kociemba
except ImportError:
    kociemba = None


# ---------------------------------------------------------------------- #
# Model per ukuran kubus
# ---------------------------------------------------------------------- #
@torch.no_grad()
def forward_with_activations(model: CubeNet, states: np.ndarray):
    """Forward pass yang juga mengembalikan aktivasi 4 lapisan tersembunyi."""
    x = F.one_hot(torch.from_numpy(states).long(), num_classes=6).float().flatten(1)
    h1 = model.input[:3](x)
    h = model.input[3:](h1)
    layers = [h1, h]
    mid = len(model.blocks) // 2
    for i, block in enumerate(model.blocks):
        h = block(h)
        if i == mid - 1:
            layers.append(h)
    layers.append(h)
    probs = torch.softmax(model.output(h), dim=1)
    return [l.numpy() for l in layers], probs.numpy()


class Bundle:
    """Satu model beserta semua yang dibutuhkan untuk satu ukuran kubus."""

    def __init__(self, size: int, path: Path):
        self.size = size
        ckpt = torch.load(path, map_location="cpu")
        self.model = CubeNet(**ckpt["config"]).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        if size == 3:
            self.moves, self.perms = QUARTER_MOVES, QUARTER_PERMS
        else:
            p = get_puzzle(size)
            self.moves, self.perms = p.quarter_moves, p.quarter_perms
        self.info = {
            "size": size,
            "model_file": path.name,
            "train_steps": int(ckpt.get("step", 0)),
            "parameters": sum(t.numel() for t in self.model.parameters()),
            "layers": [6 * size * size * 6, self.model.config["hidden1"], self.model.config["hidden2"],
                       self.model.config["hidden2"], self.model.config["hidden2"], len(self.moves)],
            "moves": self.moves,
            "show_neurons": SHOW_NEURONS,
            "kociemba": size == 3 and kociemba is not None,
        }
        self._pick_neurons()

    def new_cube(self, moves=()):
        cube = Cube() if self.size == 3 else CubeN(self.size)
        return cube.apply_sequence(list(moves))

    def _pick_neurons(self):
        """Pilih neuron yang paling 'hidup' (variansi tertinggi) pada kubus acak."""
        rng = np.random.default_rng(0)
        states = []
        for _ in range(512):
            c = self.new_cube()
            c.scramble(int(rng.integers(1, 26)), rng=rng)
            states.append(c.state)
        layers, _ = forward_with_activations(self.model, np.stack(states))
        self.picks, self.scales = [], []
        for acts, k in zip(layers, SHOW_NEURONS):
            idx = np.sort(np.argsort(acts.var(axis=0))[-k:])
            self.picks.append(idx)
            self.scales.append(float(np.percentile(acts[:, idx], 99)) or 1.0)

    def frames(self, states: np.ndarray):
        layers, probs = forward_with_activations(self.model, states)
        return [{
            "input": states[n].tolist(),
            "hidden": [np.clip(l[n, idx] / s, 0, 1).round(3).tolist()
                       for l, idx, s in zip(layers, self.picks, self.scales)],
            "output": probs[n].round(4).tolist(),
        } for n in range(len(states))]


BUNDLES: dict[int, Bundle] = {}


def get_bundle(size: int) -> Bundle:
    if size not in BUNDLES:
        raise HTTPException(404, f"Model {size}x{size} belum dimuat. Latih dulu: "
                                 f"python -m scripts.train --puzzle {size} --save models/cubenet_{size}x{size}.pt")
    return BUNDLES[size]


# ---------------------------------------------------------------------- #
# Utilitas solusi
# ---------------------------------------------------------------------- #
def group_moves(raw: list[str]):
    """Gabungkan gerakan seperempat berurutan pada lapisan yang sama (R R -> R2, 2U 2U -> 2U2).
    Mengembalikan list (notasi, indeks gerakan mentah pertama di grup)."""
    groups: list[list] = []
    for i, m in enumerate(raw):
        base, amount = (m[:-1], 3) if m.endswith("'") else (m, 1)
        if groups and groups[-1][0] == base:
            groups[-1][1] = (groups[-1][1] + amount) % 4
            if groups[-1][1] == 0:
                groups.pop()
            continue
        groups.append([base, amount, i])
    return [(b + {1: "", 2: "2", 3: "'"}[a], start) for b, a, start in groups]


def normalize_tokens(text: str, size: int) -> list[str]:
    text = text.replace("’", "'").replace("`", "'").replace("′", "'").replace(",", " ")
    tokens = text.split()
    if size == 3:  # "r2'" -> "R2", "u" -> "U"
        return parse_sequence([t[0].upper() + t[1:].replace("2'", "2") for t in tokens])
    tokens = [t.replace("2'", "2") for t in tokens]
    puzzle = get_puzzle(size)
    for t in tokens:
        puzzle.token_perm(t)  # memvalidasi; huruf kecil (r, u) = gerakan lebar
    return tokens


# ---------------------------------------------------------------------- #
# API
# ---------------------------------------------------------------------- #
app = FastAPI(title="Rubik AI")


class ScrambleRequest(BaseModel):
    size: int = 3
    length: int | None = Field(None, ge=1, le=60)


class ParseRequest(BaseModel):
    size: int = 3
    text: str


class SolveRequest(BaseModel):
    size: int = 3
    moves: list[str] = Field(default_factory=list, description="semua gerakan sejak kubus selesai")
    beam: int = Field(1024, ge=16, le=16384)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    return {"sizes": {str(k): b.info for k, b in BUNDLES.items()}}


@app.post("/api/scramble")
def scramble(req: ScrambleRequest):
    b = get_bundle(req.size)
    if req.size == 3:
        return {"moves": Cube().scramble(req.length or 20)}
    # 4x4: AI andal sampai sekitar 15 gerakan seperempat (lihat eksperimen 4)
    return {"moves": compress_moves(CubeN(req.size).scramble(req.length or 15))}


@app.post("/api/parse")
def parse(req: ParseRequest):
    get_bundle(req.size)
    try:
        moves = normalize_tokens(req.text, req.size)
    except ValueError as e:
        hint = ("U D R L F B dengan akhiran ' atau 2" if req.size == 3
                else "U R F D L B, lapisan 2U/3R, lebar Uw/Rw, rotasi x/y/z, dengan akhiran ' atau 2")
        raise HTTPException(400, f"{e}. Gunakan {hint}, dipisah spasi.")
    if not moves:
        raise HTTPException(400, "Scramble masih kosong. Tulis gerakan seperti R U R' U'.")
    return {"moves": moves}


@app.post("/api/analyze")
def analyze(req: SolveRequest):
    """Aktivasi jaringan + state baku kubus saat ini (tanpa mencari solusi)."""
    b = get_bundle(req.size)
    try:
        cube = b.new_cube(req.moves)
    except ValueError as e:
        raise HTTPException(400, str(e))
    frame = b.frames(cube.state[None])[0]
    if not cube.is_solved():
        frame["chosen"] = int(np.argmax(frame["output"]))
    return {"frame": frame, "solved": cube.is_solved(), "state": cube.state.tolist()}


@app.post("/api/solve")
def solve(req: SolveRequest):
    b = get_bundle(req.size)
    try:
        cube = b.new_cube(req.moves)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if cube.is_solved():
        return {"solved": True, "moves": [], "steps": [], "frames": b.frames(cube.state[None]),
                "stats": {"length": 0, "time_ms": 0, "nodes": 0, "beam": req.beam, "avg_confidence": 1.0},
                "kociemba": None, "wca": None}

    res = beam_search(b.model, cube, beam_width=req.beam, max_depth=40 if req.size == 3 else 60)
    if not res.solved:
        tip = "" if req.size == 3 else " Untuk 4x4, AI andal sampai scramble sekitar 15 gerakan."
        return {"solved": False, "message": (
            f"AI belum menemukan solusi dengan lebar pencarian {req.beam} ({res.time_sec:.1f} dtk). "
            f"Coba pilih yang lebih lebar.{tip}")}

    # Semua state di sepanjang jalur solusi (sebelum setiap gerakan + state akhir)
    states = [cube.state]
    for m in res.moves:
        states.append(states[-1][b.perms[b.moves.index(m)]])
    states = np.stack(states)

    grouped = group_moves(res.moves)
    starts = [s for _, s in grouped] + [len(res.moves)]
    frames = b.frames(states[starts])
    steps = []
    for (notation, start), frame in zip(grouped, frames):
        chosen = b.moves.index(res.moves[start])
        frame["chosen"] = chosen
        steps.append({"move": notation, "confidence": frame["output"][chosen]})
    moves = [s["move"] for s in steps]

    k, wca = None, None
    if req.size == 3 and kociemba:
        t0 = time.perf_counter()
        ksol = kociemba.solve(cube.to_facelet_string()).split()
        k = {"moves": ksol, "length": len(ksol), "time_ms": (time.perf_counter() - t0) * 1000}
    if req.size == 4:
        from scripts.solve_ai_4x4 import to_wca
        wca = to_wca(moves)

    return {
        "solved": cube.copy().apply_sequence(moves).is_solved(),
        "moves": moves,
        "steps": steps,
        "frames": frames,
        "stats": {
            "length": len(steps),
            "quarter_length": len(res.moves),
            "time_ms": res.time_sec * 1000,
            "nodes": res.nodes,
            "beam": req.beam,
            "avg_confidence": float(np.mean([s["confidence"] for s in steps])),
        },
        "kociemba": k,
        "wca": wca,
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/cubenet.pt", help="model kubus 3x3")
    p.add_argument("--model4", default="models/cubenet_4x4.pt", help="model kubus 4x4 (opsional)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    import uvicorn

    for size, path in ((3, args.model), (4, args.model4)):
        path = Path(path) if Path(path).is_absolute() else ROOT / path
        if path.exists():
            print(f"Memuat model {size}x{size}: {path.name} ...")
            BUNDLES[size] = Bundle(size, path)
        else:
            print(f"Model {size}x{size} tidak ditemukan ({path}), dilewati.")
    if not BUNDLES:
        raise SystemExit("Tidak ada model yang bisa dimuat. Latih dulu: python -m scripts.train")

    url = f"http://127.0.0.1:{args.port}"
    print(f"Website siap di {url}  (tekan Ctrl+C untuk berhenti)")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()