"""
Server lokal untuk website Rubik AI.

Jalankan dari folder root proyek:
    python -m web.server
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
from rubik.solver import beam_search

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
MODEL_PATH = ROOT / "models" / "cubenet.pt"

# Jumlah neuron yang ditampilkan per lapisan tersembunyi (sampel dari ribuan neuron asli)
SHOW_NEURONS = [20, 18, 18, 18]

try:
    import kociemba
except ImportError:
    kociemba = None


# ---------------------------------------------------------------------- #
# Model & aktivasi neuron
# ---------------------------------------------------------------------- #
def load(path: Path) -> tuple[CubeNet, dict]:
    if not path.exists():
        raise SystemExit(f"Model tidak ditemukan di {path}. Latih dulu: python -m scripts.train")
    ckpt = torch.load(path, map_location="cpu")
    model = CubeNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    info = {
        "train_steps": int(ckpt.get("step", 0)),
        "parameters": sum(p.numel() for p in model.parameters()),
        "layers": [324, model.config["hidden1"], model.config["hidden2"],
                   model.config["hidden2"], model.config["hidden2"], 12],
    }
    return model, info


@torch.no_grad()
def forward_with_activations(model: CubeNet, states: np.ndarray):
    """Forward pass yang juga mengembalikan aktivasi lapisan tersembunyi.

    Lapisan yang ditampilkan: dense 1, dense 2, setelah residual block ke-2, setelah block terakhir.
    """
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


def pick_neurons(model: CubeNet):
    """Pilih neuron yang paling 'hidup' (variansi aktivasi tertinggi) pada kubus acak,
    supaya animasinya informatif. Juga hitung skala normalisasi per lapisan."""
    rng = np.random.default_rng(0)
    states = []
    for _ in range(512):
        c = Cube()
        c.scramble(int(rng.integers(1, 26)), rng=rng)
        states.append(c.state)
    layers, _ = forward_with_activations(model, np.stack(states))
    picks, scales = [], []
    for acts, k in zip(layers, SHOW_NEURONS):
        idx = np.sort(np.argsort(acts.var(axis=0))[-k:])
        picks.append(idx)
        scales.append(float(np.percentile(acts[:, idx], 99)) or 1.0)
    return picks, scales


def network_frame(states: np.ndarray, picks, scales):
    layers, probs = forward_with_activations(MODEL, states)
    frames = []
    for n in range(len(states)):
        frames.append({
            "input": states[n].tolist(),
            "hidden": [np.clip(l[n, idx] / s, 0, 1).round(3).tolist()
                       for l, idx, s in zip(layers, picks, scales)],
            "output": probs[n].round(4).tolist(),
        })
    return frames


# ---------------------------------------------------------------------- #
# Utilitas solusi
# ---------------------------------------------------------------------- #
def group_moves(raw: list[str]):
    """Gabungkan gerakan seperempat berurutan pada wajah yang sama (R R -> R2).
    Mengembalikan list (notasi, indeks gerakan mentah pertama di grup)."""
    groups: list[list] = []
    for i, m in enumerate(raw):
        face, amount = m[0], 3 if m.endswith("'") else 1
        if groups and groups[-1][0] == face:
            groups[-1][1] = (groups[-1][1] + amount) % 4
            if groups[-1][1] == 0:
                groups.pop()
            continue
        groups.append([face, amount, i])
    return [(f + {1: "", 2: "2", 3: "'"}[a], start) for f, a, start in groups]


def normalize_notation(text: str) -> list[str]:
    text = text.replace("’", "'").replace("`", "'").replace("′", "'").replace(",", " ")
    # "r2'" -> "R2", "u" -> "U"
    moves = [t[0].upper() + t[1:].replace("2'", "2") for t in text.split()]
    return parse_sequence(moves)


# ---------------------------------------------------------------------- #
# API
# ---------------------------------------------------------------------- #
app = FastAPI(title="Rubik AI")


class ScrambleRequest(BaseModel):
    length: int = Field(20, ge=1, le=60)


class ParseRequest(BaseModel):
    text: str


class SolveRequest(BaseModel):
    moves: list[str] = Field(default_factory=list, description="semua gerakan sejak kubus selesai")
    beam: int = Field(1024, ge=16, le=16384)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    return {**MODEL_INFO, "kociemba": kociemba is not None, "show_neurons": SHOW_NEURONS,
            "moves": QUARTER_MOVES}


@app.post("/api/scramble")
def scramble(req: ScrambleRequest):
    return {"moves": Cube().scramble(req.length)}


@app.post("/api/parse")
def parse(req: ParseRequest):
    try:
        moves = normalize_notation(req.text)
    except ValueError as e:
        raise HTTPException(400, f"{e}. Gunakan U D R L F B dengan akhiran ' atau 2, dipisah spasi.")
    if not moves:
        raise HTTPException(400, "Scramble masih kosong. Tulis gerakan seperti R U R' U'.")
    return {"moves": moves}


@app.post("/api/analyze")
def analyze(req: SolveRequest):
    """Aktivasi jaringan untuk state kubus saat ini (tanpa mencari solusi)."""
    try:
        cube = Cube().apply_sequence(req.moves)
    except ValueError as e:
        raise HTTPException(400, str(e))
    frame = network_frame(cube.state[None], PICKS, SCALES)[0]
    if not cube.is_solved():
        frame["chosen"] = int(np.argmax(frame["output"]))
    return {"frame": frame, "solved": cube.is_solved()}


@app.post("/api/solve")
def solve(req: SolveRequest):
    try:
        cube = Cube().apply_sequence(req.moves)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if cube.is_solved():
        frame = network_frame(cube.state[None], PICKS, SCALES)[0]
        return {"solved": True, "moves": [], "frames": [frame], "stats": {
            "length": 0, "time_ms": 0, "nodes": 0, "beam": req.beam}, "kociemba": None}

    res = beam_search(MODEL, cube, beam_width=req.beam, max_depth=40)
    if not res.solved:
        return {"solved": False, "message": (
            f"AI belum menemukan solusi dengan beam {req.beam} ({res.time_sec:.1f} dtk). "
            "Coba pilih beam yang lebih lebar.")}

    # Semua state di sepanjang jalur solusi (sebelum setiap gerakan + state akhir)
    states = [cube.state]
    for m in res.moves:
        states.append(states[-1][QUARTER_PERMS[QUARTER_MOVES.index(m)]])
    states = np.stack(states)

    grouped = group_moves(res.moves)
    starts = [s for _, s in grouped] + [len(res.moves)]
    frames = network_frame(states[starts], PICKS, SCALES)

    steps = []
    for (notation, start), frame in zip(grouped, frames):
        chosen = QUARTER_MOVES.index(res.moves[start])
        frame["chosen"] = chosen
        steps.append({"move": notation, "confidence": frame["output"][chosen]})

    k = None
    if kociemba:
        t0 = time.perf_counter()
        ksol = kociemba.solve(cube.to_facelet_string()).split()
        k = {"moves": ksol, "length": len(ksol), "time_ms": (time.perf_counter() - t0) * 1000}

    final = cube.copy().apply_sequence([s["move"] for s in steps])
    return {
        "solved": final.is_solved(),
        "moves": [s["move"] for s in steps],
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
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main():
    global MODEL, MODEL_INFO, PICKS, SCALES
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    import uvicorn

    print("Memuat model...")
    MODEL, MODEL_INFO = load(Path(args.model))
    PICKS, SCALES = pick_neurons(MODEL)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Website siap di {url}  (tekan Ctrl+C untuk berhenti)")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
