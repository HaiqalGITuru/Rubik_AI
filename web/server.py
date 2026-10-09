"""
Server lokal untuk website Rubik AI (kubus 3x3).

Jalankan dari folder root proyek:
    python -m web.server
    python -m web.server --model models/cubenet_large_infer.pt      # model lain
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

# Jumlah neuron yang ditampilkan per lapisan tersembunyi (sampel dari ribuan neuron asli)
SHOW_NEURONS = [20, 18, 18, 18]

try:
    import kociemba
except ImportError:
    kociemba = None


# ---------------------------------------------------------------------- #
# Model
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
    """Model beserta semua yang dibutuhkan website."""

    def __init__(self, path: Path):
        ckpt = torch.load(path, map_location="cpu")
        self.model = CubeNet(**ckpt["config"]).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.moves, self.perms = QUARTER_MOVES, QUARTER_PERMS
        self.info = {
            "model_file": path.name,
            "train_steps": int(ckpt.get("step", 0)),
            "parameters": sum(t.numel() for t in self.model.parameters()),
            "layers": [54 * 6, self.model.config["hidden1"], self.model.config["hidden2"],
                       self.model.config["hidden2"], self.model.config["hidden2"], len(self.moves)],
            "moves": self.moves,
            "show_neurons": SHOW_NEURONS,
            "kociemba": kociemba is not None,
        }
        self._pick_neurons()

    def _pick_neurons(self):
        """Pilih neuron yang paling 'hidup' (variansi tertinggi) pada kubus acak."""
        rng = np.random.default_rng(0)
        states = []
        for _ in range(512):
            c = Cube()
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


BUNDLE: Bundle | None = None


def get_bundle() -> Bundle:
    if BUNDLE is None:
        raise HTTPException(503, "Model belum dimuat. Latih dulu: python -m scripts.train")
    return BUNDLE


# ---------------------------------------------------------------------- #
# Utilitas solusi
# ---------------------------------------------------------------------- #
def group_moves(raw: list[str]):
    """Gabungkan gerakan seperempat berurutan pada wajah yang sama (R R -> R2).
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


def normalize_tokens(text: str) -> list[str]:
    """Rapikan tanda petik & huruf kecil: "r2'" -> "R2", "u" -> "U"."""
    text = text.replace("’", "'").replace("`", "'").replace("′", "'").replace(",", " ")
    return parse_sequence([t[0].upper() + t[1:].replace("2'", "2") for t in text.split()])


# ---------------------------------------------------------------------- #
# API
# ---------------------------------------------------------------------- #
app = FastAPI(title="Rubik AI")


@app.middleware("http")
async def no_cache(request, call_next):
    """Supaya browser selalu memuat app.js/style.css terbaru (tidak perlu Ctrl+Shift+R)."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


class ScrambleRequest(BaseModel):
    length: int | None = Field(None, ge=1, le=60)


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
    return get_bundle().info


@app.post("/api/scramble")
def scramble(req: ScrambleRequest):
    get_bundle()
    return {"moves": Cube().scramble(req.length or 20)}


@app.post("/api/parse")
def parse(req: ParseRequest):
    get_bundle()
    try:
        moves = normalize_tokens(req.text)
    except ValueError as e:
        raise HTTPException(400, f"{e}. Gunakan U D R L F B dengan akhiran ' atau 2, dipisah spasi.")
    if not moves:
        raise HTTPException(400, "Scramble masih kosong. Tulis gerakan seperti R U R' U'.")
    return {"moves": moves}


@app.post("/api/analyze")
def analyze(req: SolveRequest):
    """Aktivasi jaringan untuk kubus saat ini (tanpa mencari solusi)."""
    b = get_bundle()
    try:
        cube = Cube().apply_sequence(req.moves)
    except ValueError as e:
        raise HTTPException(400, str(e))
    frame = b.frames(cube.state[None])[0]
    if not cube.is_solved():
        frame["chosen"] = int(np.argmax(frame["output"]))
    return {"frame": frame, "solved": cube.is_solved()}


@app.post("/api/solve")
def solve(req: SolveRequest):
    b = get_bundle()
    try:
        cube = Cube().apply_sequence(req.moves)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if cube.is_solved():
        return {"solved": True, "moves": [], "steps": [], "frames": b.frames(cube.state[None]),
                "stats": {"length": 0, "time_ms": 0, "nodes": 0, "beam": req.beam, "avg_confidence": 1.0},
                "kociemba": None}

    res = beam_search(b.model, cube, beam_width=req.beam, max_depth=40)
    if not res.solved:
        return {"solved": False, "message": (
            f"AI belum menemukan solusi dengan lebar pencarian {req.beam} ({res.time_sec:.1f} dtk). "
            f"Coba pilih yang lebih lebar.")}

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

    k = None
    if kociemba:
        t0 = time.perf_counter()
        ksol = kociemba.solve(cube.to_facelet_string()).split()
        k = {"moves": ksol, "length": len(ksol), "time_ms": (time.perf_counter() - t0) * 1000}

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
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main():
    global BUNDLE
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/cubenet.pt")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    import uvicorn

    path = Path(args.model) if Path(args.model).is_absolute() else ROOT / args.model
    if not path.exists():
        raise SystemExit(f"Model tidak ditemukan: {path}. Latih dulu: python -m scripts.train")
    print(f"Memuat model: {path.name} ...")
    BUNDLE = Bundle(path)

    url = f"http://127.0.0.1:{args.port}"
    print(f"Website siap di {url}  (tekan Ctrl+C untuk berhenti)")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
