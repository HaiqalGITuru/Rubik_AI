"""
Ekspor model untuk dipakai (inference), tanpa state optimizer, dalam float16.

Checkpoint training berisi model + state optimizer Adam (3x ukuran model).
Untuk website, evaluasi, dan GitHub, cukup bobot modelnya saja.

    python -m scripts.export_model models/cubenet_large.pt models/cubenet_large_infer.pt
"""

import argparse
import os

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", help="checkpoint training")
    p.add_argument("dst", help="file hasil ekspor")
    p.add_argument("--float32", action="store_true", help="simpan float32 (default float16, 2x lebih kecil)")
    args = p.parse_args()

    ckpt = torch.load(args.src, map_location="cpu")
    state = {k: (v.half() if v.is_floating_point() and not args.float32 else v)
             for k, v in ckpt["state_dict"].items()}
    torch.save({"config": ckpt["config"], "state_dict": state, "step": ckpt.get("step", 0)}, args.dst)
    mb = lambda f: os.path.getsize(f) / 1e6
    print(f"{args.src} ({mb(args.src):.0f} MB) -> {args.dst} ({mb(args.dst):.0f} MB), langkah {ckpt.get('step', 0)}")


if __name__ == "__main__":
    main()
