"""
Export a model for inference: no optimizer state, stored in float16.

A training checkpoint holds the model plus the Adam optimizer state (3x the model size).
The website, evaluation and GitHub only need the model weights.

    python -m scripts.export_model models/cubenet_large.pt models/cubenet_large_infer.pt
"""

import argparse
import os

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", help="training checkpoint")
    p.add_argument("dst", help="exported file")
    p.add_argument("--float32", action="store_true", help="store float32 (default float16, 2x smaller)")
    args = p.parse_args()

    ckpt = torch.load(args.src, map_location="cpu")
    state = {k: (v.half() if v.is_floating_point() and not args.float32 else v)
             for k, v in ckpt["state_dict"].items()}
    torch.save({"config": ckpt["config"], "state_dict": state, "step": ckpt.get("step", 0)}, args.dst)
    mb = lambda f: os.path.getsize(f) / 1e6
    print(f"{args.src} ({mb(args.src):.0f} MB) -> {args.dst} ({mb(args.dst):.0f} MB), step {ckpt.get('step', 0)}")


if __name__ == "__main__":
    main()
