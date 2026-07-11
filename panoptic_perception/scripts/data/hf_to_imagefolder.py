"""
Download ImageNet-1k from HuggingFace and convert to standard ImageFolder layout
that panoptic_perception.dataset.imagenet_dataset.ImageNetDataset expects:

    <root>/train/<wnid>/*.JPEG
    <root>/val/<wnid>/*.JPEG

Usage:
    python -m panoptic_perception.scripts.data.hf_to_imagefolder /workspace/data/imagenet

Requires HF login first:
    huggingface-cli login       # accept ILSVRC/imagenet-1k terms on the website beforehand
    export HF_HUB_ENABLE_HF_TRANSFER=1

Storage: ~300 GB peak (HF parquet cache + extracted JPEGs), ~150 GB after cleanup.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def convert_split(out_root: Path, split_hf: str, split_dir: str, cache_dir: str):
    print(f"[hf-to-imagefolder] loading split '{split_hf}' (cache={cache_dir}) ...", flush=True)
    ds = load_dataset("ILSVRC/imagenet-1k", split=split_hf, cache_dir=cache_dir)
    id2wnid = ds.features["label"].names  # index -> wnid string (e.g. 'n01440764')

    target = out_root / split_dir
    target.mkdir(parents=True, exist_ok=True)

    # Pre-create wnid dirs so mkdir isn't called 1.28M times.
    for wnid in id2wnid:
        (target / wnid).mkdir(exist_ok=True)

    for i, ex in enumerate(tqdm(ds, desc=split_dir, unit="img")):
        wnid = id2wnid[ex["label"]]
        out_path = target / wnid / f"{split_dir}_{i}.JPEG"
        # HF returns a PIL.Image. Some ImageNet examples are grayscale; force RGB.
        ex["image"].convert("RGB").save(out_path, "JPEG", quality=95)


def verify(out_root: Path):
    for split in ("train", "val"):
        split_dir = out_root / split
        if not split_dir.is_dir():
            print(f"[verify] MISSING: {split_dir}")
            continue
        wnid_count = sum(1 for _ in split_dir.iterdir() if _.is_dir())
        img_count = sum(1 for _ in split_dir.rglob("*.JPEG"))
        print(f"[verify] {split}: {wnid_count} wnid dirs, {img_count} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root", help="target dir, e.g. /workspace/data/imagenet")
    ap.add_argument("--cache-dir", default=None,
                    help="HF cache dir. Default: <out_root>/hf_cache")
    ap.add_argument("--keep-cache", action="store_true",
                    help="do NOT delete the HF cache after conversion")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir or str(out_root / "hf_cache")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    if not os.environ.get("HF_TOKEN") and not (Path.home() / ".cache/huggingface/token").exists():
        print("[hf-to-imagefolder] no HF token found. Run `huggingface-cli login` first "
              "(and accept ILSVRC/imagenet-1k terms on the dataset page).", file=sys.stderr)
        sys.exit(1)

    convert_split(out_root, split_hf="train", split_dir="train", cache_dir=cache_dir)
    convert_split(out_root, split_hf="validation", split_dir="val", cache_dir=cache_dir)

    verify(out_root)

    if not args.keep_cache:
        print(f"[hf-to-imagefolder] cleaning HF cache at {cache_dir} ...")
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
