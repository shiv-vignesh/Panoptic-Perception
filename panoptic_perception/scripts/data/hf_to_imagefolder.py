"""
Download ImageNet-1k from HuggingFace and convert to standard ImageFolder layout
that panoptic_perception.dataset.imagenet_dataset.ImageNetDataset expects:

    <root>/train/<wnid>/*.JPEG
    <root>/val/<wnid>/*.JPEG

Downloads only train + validation parquets (skips the ~14 GB test split we don't
need), then extracts image bytes with pyarrow. Idempotent — skips a split whose
extracted image count already matches the expected total. If a prior buggy run
extracted train into human-readable class dirs (e.g. "tench, Tinca tinca"),
those get renamed to wnids on the fly.

Usage:
    python -m panoptic_perception.scripts.data.hf_to_imagefolder /workspace/data/imagenet

Requires HF login first:
    hf auth login               # accept ILSVRC/imagenet-1k terms on the website beforehand
    export HF_XET_HIGH_PERFORMANCE=1

Storage: ~150 GB permanent (extracted JPEGs), ~140 GB transient (parquet cache),
so peak is ~290 GB. HF snapshot dir is auto-deleted after extraction unless
--keep-cache is passed.
"""

import argparse
import io
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import get_token, snapshot_download
from PIL import Image
from tqdm import tqdm


REPO_ID = "ILSVRC/imagenet-1k"
REPO_TYPE = "dataset"

ALLOW_PATTERNS = [
    "data/train-*.parquet",
    "data/validation-*.parquet",
    "*.md",
    "*.json",
]

TRAIN_TOTAL = 1281167
VAL_TOTAL = 50000

# Canonical ILSVRC label_idx -> [wnid, short_name] mapping (Keras / TF convention).
# Matches HF's imagenet-1k label ordering exactly.
IMAGENET_CLASS_INDEX_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
)


def load_id2wnid() -> list:
    """Return list of 1000 wnids in HF's label_idx order."""
    with urllib.request.urlopen(IMAGENET_CLASS_INDEX_URL, timeout=30) as f:
        idx_map = json.load(f)
    return [idx_map[str(i)][0] for i in range(1000)]


def load_id2hf_name() -> list:
    """Return list of 1000 HF display names (streaming schema fetch, no data)."""
    from datasets import load_dataset
    ds = load_dataset(REPO_ID, split="train", streaming=True)
    return ds.features["label"].names


def rename_display_dirs_to_wnid(split_target: Path, id2wnid: list, id2hf_name: list):
    """One-shot migration: if a prior buggy run wrote display-name dirs, rename them."""
    name_to_wnid = {name: id2wnid[i] for i, name in enumerate(id2hf_name)}
    renamed = 0
    for d in split_target.iterdir():
        if not d.is_dir():
            continue
        if d.name in name_to_wnid:
            wnid = name_to_wnid[d.name]
            if wnid != d.name:
                target = split_target / wnid
                if target.exists():
                    # merge: move files from d into wnid dir, then remove d
                    for f in d.iterdir():
                        f.rename(target / f.name)
                    d.rmdir()
                else:
                    d.rename(target)
                renamed += 1
    if renamed:
        print(f"[migrate] renamed {renamed} display-name dirs to wnids under {split_target}")


def already_extracted(target: Path, expected: int) -> bool:
    if not target.is_dir():
        return False
    n = sum(1 for _ in target.rglob("*.JPEG"))
    return n >= expected


def convert_parquets(snapshot_dir: Path, out_root: Path, split_glob: str, split_dir: str,
                    id2wnid: list):
    parquet_files = sorted((snapshot_dir / "data").glob(split_glob))
    if not parquet_files:
        raise RuntimeError(f"no parquets matched {split_glob} under {snapshot_dir}/data")

    target = out_root / split_dir
    target.mkdir(parents=True, exist_ok=True)
    for wnid in id2wnid:
        (target / wnid).mkdir(exist_ok=True)

    running_idx = 0
    for pq_path in tqdm(parquet_files, desc=f"{split_dir}: parquets", unit="file"):
        table = pq.read_table(pq_path)
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        for img_obj, label in tqdm(zip(images, labels), total=len(labels),
                                    desc=pq_path.name, leave=False):
            raw = img_obj["bytes"] if isinstance(img_obj, dict) else img_obj
            wnid = id2wnid[label]
            out_path = target / wnid / f"{split_dir}_{running_idx}.JPEG"
            with Image.open(io.BytesIO(raw)) as im:
                im.convert("RGB").save(out_path, "JPEG", quality=95)
            running_idx += 1


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
                    help="HF hub cache dir. Default: $HF_HOME/hub if set, else HF's default.")
    ap.add_argument("--keep-cache", action="store_true",
                    help="do NOT delete the HF hub snapshot after conversion")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    if get_token() is None:
        print("[hf-to-imagefolder] no HF token found. Run `hf auth login` first "
              "(and accept ILSVRC/imagenet-1k terms on the dataset page).", file=sys.stderr)
        sys.exit(1)

    print("[hf-to-imagefolder] resolving wnid ordering (imagenet_class_index.json)")
    id2wnid = load_id2wnid()

    print("[hf-to-imagefolder] snapshot_download (train + validation parquets)")
    snapshot_dir = Path(snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=ALLOW_PATTERNS,
        cache_dir=args.cache_dir,
        max_workers=8,
    ))
    print(f"[hf-to-imagefolder] snapshot at {snapshot_dir}")

    # Fix any display-name dirs left over from a prior buggy extraction.
    train_target = out_root / "train"
    val_target = out_root / "val"
    if train_target.is_dir() or val_target.is_dir():
        print("[hf-to-imagefolder] checking for display-name dirs to migrate")
        id2hf_name = load_id2hf_name()
        if train_target.is_dir():
            rename_display_dirs_to_wnid(train_target, id2wnid, id2hf_name)
        if val_target.is_dir():
            rename_display_dirs_to_wnid(val_target, id2wnid, id2hf_name)

    if already_extracted(train_target, TRAIN_TOTAL):
        print(f"[hf-to-imagefolder] train already extracted ({TRAIN_TOTAL} imgs); skipping")
    else:
        convert_parquets(snapshot_dir, out_root, "train-*.parquet", "train", id2wnid)

    if already_extracted(val_target, VAL_TOTAL):
        print(f"[hf-to-imagefolder] val already extracted ({VAL_TOTAL} imgs); skipping")
    else:
        convert_parquets(snapshot_dir, out_root, "validation-*.parquet", "val", id2wnid)

    verify(out_root)

    if not args.keep_cache:
        # snapshot_dir points to .../snapshots/<hash>/; the datasets--<repo> root is
        # two levels up. Remove the whole cache to reclaim disk.
        cache_root = snapshot_dir.parent.parent
        print(f"[hf-to-imagefolder] cleaning HF cache at {cache_root}")
        shutil.rmtree(cache_root, ignore_errors=True)


if __name__ == "__main__":
    main()
