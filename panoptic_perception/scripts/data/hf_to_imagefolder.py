"""
Download ImageNet-1k from HuggingFace and convert to standard ImageFolder layout
that panoptic_perception.dataset.imagenet_dataset.ImageNetDataset expects:

    <root>/train/<wnid>/*.JPEG
    <root>/val/<wnid>/*.JPEG

Downloads only train + validation parquets (skips the ~14 GB test split we don't
need), then extracts image bytes with pyarrow. Reuses any parquets already in the
HF hub cache from a prior interrupted run.

Usage:
    python -m panoptic_perception.scripts.data.hf_to_imagefolder /workspace/data/imagenet

Requires HF login first:
    hf auth login               # accept ILSVRC/imagenet-1k terms on the website beforehand
    export HF_XET_HIGH_PERFORMANCE=1

Storage: ~150 GB permanent (extracted JPEGs), ~140 GB transient (parquet cache),
so peak is ~290 GB. HF cache is auto-deleted after extraction unless --keep-cache
is passed.
"""

import argparse
import io
import os
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import get_token, snapshot_download
from PIL import Image
from tqdm import tqdm


REPO_ID = "ILSVRC/imagenet-1k"
REPO_TYPE = "dataset"

# HF stores this dataset's parquets under data/train-*.parquet and data/val-*.parquet.
# Explicit patterns skip the test split (~14 GB) we don't need.
ALLOW_PATTERNS = [
    "data/train-*.parquet",
    "data/val-*.parquet",
    "*.md",
    "*.json",
]


def get_wnid_mapping(snapshot_dir: Path) -> list:
    """Read the label index → wnid mapping baked into the dataset's info files."""
    from datasets import load_dataset
    ds = load_dataset(REPO_ID, split="validation", streaming=True)
    return ds.features["label"].names


def convert_parquets(snapshot_dir: Path, out_root: Path, split_glob: str, split_dir: str,
                    id2wnid: list):
    """Iterate every parquet file matching split_glob and write PIL-decoded JPEGs
    into out_root/split_dir/<wnid>/. Uses pyarrow row-by-row to keep RAM bounded."""
    parquet_files = sorted((snapshot_dir / "data").glob(split_glob))
    if not parquet_files:
        raise RuntimeError(f"no parquets matched {split_glob} under {snapshot_dir}/data")

    target = out_root / split_dir
    target.mkdir(parents=True, exist_ok=True)
    # Pre-create wnid dirs so mkdir isn't called ~1.28M times.
    for wnid in id2wnid:
        (target / wnid).mkdir(exist_ok=True)

    running_idx = 0
    for pq_path in tqdm(parquet_files, desc=f"{split_dir}: parquets", unit="file"):
        table = pq.read_table(pq_path)
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        for img_obj, label in tqdm(zip(images, labels), total=len(labels),
                                    desc=pq_path.name, leave=False):
            # HF encodes images as {"bytes": b"...", "path": "..."} dicts.
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
                    help="do NOT delete the HF hub cache after conversion")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    if get_token() is None:
        print("[hf-to-imagefolder] no HF token found. Run `hf auth login` first "
              "(and accept ILSVRC/imagenet-1k terms on the dataset page).", file=sys.stderr)
        sys.exit(1)

    print(f"[hf-to-imagefolder] snapshot_download only train + val parquets ...", flush=True)
    snapshot_dir = Path(snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=ALLOW_PATTERNS,
        cache_dir=args.cache_dir,
        max_workers=8,
    ))
    print(f"[hf-to-imagefolder] snapshot at {snapshot_dir}")

    print(f"[hf-to-imagefolder] loading wnid mapping ...")
    id2wnid = get_wnid_mapping(snapshot_dir)
    assert len(id2wnid) == 1000, f"expected 1000 wnids, got {len(id2wnid)}"

    convert_parquets(snapshot_dir, out_root, "train-*.parquet", "train", id2wnid)
    convert_parquets(snapshot_dir, out_root, "val-*.parquet", "val", id2wnid)

    verify(out_root)

    if not args.keep_cache:
        # Delete the on-disk snapshot dir. HF cache is at
        # $HF_HOME/hub/datasets--ILSVRC--imagenet-1k, cache_dir if overridden.
        # We only unlink the snapshot pointing at these parquets; blobs may remain
        # in the shared blob store (small).
        print(f"[hf-to-imagefolder] cleaning snapshot at {snapshot_dir.parent} ...")
        shutil.rmtree(snapshot_dir.parent.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
