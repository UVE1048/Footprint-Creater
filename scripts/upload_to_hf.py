#!/usr/bin/env python3
"""Publish the pretrained checkpoints to the Hugging Face Hub.

Hugging Face is where most people (and most AI assistants) look for models, so
the weights live there in addition to GitHub Releases. The model card in
``hf/hrnet-w48-footprint/README.md`` is uploaded as the repo README.

Setup:

    pip install huggingface_hub
    export HF_TOKEN=hf_xxxxxxxx        # https://huggingface.co/settings/tokens

Usage:

    python scripts/upload_to_hf.py                     # publish W48
    python scripts/upload_to_hf.py --model w32         # publish W32 instead
    python scripts/upload_to_hf.py --dry-run           # show what would be sent
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "w48": dict(
        repo_id="UVE1048/hrnet-w48-footprint",
        local="hrnet_w48_iter_6000.pth",
        remote="hrnet_w48_iter_6000.pth",
        card="hf/hrnet-w48-footprint/README.md",
        url="https://github.com/UVE1048/Footprint-Creater/releases/download/models-v1/hrnet_w48_iter_6000.pth",
    ),
    "w32": dict(
        repo_id="UVE1048/hrnet-w32-footprint",
        local="iter_40000.pth",
        remote="iter_40000.pth",
        card="hf/hrnet-w32-footprint/README.md",
        url="https://github.com/UVE1048/Footprint-Creater/releases/download/models-v1/iter_40000.pth",
    ),
}


def ensure_weights(target: Path, url: str) -> None:
    """Download from the GitHub release if the checkpoint is not present."""
    if target.exists():
        print(f"  using local weights: {target} ({target.stat().st_size / 1e6:.0f} MB)")
        return
    print(f"  local weights missing, downloading from GitHub Releases ...")
    tmp = target.with_suffix(target.suffix + ".part")

    def _progress(blocks: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(blocks * block_size, total)
        sys.stdout.write(f"\r  {done / 1e6:7.1f}/{total / 1e6:.1f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    print()
    tmp.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="w48")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = MODELS[args.model]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if not token and not args.dry_run:
        print("Missing HF_TOKEN. Create one at https://huggingface.co/settings/tokens")
        print("with 'write' permission, then: export HF_TOKEN=hf_xxx")
        return 1

    weights = ROOT / "model" / spec["local"]
    card = ROOT / spec["card"]

    print(f"Model     : {spec['repo_id']}")
    print(f"Card      : {card.relative_to(ROOT)}")
    print(f"Weights   : {weights.relative_to(ROOT)}")
    print(f"Dry run   : {args.dry_run}")

    if not card.exists():
        print(f"  x model card not found: {card}")
        return 1

    if args.dry_run:
        # dry run 不下载 542 MB 权重，只报告状态
        if weights.exists():
            print(f"  weights present: {weights.stat().st_size / 1e6:.0f} MB")
        else:
            print("  weights not present yet; will be downloaded on a real run")
        print("\nDry run only, nothing uploaded.")
        return 0

    ensure_weights(weights, spec["url"])

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=spec["repo_id"], repo_type="model", exist_ok=True)
    print("  + repo ready")

    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=spec["repo_id"],
        commit_message="Add model card",
    )
    print("  + model card uploaded")

    api.upload_file(
        path_or_fileobj=str(weights),
        path_in_repo=spec["remote"],
        repo_id=spec["repo_id"],
        commit_message=f"Add {spec['remote']}",
    )
    print("  + weights uploaded")

    print(f"\nDone: https://huggingface.co/{spec['repo_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
