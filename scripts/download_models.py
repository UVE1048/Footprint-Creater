#!/usr/bin/env python3
"""Download pretrained weights for Footprint-Creater from GitHub Releases.

Weights are ~773 MB in total, which exceeds what is sensible to keep in git
(GitHub's free LFS quota is 1 GB of storage *and* bandwidth), so they are
published as release assets instead.

Usage:
    python scripts/download_models.py            # download all available weights
    python scripts/download_models.py --model w32
    python scripts/download_models.py --check    # verify local files only
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "UVE1048/Footprint-Creater"
TAG = "models-v1"
BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}"

# model key -> (asset filename, sha256, description)
MODELS = {
    "w32": (
        "iter_40000.pth",
        "14eb942156ff3e520362329da855daeb8d15bf887a242530b52f29de9e67ea0c",
        "HRNet-W32 + FCNHead, 40000 iters (~231 MB) - default, faster",
    ),
    "w48": (
        "hrnet_w48_iter_6000.pth",
        "73db3f97998fc62cbbcb83897925d1c6ffaef2e1dadd78b5b7f90267dab5cf18",
        "HRNet-W48 + FCNHead, 6000 iters (~542 MB) - higher quality, slower",
    ),
}

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, expected: str) -> bool:
    if not path.exists():
        return False
    actual = sha256_of(path)
    if actual != expected:
        print(f"  ! checksum mismatch for {path.name}")
        print(f"    expected {expected}")
        print(f"    actual   {actual}")
        return False
    return True


def download(key: str, force: bool = False) -> bool:
    filename, expected, desc = MODELS[key]
    target = MODEL_DIR / filename
    print(f"\n[{key}] {desc}")

    if not force and verify(target, expected):
        print(f"  = already present and verified: {target}")
        return True

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{filename}"
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"  -> {url}")

    def _progress(blocks: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(blocks * block_size, total)
        pct = done * 100 / total
        bar = "#" * int(pct // 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%  {done / 1e6:7.1f}/{total / 1e6:.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    except Exception as exc:  # noqa: BLE001 - surface any network/IO failure to the user
        print(f"\n  x download failed: {exc}")
        print("    The release asset may not be published yet - see model/README.md")
        tmp.unlink(missing_ok=True)
        return False

    print()
    if not verify(tmp, expected):
        tmp.unlink(missing_ok=True)
        print("  x checksum verification failed; file discarded")
        return False

    tmp.replace(target)
    print(f"  + saved to {target}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        action="append",
        help="download only the given model(s); repeatable. Default: all",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--check", action="store_true", help="only verify existing local files")
    args = parser.parse_args()

    keys = args.model or sorted(MODELS)

    if args.check:
        ok = True
        for key in keys:
            filename, expected, _ = MODELS[key]
            target = MODEL_DIR / filename
            good = verify(target, expected)
            print(f"{'OK  ' if good else 'MISS'} {filename}")
            ok = ok and good
        return 0 if ok else 1

    results = [download(key, force=args.force) for key in keys]
    print()
    if all(results):
        print("All requested weights are ready.")
        return 0
    print("Some weights could not be downloaded.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
