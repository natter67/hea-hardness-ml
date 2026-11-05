"""Download the MPEA dataset (Borg et al., Sci. Data 7, 430 (2020)).

Usage:
    python scripts/download_data.py

WHY a script instead of "just download it once"?
    Reproducibility. Anyone (including future you) can regenerate
    data/MPEA_dataset.csv and know *exactly* which version they got.

WHY pin a commit SHA instead of using the "master" branch URL?
    A branch is a moving target: if the authors push a correction, the file
    behind ".../master/MPEA_dataset.csv" silently changes and your numbers
    shift with no code change. A commit SHA is immutable, so the URL below
    will always return the same bytes. We additionally verify a SHA-256
    checksum so a truncated/corrupted download fails loudly instead of
    quietly producing a slightly smaller dataset.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

# Commit on CitrineInformatics/MPEA_dataset "master" at the time this project
# was scaffolded. Bump both constants together if you deliberately upgrade.
COMMIT = "d31e2510cccc3fb9e79d43ad62652f64ef8c2c5e"
URL = f"https://raw.githubusercontent.com/CitrineInformatics/MPEA_dataset/{COMMIT}/MPEA_dataset.csv"
SHA256 = "c5504d93fd324d1be26cf814d0694ee9ee95578d68a0b38aba6813b939fa2c5d"

DEST = Path(__file__).resolve().parents[1] / "data" / "MPEA_dataset.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if DEST.exists() and sha256(DEST) == SHA256:
        print(f"{DEST} already present and checksum matches; nothing to do.")
        return 0

    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    tmp = DEST.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)

    got = sha256(tmp)
    if got != SHA256:
        tmp.unlink()
        print(f"Checksum mismatch: expected {SHA256}, got {got}", file=sys.stderr)
        return 1

    tmp.replace(DEST)  # atomic rename: never leaves a half-written CSV behind
    print(f"Saved to {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
