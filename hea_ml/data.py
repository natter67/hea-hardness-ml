"""Loading the MPEA dataset and turning formulas into a canonical composition key.

Everything that more than one place needs (the EDA notebook *and* the CV demo)
lives here, so the two can never silently disagree about, e.g., which rows
count as "has a hardness value" or how compositions are normalized.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "MPEA_dataset.csv"

# The raw CSV uses long, LaTeX-flavoured column names such as
# "PROPERTY: Test temperature ($^\circ$C)". Renaming once, up front, keeps the
# rest of the code readable. Only the columns we actually use are renamed;
# the rest keep their original names.
COLUMN_RENAMES = {
    "IDENTIFIER: Reference ID": "ref_id",
    "FORMULA": "formula",
    "PROPERTY: Microstructure": "microstructure",
    "PROPERTY: Processing method": "processing",
    "PROPERTY: BCC/FCC/other": "phase_class",
    "PROPERTY: HV": "HV",
    "PROPERTY: Type of test": "test_type",
    "PROPERTY: Test temperature ($^\\circ$C)": "test_temp_C",
    "REFERENCE: doi": "doi",
    "REFERENCE: year": "year",
}

# Default rounding for the composition key (see normalize_formula). 3 decimals
# of *atomic fraction* = 0.1 at.% resolution, well below the accuracy with
# which as-cast alloy compositions are actually controlled, but coarse enough
# that "Co0.667" and "Co2/3" written by two different papers collapse together.
DEFAULT_DECIMALS = 3


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV exactly as published, with friendlier column names."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path).rename(columns=COLUMN_RENAMES)


@lru_cache(maxsize=None)  # same formula strings repeat a lot; parse each once
def normalize_formula(formula: str, decimals: int = DEFAULT_DECIMALS) -> str:
    """Return a canonical string for a composition, independent of how it was written.

    WHY this matters: the dataset records compositions as molar *ratios*, and
    different papers write the same alloy differently, e.g.

        "Al0.5 Co1 Cr1 Fe1 Ni1"   and   "Al1 Co2 Cr2 Fe2 Ni2"

    are the same material. If we grouped on the raw string, those two rows
    would land in different CV groups and a "grouped" split would still leak.

    Steps:
      1. Parse with pymatgen (handles element symbols + numeric amounts).
      2. Convert to atomic *fractions* (sum to 1) -> removes the scale ambiguity.
      3. Round each fraction -> removes tiny float/typing differences
         (0.667 vs 0.6667 vs 2/3).
      4. Sort elements alphabetically and join -> removes ordering ambiguity.
    """
    frac = Composition(formula).fractional_composition
    parts = sorted((el.symbol, round(amt, decimals)) for el, amt in frac.items())
    return " ".join(f"{sym}{amt:.{decimals}f}" for sym, amt in parts)


def load_hardness(
    path: Path = DATA_PATH,
    room_temp_only: bool = True,
    decimals: int = DEFAULT_DECIMALS,
) -> pd.DataFrame:
    """Rows that report Vickers hardness, plus a `comp_key` grouping column.

    room_temp_only:
        Hardness drops with temperature, so an 800 °C measurement of an alloy
        is not the same target as its 25 °C hardness. Mixing them would add
        label noise that composition features cannot explain (the model sees
        identical inputs with very different outputs). We keep rows tested at
        25 °C *and* rows with no temperature recorded, on the assumption that
        an unstated temperature means room temperature (Vickers testing is
        overwhelmingly done at RT). That is an assumption, not a fact — the
        EDA notebook looks at it. Set False to keep everything.
    """
    df = load_raw(path)
    df = df[df["HV"].notna()].copy()
    if room_temp_only:
        df = df[df["test_temp_C"].isna() | (df["test_temp_C"] == 25)]
    df["comp_key"] = [normalize_formula(f, decimals) for f in df["formula"]]
    return df.reset_index(drop=True)
