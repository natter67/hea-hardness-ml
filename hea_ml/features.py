"""Composition -> feature vector, in two deliberately separate blocks.

Set A  "magpie"   : a big, generic, data-driven descriptor set (~132 columns)
Set B  "physics"  : four classic, hand-picked HEA design parameters

WHY keep them separate?
    They answer different questions. Set A asks "how well can a flexible model
    do with lots of generic elemental statistics?" Set B asks "how much do the
    four parameters metallurgists actually use to design HEAs explain?"
    Later you will want to compare A vs B vs A+B, and that comparison is only
    clean if the blocks are built independently and concatenated explicitly.

Every function here takes a list/Series of formula strings and returns a
DataFrame with one row per formula, in the same order, so the result can be
lined up with the target column by position.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from matminer.utils.data import MagpieData, MixingEnthalpy
from pymatgen.core import Composition, Element

# Gas constant, J/(mol·K). ΔS_mix is reported in these units (some papers
# report it in units of R instead; divide by R to convert).
R = 8.314462618

# ---------------------------------------------------------------------------
# Set A — matminer ElementProperty "magpie" preset
# ---------------------------------------------------------------------------
#
# What it does: for each of 22 elemental properties (melting point, atomic
# radius, electronegativity, number of d electrons, ...) it looks up the value
# for every element in the alloy and summarises them with 6 statistics,
# weighted by atomic fraction:
#     minimum, maximum, range, mean, avg_dev (mean absolute deviation), mode
# -> 22 x 6 = 132 features.
#
# Reference: Ward et al., npj Comput. Mater. 2, 16028 (2016).
#
# Why it's a good baseline: it is the de-facto standard composition
# featurisation in materials ML, so results are comparable with the
# literature, and it needs zero domain-specific choices from us.
#
# Why it's not the whole story: it was designed for *all* inorganic
# compounds, so many columns (e.g. GSbandgap for metals ~ 0) carry little
# signal for HEAs, and none of its features encode *pairwise* chemistry
# (how much two specific elements like or dislike each other). That's what
# Set B's ΔH_mix adds.


def magpie_features(formulas: pd.Series | list[str]) -> pd.DataFrame:
    """Set A: 132 Magpie statistics per composition."""
    featurizer = ElementProperty.from_preset("magpie")
    comps = [Composition(f) for f in formulas]
    # featurize_many parallelises with multiprocessing by default; n_jobs=1
    # keeps it deterministic and avoids fork overhead for ~500 rows.
    featurizer.set_n_jobs(1)
    rows = featurizer.featurize_many(comps, pbar=False)
    cols = [f"A_{c}" for c in featurizer.feature_labels()]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Set B — four classic HEA parameters
# ---------------------------------------------------------------------------
#
# All four are composition-weighted averages or spreads; c_i is the atomic
# fraction of element i (sum c_i = 1).
#
#   δ (delta)  atomic size mismatch, %         Zhang et al., Adv. Eng. Mater. 10, 534 (2008)
#              δ = 100 * sqrt( Σ c_i (1 - r_i / r̄)^2 ),   r̄ = Σ c_i r_i
#              WHY for hardness: big atoms next to small ones distort the
#              lattice; dislocations must push through that rough strain
#              landscape -> solid-solution strengthening. Large δ (> ~6.6 %)
#              also tends to push alloys out of simple solid solutions into
#              intermetallics, which are hard and brittle.
#
#   ΔS_mix     ideal configurational entropy, J/(mol·K)
#              ΔS_mix = -R Σ c_i ln c_i
#              WHY: the "high entropy" in HEA. It is maximised by equimolar
#              compositions and grows with the number of elements. Its link
#              to hardness is indirect (it stabilises solid solutions), so
#              don't be surprised if it ends up a weak predictor.
#
#   ΔH_mix     mixing enthalpy, kJ/mol (regular-solution / Miedema model)
#              ΔH_mix = Σ_{i<j} 4 ΔH_ij c_i c_j
#              WHY: strongly negative values mean the elements bond strongly
#              to each other -> ordering / intermetallic formation (often
#              harder). Positive values -> tendency to phase-separate.
#              The factor 4: ΔH_ij is tabulated as the enthalpy of an
#              *equimolar* binary A50B50 liquid; in the regular-solution
#              model ΔH = Ω c_A c_B, and at c_A = c_B = 0.5 that is Ω/4,
#              so Ω_ij = 4 ΔH_ij.
#              Source of ΔH_ij: Takeuchi & Inoue, Mater. Trans. 46, 2817
#              (2005), shipped with matminer as MixingEnthalpy. We use the
#              published table rather than typing values in by hand —
#              hand-typed tables are a classic source of silent sign/units
#              errors.
#
#   VEC        valence electron concentration (no units)
#              VEC = Σ c_i VEC_i
#              WHY: Guo et al., J. Appl. Phys. 109, 103505 (2011) found
#              VEC >= 8 -> FCC, VEC < 6.87 -> BCC. FCC HEAs are typically
#              soft (~150–250 HV), BCC ones much harder, so VEC is a strong
#              proxy for crystal structure, which drives hardness.
#
# Data sources for the per-element numbers (all tabulated, none typed here):
#   r_i     MagpieData "MiracleRadius": metallic radii of Miracle et al.,
#           Int. Mater. Rev. 55, 218 (2010). Same radii matminer's own
#           YangSolidSolution/WenAlloys featurizers use for δ.
#   VEC_i   "valence_electrons" column of matminer's Miedema.csv (the table
#           WenAlloys uses). It follows the HEA convention (s + d electrons:
#           Cu 11, Ni 10, Hf 4, W 6, Al 3).
#           NOTE: do *not* use Magpie's "NValence" here — it also counts
#           filled f electrons, giving Hf = 18, Ta = 19, W = 20, which would
#           wildly overstate VEC for refractory HEAs.

_MAGPIE = MagpieData()
_MIX = MixingEnthalpy()
_MIEDEMA_TABLE = (
    pd.read_csv(
        Path(__import__("matminer").__file__).parent / "utils" / "data_files" / "Miedema.csv"
    )
    .set_index("element")["valence_electrons"]
)


def _fractions(formula: str) -> tuple[list[str], np.ndarray]:
    """Element symbols and their atomic fractions (summing to 1)."""
    frac = Composition(formula).fractional_composition
    symbols = [el.symbol for el in frac]
    c = np.array([frac[el] for el in frac], dtype=float)
    return symbols, c


def delta_size_mismatch(symbols: list[str], c: np.ndarray) -> float:
    """δ in percent."""
    r = np.array([_MAGPIE.get_elemental_property(Element(s), "MiracleRadius") for s in symbols])
    r_bar = np.sum(c * r)
    return 100.0 * np.sqrt(np.sum(c * (1.0 - r / r_bar) ** 2))


def mixing_entropy(c: np.ndarray) -> float:
    """ΔS_mix in J/(mol·K). c > 0 always here, so log is safe."""
    return -R * np.sum(c * np.log(c))


def mixing_enthalpy(symbols: list[str], c: np.ndarray) -> float:
    """ΔH_mix in kJ/mol, summed over every unordered pair of elements."""
    total = 0.0
    for (i, a), (j, b) in combinations(enumerate(symbols), 2):
        dH_ab = _MIX.get_mixing_enthalpy(Element(a), Element(b))  # kJ/mol, A50B50 liquid
        if np.isnan(dH_ab):
            # Fail loudly rather than quietly treating a missing pair as 0
            # (which would look like "ideal mixing" and bias the feature).
            raise KeyError(f"No tabulated mixing enthalpy for {a}-{b}")
        total += 4.0 * dH_ab * c[i] * c[j]
    return total


def valence_electron_concentration(symbols: list[str], c: np.ndarray) -> float:
    vec_i = _MIEDEMA_TABLE.loc[symbols].to_numpy(dtype=float)
    return float(np.sum(c * vec_i))


def physics_features(formulas: pd.Series | list[str]) -> pd.DataFrame:
    """Set B: δ, ΔS_mix, ΔH_mix, VEC per composition."""
    rows = []
    for f in formulas:
        symbols, c = _fractions(f)
        rows.append(
            {
                "B_delta_pct": delta_size_mismatch(symbols, c),
                "B_dS_mix_J_molK": mixing_entropy(c),
                "B_dH_mix_kJ_mol": mixing_enthalpy(symbols, c),
                "B_VEC": valence_electron_concentration(symbols, c),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience: build either block or both, with optional on-disk caching
# ---------------------------------------------------------------------------

FEATURE_SETS = {"A": magpie_features, "B": physics_features}


def build_features(formulas: pd.Series | list[str], which: str = "AB") -> pd.DataFrame:
    """Concatenate the requested blocks side-by-side ("A", "B" or "AB").

    The column prefixes (A_ / B_) make it trivial to slice the blocks apart
    again later, e.g. X.filter(regex="^B_").
    """
    blocks = [FEATURE_SETS[k](list(formulas)) for k in which]
    return pd.concat(blocks, axis=1)


if __name__ == "__main__":
    # Quick sanity check against a well-known alloy, Cantor's CoCrFeMnNi.
    # Expected: ΔS_mix = R ln 5 ≈ 13.38 J/(mol·K), ΔH_mix = -4.16 kJ/mol
    # (the value quoted throughout the HEA literature from the same Takeuchi
    # table), VEC = 8.0. δ comes out ≈ 2.2 % with Miracle radii; papers using
    # other radius tables quote 3–4 %. That spread is a useful reminder that δ
    # is only meaningful *relative to the radius table used*, so never mix δ
    # values from different sources.
    print(physics_features(["Co1 Cr1 Fe1 Mn1 Ni1"]).T)
