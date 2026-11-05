# hea-hardness-ml

Predicting the **Vickers hardness (HV)** of high-entropy alloys (HEAs) from **composition alone**,
using the MPEA dataset of Borg et al., *Sci. Data* 7, 430 (2020).

This repo is a learning project, so the code is commented heavily and explains *why* as well
as *what*.

## Status: Phases 1–3 (scaffold)

| Phase | What | Where |
|---|---|---|
| 1. Data | Pinned, checksummed download of `MPEA_dataset.csv` | `scripts/download_data.py`, `data/` |
| 2. EDA | Rows vs distinct compositions, HV distribution and outliers, element frequency, duplicate-composition leakage, phase reporting | `notebooks/01_eda.ipynb` |
| 3. Features + honest CV | Two feature blocks (Set A: Magpie; Set B: δ, ΔS_mix, ΔH_mix, VEC), and a demo that scores one model with random `KFold` vs `GroupKFold` | `hea_ml/features.py`, `scripts/cv_split_demo.py` |

Deliberately **not** done yet: hyper-parameter tuning, feature selection, SHAP, write-up.

## Layout

```
data/MPEA_dataset.csv        raw data (unmodified, Apache-2.0, see data/README.md)
hea_ml/data.py               load CSV, filter to HV rows, normalized composition key
hea_ml/features.py           Set A (matminer Magpie) and Set B (HEA physics parameters)
notebooks/01_eda.ipynb       EDA with written answers (generated, executed)
scripts/download_data.py     reproducible download
scripts/build_eda_notebook.py  source of the EDA notebook; rebuilds + executes it
scripts/cv_split_demo.py     random KFold vs GroupKFold on the same model
```

## Setup

Python 3.11+ (developed on 3.13). Use a virtual environment: on some Linux distros the system
`setuptools` can't build one of pymatgen's dependencies (`bibtexparser`), and a fresh venv
avoids that.

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -U pip setuptools wheel && pip install -r requirements.txt
```

```bash
python scripts/download_data.py
```

## Run

```bash
python scripts/cv_split_demo.py
```

Options: `--features A|B|AB` (default `AB`), `--decimals N` (rounding of atomic fractions used to
define "same composition", default 3).

To regenerate the EDA notebook after editing `scripts/build_eda_notebook.py`:

```bash
python scripts/build_eda_notebook.py
```

## First results (5-fold CV, seed 42, unmodified `GradientBoostingRegressor`)

501 rows (HV at 25 °C or unstated), 376 distinct compositions.

| Features | Split | R² | MAE (HV) | RMSE (HV) |
|---|---|---|---|---|
| A+B (136) | random KFold | 0.786 ± 0.062 | 67.5 ± 4.0 | 96.8 ± 11.3 |
| A+B (136) | GroupKFold by composition | 0.720 ± 0.072 | 77.2 ± 4.4 | 110.1 ± 11.9 |
| B only (4) | random KFold | 0.696 ± 0.073 | 84.6 ± 7.7 | 115.9 ± 10.8 |
| B only (4) | GroupKFold by composition | 0.583 ± 0.093 | 100.6 ± 8.5 | 134.8 ± 14.8 |

Random splitting overstates performance on unseen alloys by ~10–16 HV MAE, because ~35 % of
random-split test rows have an identical composition in the training set (EDA, Q4).
The **GroupKFold** numbers are the honest ones. These come from a single seed; differences of a
few HV between setups are within noise.

## Key choices (and why)

* **Composition key** = atomic fractions rounded to 3 decimals, elements sorted. Makes
  `Al0.5 Co1 …` and `Al1 Co2 …` the same alloy. Near-duplicates like `Al0.3` vs `Al0.304` still
  form separate groups; `--decimals 2` is a stricter alternative.
* **Room-temperature only**: 29 elevated-temperature HV rows are dropped (different target).
  Rows with no temperature recorded are assumed to be room temperature.
* **Tabulated data, not hand-typed**: ΔH_ij from Takeuchi & Inoue (2005) and radii from Miracle
  et al. (2010), both as shipped with matminer; VEC per element from matminer's Miedema table
  (s+d electron convention). See the comments in `hea_ml/features.py`.
