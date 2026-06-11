# Maser Candidate Selection

Machine-learning workflow for prioritizing H2O megamaser candidates from
X-ray, mid-infrared, and WISE photometry.

Start with:

- `docs/RESEARCH_PLAN.md` for the project overview and scientific rationale.
- `docs/PREREGISTRATION.md` for the confirmatory analysis draft.
- `src/maser_data.py` for the shared data-loading and cleaning code.

All analyses should load data through `src/maser_data.py`, not directly from
the raw tables in `data/raw/`. The loader owns the duplicate handling, WISE
nonmaser contamination purge, per-galaxy WISE deduplication, and X-ray/WISE
coordinate match.

## Layout

```text
data/raw/                 Frozen input tables
docs/                     Research plan, preregistration, data notes
src/                      Shared Python code
README.md                 Repo overview and smoke test
requirements.txt          Python dependencies
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Smoke Test

```bash
python -m src.maser_data
```

Expected summary:

```text
X-ray sample: 641 galaxies, masers=68 (mega+=53, disk=24)
WISE sample: 4450 galaxies, masers=174, megamasers (lum>=10): 114, lum unknown: 35
X-ray+WISE: 602/641 matched within 6 arcsec (62/68 masers)
WISE rows assigned to two X-ray rows: 0

Label disagreements (6) -- X-ray maser type vs WISE label:
```

The six listed disagreements should be M31, IC 750, NGC 4261, Arp 220,
IRAS 15480-0344, and IGR J16385-2057.

## Data Notes

The raw data tables in `data/raw/` are the frozen inputs listed in
`docs/PREREGISTRATION.md`.
