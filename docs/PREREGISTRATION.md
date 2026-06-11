# Pre-Registration: Maser Candidate Selection, Confirmatory Analyses

**Status: DRAFT (not yet in force).** This document binds nothing until the
team reviews it, fills in the sign-off block, and commits the final version.
After sign-off, the analyses in Section 4 are *confirmatory*: they are run
exactly as written, once, and reported regardless of outcome. Everything else
is exploratory and will be labeled as such in any write-up.

| | |
|---|---|
| Drafted | 2026-06-11 |
| Signed off | _(date, after team review)_ |
| Team | _(names)_ |
| Amendments | none _(see Section 8)_ |
| Checksums verified | _(confirm Section 1 checksums match files at sign-off; regenerate if data updated since drafting)_ |

---

## 1. Frozen inputs

All analyses load data exclusively through `src/maser_data.py`. That file and
the four raw data files in `data/raw/` are all frozen inputs:
`src/maser_data.py` makes substantive decisions (which duplicate to drop, how
to purge contaminated nonmaser rows, which coordinate to use as the join key)
that are just as binding as the data themselves. Any change to any of these
five files requires an amendment under Section 8.

```
e94771f0854aa49d3ff15640382b33c0  src/maser_data.py
9ac8205e8ed48f4a68b41ab8654e3f88  data/raw/Table_1_data.txt
727aa8ca9e2bb009b48bdf302f9652be  data/raw/table_2_data.txt
68a2fb3383d64ba3c4e1979e48257c78  data/raw/allwise_masers_cleaned_6arcs.csv
f7e7e1755235801c4701d53752391567  data/raw/allwise_nonmasers_cleaned_6arcs.csv
```

## 2. Targets and samples

**Track A: X-ray sample** (`src.maser_data.load_xray_sample()`, 641 galaxies):
- **Target**: megamaser-or-disk (`maser` ∈ {2,3,4}) vs. rest, 53 positives.
  Chosen because it matches the megamaser detection rates of Kuo et al. (2020)
  Table 4 and the N_H / L₁₂μm physics; kilomasers (type 1) count as negatives,
  consistent with the paper's accounting.
- **Sample**: all 641 galaxies. Labels as published in Kuo et al. (2020),
  including the 6 galaxies whose labels the AllWISE catalog disputes
  (robustness check R3 re-runs without them).
- **Features**: `L_12um_bestfit_1` and `Lob` only (the paper's Figure-5 plane, 
  the "same information, better boundary?" framing). No distance.

**Track B: combined X-ray + WISE sample**
(`src.maser_data.load_xray_with_wise()`):
- **Target**: megamaser-or-disk (`maser` ∈ {2,3,4}) vs. rest, identical to
  Track A's target.
- **Sample**: the 602 Track-A galaxies with a WISE counterpart within 6″
  (50 of the 53 positives). The 3 unmatched positives are excluded, so both
  arms of comparison B1 see the same galaxies and share folds. This is a
  *distinct* sample from Track A (602 vs. 641, 50 vs. 53 positives), which is
  why it is its own track rather than a third Track-A comparison: it cannot
  share folds with A1/A2.
- **Features**: vary by arm (see B1).

**Track C: AllWISE full-survey sample** (`src.maser_data.load_wise()`, cleaned):
- **Target**: megamaser, defined as `maser_lum` ≥ 10 L_sun, 114 positives.
- **Sample**: all cleaned nonmasers with velocity < 21,000 km/s (z < 0.07) as
  negatives. Maser rows that are *not* clean positives (25 kilomasers, 35 with
  flagged/unknown luminosity) are **excluded from the sample entirely**, as
  they are neither positives nor trustworthy negatives. Rows with missing velocity
  are excluded.
- **Features**: W1−W2 and W2−W3 colors only. W4-based features are excluded
  from confirmatory analyses (upper-limit artifact risk; see
  `docs/RESEARCH_PLAN.md`).

## 3. Models, protocol, and seeds

- **Primary model (all tracks)**: scikit-learn `LogisticRegression`
  (L2, `C = 1.0` fixed, no tuning), `class_weight = None`, inside a
  `Pipeline(median impute → StandardScaler → LR)`. No resampling of any kind.
- **Challenger model (Tracks A and C)**: `GradientBoostingClassifier` with
  `max_depth = 2`, `n_estimators = 100`, `learning_rate = 0.1`, all fixed.
- **Validation**: 20× repeated stratified 5-fold CV. Repeat *r* uses
  `StratifiedKFold(n_splits=5, shuffle=True, random_state=r)` for
  r = 0…19. Identical splits are reused for every model and baseline
  (paired comparisons).
- Held-out predicted probabilities are pooled within each repeat to form one
  PR curve per repeat; figures show the median curve and 5th–95th percentile
  band across the 20 repeats, captioned as split-to-split variability (not a
  confidence interval).
- The confirmatory observing budget is top 50 candidates for every track. In
  5-fold CV, model-vs-model and feature-set comparisons therefore use per-fold
  precision@10, selecting 10 held-out galaxies from each of 5 folds for a
  50-candidate full-repeat equivalent.

## 4. Confirmatory comparisons (run once, reported regardless of outcome)

**A1: Fitted boundary vs. the paper's best hand-drawn cut (the headline).**
In each held-out fold: re-score the paper's best combined cut
(inferred N_H ≥ 10^24 cm^-2 AND `L_12um_bestfit_1` > 42, with inferred N_H
computed from the paper's L_2-10^obs / L_12um^AGN Figure-5 method, not from
the tabulated spectral-fit `logNH`) to get its (precision, recall). Before
sign-off, the analysis script must name the exact helper that implements this
published mapping and must reproduce the relevant Table-4 in-sample counts as
a sanity check. To compare to the logistic model, use the highest model
precision among held-out thresholds with recall >= the cut's held-out recall;
if the cut's held-out recall is 0, use precision@1 for the model and define the
cut precision as 0 when it selects no galaxies. Analyze the 100 paired per-fold
precision differences (model - cut).

**A2: Does the challenger beat the simple model?**
Per held-out fold: precision@10 (top 10 of ~128 test galaxies ≈ a top-50
campaign at full-sample scale) for GBT vs. LR on identical folds. Analyze the
100 paired per-fold differences (GBT − LR).

**B1: Does adding WISE information to the X-ray features help?**
On the Track-B sample (602 galaxies, 50 positives), two logistic models on
identical folds: features {`L_12um_bestfit_1`, `Lob`} vs.
{`L_12um_bestfit_1`, `Lob`, W1−W2, W2−W3}. Per-fold precision@10, 100 paired
differences (expanded − baseline). This answers "does *more information*
help?", deliberately kept separate from A1's "does a *better boundary* help?"
and A2's "does a *fancier model* help?". Note the folds here are Track B's own
(50 positives), not Track A's, so B1's numbers are not directly comparable to
A1/A2's; the comparison that matters is internal (expanded vs. baseline).

**C1: Fitted WISE boundary vs. the standard IR color cut.**
As A1, on Track C: the Stern cut (W1−W2 ≥ 0.8) re-scored per held-out fold;
logistic model's precision at the Stern cut's per-fold recall using the same
"highest precision at recall >= cut recall" rule; 100 paired differences
(model - cut).

**C2: Does the challenger beat the simple model where the data could tell?**
As A2, on Track C: per held-out fold, precision@10 for GBT vs. LR on identical
folds; 100 paired per-fold differences (GBT − LR). This is the one
model-vs-model comparison with the most statistical power (114 positives), and
the shape question is live here too: the standard hand-drawn boundaries in WISE
color space are wedges (intersections of thresholds), which depth-2 trees
represent natively and a linear model cannot.

**Inference, all five**: Nadeau–Bengio corrected resampled t-test on the
paired per-fold differences (test fraction 1/5), reported with a 90% CI and
the effect size in science units (extra true masers per 50-pointing campaign).
A bootstrap CI of the per-fold differences is reported alongside as the
intuitive cross-check.

**Decision rules (asymmetric, pre-committed):**
- A2 / C2: adopt GBT over LR, per track, only if the 90% CI excludes zero
  **and** the point estimate is ≥ +2 masers in a top-50 campaign. Otherwise
  LR stays primary. "Can't tell" means insufficient
  data, and parsimony wins ties. The tracks decide independently: C2 finding
  a real GBT advantage does not promote GBT on Track A.
- B1: same rule. Adopt the expanded feature set only if the 90% CI excludes
  zero and the gain is ≥ +2 masers in a top-50 campaign; otherwise the
  two-feature model stays primary.
- A1/C1: no adoption decision. The result (fitted beats / ties / loses to
  hand-drawn) is reported as the finding. A tie is a publishable result: it
  says the hand-drawn boundary shape is essentially right.

**Honest power statement**: with 53 positives on Track A, the expected A2
outcome is a CI straddling zero. We pre-commit to reporting that as "could not
distinguish; adopting the simpler model," not as a failure and not as proof of
equivalence. C2 (114 positives) is the one model-vs-model comparison where a
real difference is plausibly detectable; its outcome is genuinely open.

## 5. Pre-specified robustness checks (reported, not decision-driving)

- R1: Track A primary re-run with D < 170 Mpc cut.
- R2: Track C primary re-run on the "well-searched" subsample (per-galaxy
  detectable-luminosity limit from `rms1_01`, velocity range, and distance;
  threshold = the median megamaser luminosity).
- R3: Track A primary re-run excluding the 6 label-disagreement galaxies.
- R4: LR with `C` ∈ {0.1, 10} (sensitivity of the one fixed hyperparameter).
- R5: Track A re-run per X-ray catalog (Swift-only vs. XMM-only folds scored
  separately).

## 6. Declared exploratory (no confirmatory status, labeled as such)

Interaction-term LR (the "box"), KNN (poster continuity), disk-maser target
(type 4 vs. rest, excluding the 8 type-3 candidates), staged P(maser)→P(disk)
models, added features beyond B1's (Γ, [OIII], distance), W1−W4
color, calibration analysis, Mateos-wedge baseline, and anything not listed in
Section 4.

## 8. Amendment policy

Changes after sign-off are allowed only *before* the affected analysis is run,
must be recorded in the table at the top with date and reason, and must leave
the original text visible (strike through, don't delete). If a confirmatory
analysis must change after being run, the original result is reported too.
