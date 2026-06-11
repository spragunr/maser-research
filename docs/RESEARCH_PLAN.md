# Hunting H₂O Megamasers with Machine Learning
### Team research guide

**Paper we build on:** Kuo et al. (2020), ApJ, 892:18, *"A More Efficient Search for H₂O Megamaser Galaxies"*

---

## 1. The science, in one page

**H₂O megamasers** are naturally occurring microwave lasers: 22 GHz emission from
water molecules orbiting a supermassive black hole, less than a parsec out. The
rarest and most valuable kind, **disk masers**, trace clean Keplerian orbits in the
accretion disk. They give us the most precise black-hole masses we have, and an
independent measurement of the Hubble constant. That's why people point the Green
Bank Telescope (GBT) at thousands of galaxies hoping to find more.

The problem: they're *rare*. Surveying galaxies blindly, only ~3% show any maser
and under 1% show a disk maser. Telescope time is expensive, so the game is
**candidate selection**: given what we already know about a galaxy from other
wavelengths, how do we decide whether it's worth a GBT pointing?

The physics gives us two strong clues. Masers live in AGN that are
**heavily obscured** (large column density N_H of absorbing gas along our line of
sight) and **intrinsically luminous** (bright in the mid-infrared, where the
obscuring dust re-radiates). Kuo et al. (2020) showed that selecting galaxies by
hard cuts on these two quantities raises the disk-maser hit rate from ~1% to ~9–14%.

**Our idea:** those hard cuts are hand-drawn decision boundaries in a 2-D feature
plane. A cut on N_H is a diagonal line in the (L_12μm, L_X) plane; the paper's best
combined cut is a rectangular corner. Drawing classification boundaries from data
is exactly what statistical machine learning does, except ML fits them to maximize
a principled objective instead of choosing round numbers by eye, and it returns a
*probability* for each galaxy instead of a yes/no. So the research question is:

> **Do fitted decision boundaries select maser candidates better than the paper's
> hand-drawn cuts? If so, by how many extra masers per observing campaign?**

**What the answer looks like in practice.** Imagine we have telescope time to
follow up 50 galaxies. We rank all galaxies by our model's predicted probability
and take the top 50. How many of those 50 turn out to be real masers? That
fraction is *precision@50*, and it is the number we ultimately care about:
higher precision means more real masers found per hour of GBT time. We compare
it to what the paper's best cut gives for the same 50-galaxy budget. The full
picture is a *precision-recall curve*, which shows that tradeoff across every
possible budget size, but precision@50 is the single summary number tied to
a concrete observing scenario.

Note the humble version of this question is still publishable: if a fitted model
only *ties* the paper's cuts, that tells us the paper's simple box really is the
right shape, which is worth knowing.

---

## 2. Precision and Recall vs. Detection Rate and Completeness

The paper reports two numbers for every cut, and they are *exactly* the two
standard ML classification metrics in disguise:

| Paper's term | Definition | ML term |
|---|---|---|
| **Detection rate** R | of galaxies passing the cut, the fraction that are masers | **Precision** |
| **Completeness** C | of all the masers, the fraction that pass the cut | **Recall** |

So each of the paper's cuts is one (recall, precision) point. Our models, which
output a probability per galaxy, each trace an entire **precision–recall (PR) curve**
through the same plane as we slide the probability threshold. The headline plot of
this project is: *our PR curves with the paper's cuts overlaid as points.* If the
curves pass up-and-right of those points, fitting beats hand-drawing.

One metric we will **never** report: accuracy. A model that says "no maser" for
every galaxy is ~92% accurate on our sample and 100% useless.

---

## 3. The data

We have **two datasets**, at different scales. Load both through
`src/maser_data.py`, never the raw files in `data/raw/`. That module owns all
the cleaning decisions described below, so everyone's analysis starts from
identical data.

### Dataset 1: the X-ray sample (642 galaxies, 68 masers)

Galaxies from the GBT survey that also have X-ray and mid-IR measurements
(one appears twice; see quirks). Two whitespace-delimited files, joined on
galaxy name:

- `data/raw/Table_1_data.txt`: mid-IR/optical: redshift, distance,
  **L_12μm** (mid-IR AGN luminosity from SED fitting), [OIII] luminosity, and
  the **`maser` label**.
- `data/raw/table_2_data.txt`: X-ray: observed and intrinsic 2–10 keV luminosity
  (**Lob**, Lint), column density **logNH**, photon index Γ, and which X-ray
  catalog the measurement came from (`ref_xray`).

**The label** (`maser` column):

| value | meaning | count |
|---|---|---|
| 0 | no maser detected | 574 |
| 1 | kilomaser (weak, often star-formation-related) | 15 |
| 2 | megamaser | 21 |
| 3 | disk maser candidate | 8 |
| 4 | confirmed disk maser | 24 |

**Quirks (the mechanical ones are handled by `src/maser_data.py`; the conceptual ones require judgment):**

1. **`maser = 0` means "searched and not detected," not "no maser."** The GBT has
   sensitivity limits; beyond ~170 Mpc, real masers can hide below the noise. Our
   models therefore predict *detection* probability, and some distant "negatives"
   are probably false. The robustness check re-runs key results with a D < 170 Mpc
   cut to confirm the conclusions don't depend on this.
2. **One galaxy is in the data twice** (`src/maser_data.py` drops the duplicate).
   MRK739E appears as two rows (one from Swift, one from XMM), spelled `MRK739E`
   vs `Mrk739E` with slightly different coordinates, so neither name nor coordinate
   matching catches it automatically. The XMM row is dropped; Swift has catalog
   priority. Worth knowing because the same trap can appear in any new data you join.
3. **Missing values come in costumes** (`src/maser_data.py` handles the encoding).
   Values of `---`, `-99.0`, and `-99` all mean "not available." Uncertainty
   columns where `0.0` means "not estimated," not zero uncertainty. [OIII]
   luminosity is missing for 47% of rows.
4. **Not all logNH values are equal** (no automatic fix; requires judgment).
   Swift/BAT measurements are reliable even for the most obscured AGN. Some XMM
   values were *fixed at a floor* (often 20.0) rather than fitted. Which catalog
   a galaxy comes from is itself information: `ref_xray` is available as a
   feature/diagnostic, and N_H can alternatively be inferred from position in the
   (L_12μm, Lob) plane (the paper's Figure 5 method) rather than taken from the
   tabulated spectral-fit values.

### Dataset 2: WISE photometry for (almost) the whole survey

Mid-infrared brightness
in the four WISE bands (W1–W4) for essentially the **entire GBT survey**.
After cleaning: **4,450 galaxies with 174 masers (3.9%)**. This matters because
the whole project is constrained by how few masers we have: this sample has
2.6× more of them. The colors really do separate (masers are redder in both
W1−W2 and W2−W3), but with heavy overlap.

What it supports and what it doesn't:

- **Features**: WISE magnitudes and colors (W1−W2, W2−W3, W1−W4). No X-ray
  information; that stays exclusive to Dataset 1.
- **Targets**: maser subtype is unknown for 122 of the 174, so this sample
  supports "any maser" and "megamaser" (via the listed maser luminosity ≥ 10
  L_sun, 114 galaxies), but **not** the disk-maser target.
- **Additional useful information**: for every nonmaser, the file records the GBT
  observation itself: integration time, system temperature, RMS noise,
  velocity range searched. That means we can compute, galaxy by galaxy, *how
  bright a maser would have needed to be for that observation to see it,* giving us
  a principled upgrade to the blunt D < 170 Mpc cut.

Its quirks (all handled by `src/maser_data.py`, but understand them):

1. **The raw nonmaser file is a list of observations, not galaxies.** 99 rows
   are actually *known masers* under alternate names: calibration and repeat
   pointings (`CALNGC3079`, `NGC5765B-OFF1`…), including NGC 4258 itself. Left
   in, they'd poison the negative class. Another 101 rows are repeat pointings
   of the same nonmaser galaxy under different names (3C84 = NGC1275). The
   reliable per-galaxy key is the *WISE counterpart position*, not the survey
   name or pointing coordinates; the loader purges and dedupes on it, keeping
   each galaxy's deepest observation.
2. **W3 and especially W4 are shallow**: 569 nonmasers have W4 signal-to-noise
   below 3, meaning the detection is less than 3× the noise floor and isn't
   reliable. Those W4 magnitudes are effectively upper limits rather than real
   measurements. The catch is that masers are IR-bright and almost always
   detected, so the low-SNR problem is one-sided: nonmasers look faint, masers
   look bright, and a model can learn "noisy W4 → nonmaser" as a spurious rule
   that reflects telescope sensitivity rather than astrophysics. We avoid this
   by using W1−W2 and W2−W3 as primary features (both bands are reliably
   detected for nearly everything); the loader provides low-SNR flags if you
   want to investigate further.
3. **Six galaxies are masers here but nonmasers in Dataset 1** (M31, IC 750,
   NGC 4261, Arp 220, IRAS 15480−0344, IGR J16385−2057), probably detections
   published after the 2018 survey catalog. The loader flags them
   (`label_disagree`); we exclude them from merged-label analyses until Anca
   confirms.
4. **Different base rate** (3.9% vs 8.3%): precision numbers from this sample
   and Dataset 1 live on different scales; never compare them directly
   (Rule 6).

---

## 4. The plan, in phases

### Phase 1: Reproduce the paper (sanity anchor)
Compute the paper's Table-4 detection rates and completeness values from the data
files ourselves. If we can't reproduce their numbers, we don't understand the data
yet and nothing downstream can be trusted.

### Phase 2: Baseline model and the headline plot
**Penalized (L2) logistic regression** on the same two features the paper's cuts
use: the (L_12μm, Lob) plane. This is deliberately the simplest model that can
represent the paper's boundary family: same information, fitted boundary. A second
variant adds an interaction term so the boundary can bend into the paper's "box"
shape. Evaluate with **20× repeated stratified 5-fold cross-validation**; pool
each repeat's held-out probabilities into one PR curve; plot the median curve with
a 5th–95th percentile band across repeats. Overlay the paper's cuts re-scored on
the same held-out folds, not their published in-sample numbers, which were
computed on the full dataset and would be an unfair comparison (Rule 5).

### Phase 3: Do fancier models help?
Candidates: gradient-boosted trees (shallow, depth 2 *is* the box; this is
confirmatory comparison A2), and KNN (exploratory, for continuity with the
prior student work on this data). The comparison protocol matters more than
the models (see Section 6). Honest expectation: with ~53 positives, the most
likely outcome here is "no detectable improvement over logistic regression,"
and that is a *finding* (the structure really is a line/box), not a failure.
The version of this question with real statistical power is C2, on the WISE
track (Phase 5).

### Phase 4: Track B: does more information help?
This is its own track because it runs on its own sample: the 602 galaxies that
have *both* X-ray and WISE data (50 of the 53 positives), which means it can't
share cross-validation folds with Track A (641 galaxies) or Track C (the full
WISE survey). Confirmatory comparison B1 runs logistic regression on those 602,
comparing the two-feature X-ray model {L_12μm, Lob} against the same model
augmented with WISE colors {+W1−W2, W2−W3}. This keeps "more information?"
cleanly separate from "better boundary?" (Phase 2) and "fancier model?"
(Phase 3). The three questions use the same validation machinery but must never
be conflated in one comparison. Per the preregistration, we adopt the expanded
feature set only if the gain clears the meaningful-effect threshold. Track B is
also what validates the funnel idea (Phase 8): it directly measures whether
fusing the two data sources beats either alone.

### Phase 5: Track C: the WISE-only model on the full survey
A parallel track (can be a second team) using Dataset 2: ~4,350 galaxies after
the sample cuts, 114 megamaser positives, WISE colors as features, same
protocol machinery as Phase 2. Three reasons this is exciting:
- With 114 positives, cross-validation finally stabilizes. This is the sample
  where model differences might actually be detectable. That's why the one
  model-vs-model comparison we expect to be conclusive (C2: boosted trees vs
  logistic regression) lives here.
- It is the methodologically sound redo of our group's earlier poster (KNN on
  WISE colors), now with leak-free validation and direct continuity.
- It has its own hand-drawn baselines to beat: astronomers select AGN in WISE
  color space with standard cuts (Stern: W1−W2 ≥ 0.8; the Mateos wedge). Those
  play exactly the role the N_H cuts play in the X-ray plane; re-score them on
  the same held-out folds, same fairness rule, same headline plot.
This track's confirmatory comparisons are C1 (fitted boundary vs. the Stern
cut) and C2 (boosted trees vs. logistic regression) in
`docs/PREREGISTRATION.md`,
committed *before* anyone explores the 4,450 rows.

### Phase 6: Robustness checks
These re-run already-completed analyses under different assumptions to confirm
the conclusions don't depend on the specific choices made.
- **Sensitivity label check**: re-run Track A primary with and without the
  D < 170 Mpc cut. The upgrade version uses Dataset 2's per-observation RMS
  noise, velocity range, and distance to compute each galaxy's detectable
  maser luminosity, defining a "well-searched" subsample more precisely than
  the blunt distance cut.
- **Catalog check**: re-run Track A per X-ray catalog (Swift-only vs.
  XMM-only) to check for systematic differences in logNH reliability.
- **Label-disagreement check**: re-run Track A excluding the 6 galaxies
  whose maser status conflicts between the Kuo and AllWISE catalogs.

### Phase 8: The real test: a prospective candidate list
Cross-validation is rehearsal. The honest test is **prospective**: score galaxies
that were never observed by the GBT (or were observed shallowly), rank them, and
propose the top candidates for observation. The two tracks combine into a
**funnel** that mirrors a real observing campaign: WISE is all-sky and free, so
the Track-B model screens *everything*; the X-ray model then refines the
survivors where X-ray data exist. Building
that candidate catalog (e.g., from the Swift-BAT 105-month / BASS AGN samples) is
a concrete work package of its own. One caveat we must carry honestly: our
probabilities are calibrated to *this* sample, a population already selected to
have X-ray detections (maser rate 8.3%, vs 2.7% GBT-wide). Predicted probabilities
transfer only to candidate pools selected the same way.

---

## 5. Rules we always follow (and why)

These are the project's safety rails. Most "ML beats X" claims that later collapse
break one of these.

1. **All preprocessing lives inside the cross-validation loop.** Imputation and
   scaling (anything fitted to data) must be fit on the training fold only (in
   scikit-learn: put it in a `Pipeline`). Fitting on the full dataset leaks
   test information into training.
2. **No oversampling, no SMOTE, no `class_weight='balanced'` for the primary
   model.** The internet will tell you imbalanced data needs these. It doesn't; they
   exist to move a 0.5 decision threshold, but we never use one: we work with
   the full probability ranking and PR curve. What resampling *does* do is destroy
   the calibration of the predicted probabilities, which are our actual product.
3. **Thresholds are never chosen by looking at test results.** Any single
   (precision, recall) operating point we report must come from a pre-specified
   rule, e.g., "top 50 candidates by probability" (the realistic telescope-time
   budget), not from picking the prettiest point on the test PR curve.
4. **Confirmatory analyses are pre-registered; everything else is exploratory.**
   `docs/PREREGISTRATION.md` lists a small fixed set of comparisons, each fully
   specified before being run. Once signed off, those analyses run exactly as
   written, once, and get reported whatever they show. Everything else is
   **exploratory**: absolutely worth doing (that's where ideas come from), but
   labeled as such and never promoted to a headline result after we've peeked.
   With this few positives, trying ten comparisons and reporting the best one
   would virtually guarantee a false discovery. Exploratory work can be playful
   precisely *because* it can't contaminate the confirmatory claims.
5. **Only compare out-of-sample to out-of-sample.** The paper's published rates
   are in-sample (cuts chosen on the full dataset). Re-score their cuts on our
   held-out folds so both sides play by the same rules.
6. **Matched comparisons only.** Same folds, same target definition, same sample
   cuts, same features (when the question is "better boundary?"). Precision moves
   with the maser base rate, so never compare precision across differently
   filtered samples.

**Reporting conventions** (not rules, but say these things when writing up):
- The 5th–95th percentile band across CV repeats shows split-to-split
  variability on the same galaxies, not a confidence interval for new data.
  It understates true generalization uncertainty; say so in figure captions.
- Report effect sizes in science units: "~2 more real masers in the top 50
  candidates, 90% interval [−1, +5]," not naked p-values or third-decimal AUC.

---

## 6. Comparing two models when you have 24 positives

This is the statistically hardest part of the project, and where the prior student
work on this dataset was weakest. The difference between two noisy CV scores is
itself very noisy; "model A's mean was higher" is close to meaningless here. Our
protocol:

- **Pair on identical folds.** Run both models on the exact same CV splits and
  analyze the *per-fold difference*. The dominant noise source (which galaxies
  landed in which fold) is shared and cancels.
- **Use a CV-aware test.** Naive paired t-tests on CV folds are invalid (folds
  share training data → overconfident). We use the Nadeau–Bengio corrected
  resampled t-test or the 5×2cv paired test (both in `mlxtend`), or simply
  bootstrap the per-fold differences and report the interval.
- **Compare at the operating point that matters** (precision@50), not just curve
  averages like AUC; under heavy imbalance, the high-precision corner is all the
  telescope schedule cares about.
- **Default to the simpler model.** We switch to a fancier model only if it's
  *reliably* better (interval excludes zero) *and* the gain is operationally
  meaningful. "Can't tell the difference" means insufficient data, and parsimony
  breaks the tie toward logistic regression, which is also better calibrated and
  interpretable.

---

## 7. Decisions drafted in docs/PREREGISTRATION.md (review at sign-off)

These were genuinely open questions; the pre-registration draft resolves them
as follows. They bind once the team signs off, so disagree *now*, not after
the results are in.

- **Primary target (Track A): megamaser-or-disk (types 2–4, 53 positives).**
  Matches the paper's megamaser detection rates; kilomasers count as negatives
  (they're physically different, often star-formation-powered). The
  scientifically prized disk-maser target (type 4, 24 positives, which the
  pilot pipeline used) is exploratory, with the 8 ambiguous type-3 candidates
  excluded from its sample rather than counted as noisy negatives. Track B
  uses the same target on the X-ray+WISE matched sample; Track C's WISE-only
  target is megamaser via listed luminosity ≥ 10 L_sun (114 positives).
- **Distance as a feature: excluded from all confirmatory analyses**, which
  compare against paper/literature cuts that don't use it; the D < 170 Mpc
  robustness re-run covers the sensitivity concern. The *operational* framing
  (include distance, because a telescope scheduler wants detection
  probability) is legitimate and lives in exploratory/deployment work, but the
  two framings are never mixed in one comparison.
- **Numbers worth a second look before sign-off**: the "+2 masers per
  campaign" meaningful-effect threshold, with a single top-50 observing
  campaign used for all confirmatory precision@k operating points.

---

## 8. Glossary

- **AGN**: active galactic nucleus; a feeding supermassive black hole.
- **N_H**: hydrogen column density along our sight line, in cm⁻²; log N_H ≥ 23
  is "heavily obscured," ≥ 24 is "Compton-thick."
- **Precision / detection rate**: fraction of selected candidates that are real.
- **Recall / completeness**: fraction of real masers that get selected.
- **PR curve**: precision vs. recall as the probability threshold slides.
- **Stratified k-fold CV**: split data into k parts keeping the maser fraction
  equal in each; train on k−1, test on the held-out part; rotate.
- **Calibrated probability**: "70%" means it happens ~70% of the time; needed if
  the numbers are to be used for scheduling decisions, not just ranking.
