"""Methods-evaluation harness for the maser project.

This is the ANALYSIS half of the synthetic-data work; the generator lives in
src/synth_data.py. Everything here operates on a pandas DataFrame plus column
names (features + target), so the exact same calls run on synthetic data and on
the real frames from maser_data.py. That portability is the point: develop the
comparison here, then run it unchanged on real data once the plan is frozen.

Vocabulary:
  arm    = (model_builder, feature_columns) -- one side of a comparison.
  maker  = seed -> DataFrame (e.g. maker("xray", scenario="wedge")).
A MODEL comparison fixes the columns and varies the builder; a FEATURE
comparison fixes the builder and varies the columns. Both are just two arms.

Run `python src/synth_analysis.py` for a study (provisional, see the guide).
"""
import numpy as np
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from synth_data import make_dataset, make_fused_dataset
from maser_data import FEATURES, TARGET


# --- models (the candidates the pre-registration compares) ------------------
def plain_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def interaction_lr():
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
        LogisticRegression(max_iter=1000))


def quadratic_lr():
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, interaction_only=False, include_bias=False),
        LogisticRegression(max_iter=1000))


def rf():
    return RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                  min_samples_leaf=5, random_state=0)


def gbt():
    return GradientBoostingClassifier(max_depth=2, n_estimators=100,
                                      learning_rate=0.1, random_state=0)


# --- hand-drawn cuts as fixed inside/outside rules --------------------------
# Paper/literature cuts are not probabilistic models and not rankings. They are
# fixed boolean selection rules. Compare them to fitted models at the cut's own
# held-out recall, not at an invented top-k budget.
FIG5_SLOPE = 1.019
FIG5_INTERCEPT = {23: -1.411, 24: -2.489}
KUO_L12_CUT = 42.0
STERN_W1W2_CUT = 0.8


class HardCut:
    def __init__(self, select_fn, name="cut"):
        self.select_fn, self.name = select_fn, name

    def __call__(self):          # so (cut, cols) parallels (builder, cols)
        return self

    def fit(self, X, y):         # a hand-drawn cut ignores the training labels
        return self

    def select(self, X):
        return np.asarray(self.select_fn(X), dtype=bool)


def xray_cut():
    """Kuo combined cut on FEATURES['xray'] = [L12, Lob].

    The paper's N_H threshold is inferred from the Figure-5 L12-Lob plane, not
    from tabulated spectral-fit logNH: inferred log N_H >= 24 when
    Lob <= 1.019 * L12 - 2.489. The best combined cut additionally requires
    L12 > 42. This is a hard inside/outside rule.
    """
    def select(X):
        l12, lob = X[:, 0], X[:, 1]
        nh24 = lob <= FIG5_SLOPE * l12 + FIG5_INTERCEPT[24]
        luminous = l12 > KUO_L12_CUT
        finite = np.isfinite(l12) & np.isfinite(lob)
        return finite & nh24 & luminous
    return HardCut(select, "Kuo N_H>=24 and L12>1e42 cut")


def stern_cut():
    """Stern WISE AGN cut on FEATURES['wise'] = [w1w2, w2w3].

    The Stern selector is the hard threshold W1-W2 >= 0.8.
    """
    return HardCut(lambda X: np.isfinite(X[:, 0]) & (X[:, 0] >= STERN_W1W2_CUT),
                   "Stern W1-W2>=0.8 cut")


# --- makers: seed -> DataFrame ----------------------------------------------
def maker(plane="xray", **kw):
    return lambda seed: make_dataset(plane=plane, seed=seed, **kw)


def fused_maker(**kw):
    return lambda seed: make_fused_dataset(seed=seed, **kw)


# --- evaluation harness (operates on DataFrame + column names) --------------
def _precision_at_k(y_true, proba, k):
    order = np.argsort(proba)[::-1][:k]
    return y_true[order].mean()


def _cut_precision_recall(y_true, selected):
    y_true = np.asarray(y_true)
    selected = np.asarray(selected, dtype=bool)
    n_pos = y_true.sum()
    if n_pos == 0:
        return np.nan, np.nan
    n_sel = selected.sum()
    tp = (selected & (y_true == 1)).sum()
    precision = 0.0 if n_sel == 0 else tp / n_sel
    recall = tp / n_pos
    return precision, recall


def _best_precision_at_min_recall(y_true, proba, min_recall):
    """Highest model precision among thresholds with recall >= min_recall."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n_pos = y_true.sum()
    if n_pos == 0:
        return np.nan
    if min_recall <= 0:
        return _precision_at_k(y_true, proba, 1)
    order = np.argsort(proba)[::-1]
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    denom = np.arange(1, len(y_sorted) + 1)
    recall = tp / n_pos
    precision = tp / denom
    ok = recall >= min_recall
    return np.nan if not np.any(ok) else precision[ok].max()


def cv_precision_diff(df, arm_a, arm_b, target, k, n_splits=5, n_repeats=5,
                      seed=0):
    """Per-fold precision@k difference (arm A - arm B) on identical folds.

    Each arm is (model_builder, feature_columns). Works on any DataFrame that
    has those columns, real or synthetic.
    """
    (ba, fa), (bb, fb) = arm_a, arm_b
    Xa, Xb = df[fa].to_numpy(), df[fb].to_numpy()
    y = df[target].to_numpy()
    diffs = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed + r)
        for tr, te in skf.split(Xa, y):
            if y[tr].sum() == 0 or y[te].sum() == 0:
                continue
            pa = ba().fit(Xa[tr], y[tr]).predict_proba(Xa[te])[:, 1]
            pb = bb().fit(Xb[tr], y[tr]).predict_proba(Xb[te])[:, 1]
            diffs.append(_precision_at_k(y[te], pa, k)
                         - _precision_at_k(y[te], pb, k))
    return np.array(diffs)


def cv_matched_recall_diff(df, model_arm, cut_arm, target, n_splits=5,
                           n_repeats=5, seed=0):
    """Per-fold precision difference: model at cut recall minus hard cut.

    The cut is rescored as a fixed inside/outside rule on each held-out fold.
    The fitted model is then evaluated at the same held-out recall by choosing
    the threshold with highest precision subject to recall >= the cut's recall.
    If the cut recall is 0, use model precision@1 and cut precision 0 when it
    selects no true positives, matching the pre-registration.
    """
    (bm, fm), (bc, fc) = model_arm, cut_arm
    Xm, Xc = df[fm].to_numpy(), df[fc].to_numpy()
    y = df[target].to_numpy()
    diffs = []
    cut_precs, cut_recalls, model_precs = [], [], []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed + r)
        for tr, te in skf.split(Xm, y):
            if y[tr].sum() == 0 or y[te].sum() == 0:
                continue
            model = bm().fit(Xm[tr], y[tr])
            proba = model.predict_proba(Xm[te])[:, 1]
            selected = bc().fit(Xc[tr], y[tr]).select(Xc[te])
            cut_prec, cut_recall = _cut_precision_recall(y[te], selected)
            model_prec = _best_precision_at_min_recall(y[te], proba, cut_recall)
            if np.isnan(model_prec) or np.isnan(cut_prec):
                continue
            diffs.append(model_prec - cut_prec)
            cut_precs.append(cut_prec)
            cut_recalls.append(cut_recall)
            model_precs.append(model_prec)
    return dict(diffs=np.array(diffs),
                cut_precision=np.array(cut_precs),
                cut_recall=np.array(cut_recalls),
                model_precision=np.array(model_precs))


def nadeau_bengio(diffs, n_splits, alpha=0.10):
    """One-sided NB-corrected paired t-test that mean(diffs) > 0.

    Returns (mean, ci_low, p_one_sided). The correction inflates the naive
    variance by the train/test overlap factor (rho = 1/n_splits), which the
    naive paired t-test ignores (Nadeau & Bengio 2003).
    """
    d = np.asarray(diffs)
    m, var, J = d.mean(), d.var(ddof=1), len(d)
    rho = 1.0 / n_splits
    se = np.sqrt(var * (1.0 / J + rho / (1.0 - rho)))
    if se == 0:
        return m, m, 0.0 if m > 0 else 1.0
    t = m / se
    df = J - 1
    return m, m - stats.t.ppf(1 - alpha, df) * se, stats.t.sf(t, df)


def effect_ceiling(make_big, arm_a, arm_b, target, k=2000):
    """TRUE effect at large n: how much arm A beats arm B once noise is gone.

    `make_big` should produce large-n frames (e.g. maker(..., n=40000)). Near
    zero means there is nothing for any finite-sample test to find; large means
    a low power is 'real effect, too little data', not 'no effect'.
    """
    (ba, fa), (bb, fb) = arm_a, arm_b
    tr, te = make_big(1), make_big(2)
    ya, yte = tr[target].to_numpy(), te[target].to_numpy()
    pa = ba().fit(tr[fa].to_numpy(), ya).predict_proba(te[fa].to_numpy())[:, 1]
    pb = bb().fit(tr[fb].to_numpy(), ya).predict_proba(te[fb].to_numpy())[:, 1]
    return dict(dAUC=round(roc_auc_score(yte, pa) - roc_auc_score(yte, pb), 3),
                dPrec=round(_precision_at_k(yte, pa, k)
                           - _precision_at_k(yte, pb, k), 3))


def power_study(make, arm_a, arm_b, target, k=10, n_sims=50, effect_masers=2.0,
                n_splits=5, n_repeats=5, alpha=0.10, seed0=0):
    """Fraction of simulated datasets where the asymmetric decision rule fires.

    The rule (matching the pre-registration): adopt arm A over arm B only if the
    one-sided NB CI excludes zero AND the effect is >= `effect_masers` extra true
    masers in a full-sample TOP-`k*n_splits` CAMPAIGN. Per-fold precision@k is a
    rate at the top-k/fold operating point; a fold is 1/n_splits of the sample,
    so the same rate over the full sample is a top-(k*n_splits) list. With the
    defaults that is top-50 (k=10); use k=20 for the WISE track's top-100.
    Run under an alternative truth for POWER; under a null truth (linear, or
    wise_signal=0) for the FALSE-POSITIVE rate.
    """
    campaign = k * n_splits          # full-sample candidate budget (top-50/100)
    fires, effects = 0, []
    for s in range(n_sims):
        df = make(seed0 + s)
        diffs = cv_precision_diff(df, arm_a, arm_b, target, k, n_splits,
                                  n_repeats, seed=1000 + s)
        if len(diffs) == 0:
            continue
        m, ci_low, _ = nadeau_bengio(diffs, n_splits, alpha)
        effects.append(m * campaign)
        if ci_low > 0 and m * campaign >= effect_masers:
            fires += 1
    return dict(n_sims=n_sims, campaign=campaign, rate=round(fires / n_sims, 3),
                mean_effect_masers=round(float(np.mean(effects)), 2))


def cut_power_study(make, model_arm, cut_arm, target, n_sims=50,
                    effect_masers=2.0, n_splits=5, n_repeats=5, alpha=0.10,
                    seed0=0):
    """Power for matched-recall model-vs-cut comparisons.

    The precision difference is converted to extra masers using the mean number
    of galaxies selected by the hard cut. Unlike top-k comparisons, the cut's
    selected count is part of its operating point and can vary by fold.
    """
    fires, effects, selected_counts = 0, [], []
    for s in range(n_sims):
        df = make(seed0 + s)
        out = cv_matched_recall_diff(df, model_arm, cut_arm, target, n_splits,
                                     n_repeats, seed=1000 + s)
        diffs = out["diffs"]
        if len(diffs) == 0:
            continue
        m, ci_low, _ = nadeau_bengio(diffs, n_splits, alpha)
        cut = cut_arm[0]().select(df[cut_arm[1]].to_numpy())
        n_selected = int(cut.sum())
        effect = m * n_selected
        effects.append(effect)
        selected_counts.append(n_selected)
        if ci_low > 0 and effect >= effect_masers:
            fires += 1
    return dict(n_sims=n_sims, rate=round(fires / n_sims, 3),
                mean_selected=round(float(np.mean(selected_counts)), 1),
                mean_effect_masers=round(float(np.mean(effects)), 2))


def cv_auc(build, df, features, target, n_splits=5, n_repeats=3, seed=0):
    X, y = df[features].to_numpy(), df[target].to_numpy()
    aucs = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed + r)
        for tr, te in skf.split(X, y):
            if y[tr].sum() == 0 or y[te].sum() == 0:
                continue
            p = build().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
    return np.array(aucs)


def _brier_decomposition(y_true, proba, n_bins=10):
    """Murphy decomposition: brier = reliability - resolution + uncertainty.

    reliability (calibration error, smaller is better): squared gap between
    predicted probability and observed frequency, averaged within probability
    bins. This is the number a reliability diagram draws. resolution
    (sharpness/discrimination, larger is better): how far the per-bin frequencies
    sit from the base rate -- a model that predicts the base rate for everyone is
    perfectly calibrated (reliability 0) but useless (resolution 0). uncertainty:
    base-rate variance p(1-p), fixed by the data, not the model. The three
    reconstruct brier_score_loss up to binning, and split the single Brier number
    into the two things we actually care about.
    """
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    n = len(y_true)
    base = y_true.mean()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1]), 0, n_bins - 1)
    reliability = resolution = 0.0
    for b in range(n_bins):
        m = idx == b
        nk = int(m.sum())
        if nk == 0:
            continue
        reliability += nk * (proba[m].mean() - y_true[m].mean()) ** 2
        resolution += nk * (y_true[m].mean() - base) ** 2
    return dict(reliability=reliability / n, resolution=resolution / n,
                uncertainty=base * (1.0 - base))


def cv_brier(build, df, features, target, n_splits=5, n_repeats=5, seed=0,
             truth=None):
    """Per-fold Brier score and Brier skill score on held-out predictions.

    Scores held-out probabilities against the OBSERVED target by default. Pass
    truth="z_true" to score against the latent maser label, or truth="p_true"/
    "p_obs" to score against the generator's true probabilities (synthetic only).
    Scoring against a probability is a far lower-variance calibration check than
    against binary outcomes, because it removes the 0/1 sampling noise that
    dominates at our handful of positives.

    Brier skill score (bss) = 1 - brier_model / brier_baseline, where the
    baseline predicts the TRAIN base rate for every held-out galaxy. bss>0 beats
    the base rate, bss=1 is perfect, bss<0 is worse than guessing prevalence.
    Reported alongside the raw Brier because at 3-8% prevalence the raw number is
    dominated by the base rate and is not comparable across differently filtered
    samples.
    """
    X = df[features].to_numpy()
    y = df[target].to_numpy()
    label = None if truth is None else df[truth].to_numpy()
    briers, skills = [], []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed + r)
        for tr, te in skf.split(X, y):
            if y[tr].sum() == 0 or y[te].sum() == 0:
                continue
            p = build().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            scored = y[te] if label is None else label[te]
            base = y[tr].mean()          # base rate from TRAIN only, no leak
            bs = np.mean((p - scored) ** 2)
            bs_base = np.mean((base - scored) ** 2)
            briers.append(bs)
            skills.append(np.nan if bs_base == 0 else 1.0 - bs / bs_base)
    return dict(brier=np.array(briers), bss=np.array(skills))


def calibration_study(build, make, features, target, n_sims=15, n_splits=5,
                      n_repeats=5, seed0=0):
    """Brier, skill score, and the calibration/sharpness split across sims.

    brier and bss average held-out folds over `n_sims` synthetic datasets. The
    reliability/resolution split needs many points per bin, so it is computed on
    each dataset's pooled held-out predictions, then averaged. bss_vs_truth
    rescore the same predictions against `p_obs` (the generative probability):
    the gap from bss is the part of the apparent miscalibration that is really
    just label noise rather than a bad model.
    """
    briers, skills, truth_skills = [], [], []
    rel, res, unc = [], [], []
    for s in range(n_sims):
        df = make(seed0 + s)
        out = cv_brier(build, df, features, target, n_splits, n_repeats,
                       seed=4000 + s)
        briers.append(out["brier"].mean())
        skills.append(np.nanmean(out["bss"]))
        truth_skills.append(np.nanmean(
            cv_brier(build, df, features, target, n_splits, n_repeats,
                     seed=4000 + s, truth="p_obs")["bss"]))
        X, y = df[features].to_numpy(), df[target].to_numpy()
        p_all, y_all = [], []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=5000 + s)
        for tr, te in skf.split(X, y):
            if y[tr].sum() == 0 or y[te].sum() == 0:
                continue
            p_all.append(build().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
            y_all.append(y[te])
        d = _brier_decomposition(np.concatenate(y_all), np.concatenate(p_all))
        rel.append(d["reliability"])
        res.append(d["resolution"])
        unc.append(d["uncertainty"])
    return dict(brier=round(float(np.mean(briers)), 4),
                bss=round(float(np.mean(skills)), 3),
                bss_vs_truth=round(float(np.mean(truth_skills)), 3),
                reliability=round(float(np.mean(rel)), 4),
                resolution=round(float(np.mean(res)), 4),
                uncertainty=round(float(np.mean(unc)), 4))


def optimism(build, make, features, target, n_sims=15, seed0=0):
    """Resubstitution AUC minus honest CV AUC: how much a model flatters itself

    on the data it was fit to. Flexible models inflate more at small n, which is
    exactly what looks impressive but will not replicate.
    """
    resub, cv = [], []
    for s in range(n_sims):
        df = make(seed0 + s)
        X, y = df[features].to_numpy(), df[target].to_numpy()
        resub.append(roc_auc_score(y, build().fit(X, y).predict_proba(X)[:, 1]))
        cv.append(cv_auc(build, df, features, target).mean())
    return dict(resub_auc=round(float(np.mean(resub)), 3),
                cv_auc=round(float(np.mean(cv)), 3),
                optimism=round(float(np.mean(resub) - np.mean(cv)), 3))


def k_sensitivity(build, make, features, target, ks=(5, 10, 20, 50), n_sims=15,
                  n_splits=5, seed0=0):
    """Across-fold spread of precision@k for several k. Small k is noisier, so a

    'top-50' operating point can be dominated by fold luck at our sample size.
    """
    out = {}
    for k in ks:
        per = []
        for s in range(n_sims):
            df = make(seed0 + s)
            X, y = df[features].to_numpy(), df[target].to_numpy()
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=3000 + s)
            for tr, te in skf.split(X, y):
                if y[tr].sum() == 0 or y[te].sum() == 0:
                    continue
                p = build().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                per.append(_precision_at_k(y[te], p, k))
        per = np.array(per)
        out[k] = dict(mean=round(per.mean(), 3), std=round(per.std(), 3))
    return out


if __name__ == "__main__":
    XF, XT = FEATURES["xray"], TARGET["xray"]

    print("Large-n effect ceiling, X-ray plane (test set n=40,000, GMM features):")
    print("  Values are delta AUC: candidate model AUC minus plain-LR AUC.")
    print("  scenario      interaction_lr_delta_auc      quadratic_lr_delta_auc")
    for sc in ("linear", "wedge", "box", "interaction", "blob"):
        big = maker("xray", scenario=sc, n=40000, strength=3.0,
                    feature_model="gmm")
        ci = effect_ceiling(big, (interaction_lr, XF), (plain_lr, XF), XT)
        cq = effect_ceiling(big, (quadratic_lr, XF), (plain_lr, XF), XT)
        print(f"  {sc:12s}  {ci['dAUC']:+.3f}                    {cq['dAUC']:+.3f}")
    print("  -> wedge, box, and interaction have real large-n signal;")
    print("     none except the blob clears the n=641 adoption rule below.")

    print("\nModel-vs-model power at n=641:")
    print("  power_detected = fraction of simulated datasets where CI>0 and")
    print("  mean_extra_masers_top50 >= 2.")
    for sc, a, b, note in (
            ("linear", interaction_lr, plain_lr, "false-positive check"),
            ("wedge", interaction_lr, plain_lr, "wedge present"),
            ("box", interaction_lr, plain_lr, "hard box present"),
            ("interaction", interaction_lr, plain_lr, "real interaction present"),
            ("blob", quadratic_lr, plain_lr, "positive control")):
        pw = power_study(maker("xray", scenario=sc, strength=3.0),
                         (a, XF), (b, XF), XT, n_sims=40)
        print(f"  truth={sc:12s} power_detected={pw['rate']:.2f}  "
              f"mean_extra_masers_top50={pw['mean_effect_masers']:+.2f}  ({note})")

    print("\nHEADLINE: fitted model vs the paper's hand-drawn cuts")
    print("  Hard cuts are compared at matched recall. Effects are extra masers")
    print("  at the cut's own selected-count operating point, not top-k.")
    for sc in ("wedge", "box", "interaction"):
        pw = cut_power_study(maker("xray", scenario=sc, strength=3.0),
                             (interaction_lr, XF), (xray_cut, XF), XT,
                             n_sims=40)
        print(f"  X-ray  vs Kuo cut, truth={sc:12s} "
              f"power_detected={pw['rate']:.2f}  "
              f"mean_extra_masers_at_cut={pw['mean_effect_masers']:+.2f}  "
              f"mean_cut_selected={pw['mean_selected']:.1f} galaxies")
    WF, WT = FEATURES["wise"], TARGET["wise"]
    for sc in ("wedge", "interaction"):
        pw = cut_power_study(maker("wise", scenario=sc, strength=3.0),
                             (interaction_lr, WF), (stern_cut, WF), WT,
                             n_sims=40)
        print(f"  WISE   vs Stern cut, truth={sc:12s} "
              f"power_detected={pw['rate']:.2f}  "
              f"mean_extra_masers_at_cut={pw['mean_effect_masers']:+.2f}  "
              f"mean_cut_selected={pw['mean_selected']:.1f} galaxies")

    print("\nFUSION: does adding WISE colours to X-ray help? (LR, 4 feat vs 2)")
    for ws, label in ((0.0, "redundant WISE"), (1.5, "WISE adds signal")):
        arm_a, arm_b = (plain_lr, FEATURES["fused"]), (plain_lr, FEATURES["xray"])
        ceil = effect_ceiling(
            fused_maker(scenario="linear", wise_signal=ws, n=40000,
                        strength=3.0, feature_model="gmm"),
            arm_a, arm_b, TARGET["fused"])
        pw = power_study(
            fused_maker(scenario="linear", wise_signal=ws, strength=3.0),
            arm_a, arm_b, TARGET["fused"], n_sims=40)
        print(f"  wise_signal={ws} ({label:16s}): "
              f"large_n_delta_auc={ceil['dAUC']:+.3f}  "
              f"power_detected_at_n602={pw['rate']:.2f}  "
              f"mean_extra_masers_top{pw['campaign']}={pw['mean_effect_masers']:+.2f}")

    print("\nOPTIMISM (resub AUC - honest CV AUC) at n=641, interaction truth:")
    mk = maker("xray", scenario="interaction", strength=3.0)
    for name, build in (("plain LR", plain_lr), ("interaction LR", interaction_lr),
                        ("random forest", rf), ("boosted trees", gbt)):
        o = optimism(build, mk, XF, XT, n_sims=12)
        print(f"  {name:14s}: train_auc={o['resub_auc']:.3f}  "
              f"cv_auc={o['cv_auc']:.3f}  optimism_delta_auc={o['optimism']:+.3f}")

    print("\nTOP-k NOISE (across-fold std of precision@k), X-ray, plain LR:")
    for k, v in k_sensitivity(plain_lr, mk, XF, XT).items():
        print(f"  per_fold_precision_at_{k:<2d}: mean={v['mean']:.3f}  "
              f"std_across_folds={v['std']:.3f}")

    print("\nCALIBRATION (Brier + skill score + sharpness split), X-ray, "
          "interaction truth:")
    print("  bss>0 beats the base rate; bss_vs_truth scores against p_obs (the")
    print("  generative probability) so it strips out label-noise miscalibration.")
    for name, build in (("plain LR", plain_lr), ("interaction LR", interaction_lr),
                        ("boosted trees", gbt)):
        c = calibration_study(build, mk, XF, XT, n_sims=12)
        print(f"  {name:14s}: brier={c['brier']:.4f}  bss={c['bss']:+.3f}  "
              f"bss_vs_truth={c['bss_vs_truth']:+.3f}  "
              f"reliability={c['reliability']:.4f}  resolution={c['resolution']:.4f}")
    print("  distance-censored truth (maser=0 can mean 'not detected'):")
    mkd = maker("xray", scenario="interaction", strength=3.0, distance_noise=True)
    c = calibration_study(plain_lr, mkd, XF, XT, n_sims=12)
    print(f"  plain LR      : brier={c['brier']:.4f}  bss={c['bss']:+.3f}  "
          f"bss_vs_truth={c['bss_vs_truth']:+.3f}  "
          f"reliability={c['reliability']:.4f}  resolution={c['resolution']:.4f}")
