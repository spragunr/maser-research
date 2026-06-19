"""Synthetic maser-like catalogs for blind methods exploration (DATA ONLY).

This module ONLY generates synthetic data. The harness that evaluates models on
it lives in src/synth_analysis.py.

WHY THIS EXISTS
---------------
We want to understand how our models behave (power, variance, calibration) at
our tiny positive counts WITHOUT hacking on the real galaxy labels and
p-hacking a result into the paper. Develop and tune on simulations, freeze the
plan, then look at the real data once.

THE FIREWALL (read this)
------------------------
This module reads real feature columns only, never using real labels to choose
or fit a decision boundary. The default feature sampler is a jittered bootstrap
of real feature rows; a smooth Gaussian-mixture sampler is available as
`feature_model="gmm"`, and a correlated-Gaussian summary sampler is available as
`feature_model="gaussian"`. The thing you must invent yourself, never estimate
from real labels, is the true decision boundary (`scenario`).

SAME FORMAT AS THE REAL DATA
----------------------------
make_dataset / make_fused_dataset return a pandas DataFrame whose feature and
target columns are named exactly as in maser_data.py (FEATURES / TARGET there).
So a pipeline written `X, y = df[FEATURES[t]], df[TARGET[t]]` runs unchanged on
synthetic and real data. Synthetic frames also carry diagnostic columns the real
data lacks: `z_true` (the latent true label), `detected` (whether a true maser
was bright enough to be seen), and the generative probabilities `p_true` (P of a
latent maser) and `p_obs` (P the observed target is 1, after sensitivity). That
absence is itself the point: with real data you never know which negatives are
false negatives, nor the probability that produced each label, so scoring a
model against `p_true`/`p_obs` is a calibration check available only here.

Run `python src/synth_data.py` for dataset summaries.
"""
import numpy as np
import pandas as pd
from functools import lru_cache
from sklearn.mixture import GaussianMixture

try:
    from .maser_data import load_wise, load_xray_sample, load_xray_with_wise
except ImportError:  # Support `python src/synth_data.py` from the repo root.
    from maser_data import load_wise, load_xray_sample, load_xray_with_wise

# --- X-only configuration ---------------------------------------------------
# `columns`, `target`, `dist_col` mirror the real column names in maser_data.py.
# The means/stds/correlation define the optional Gaussian feature model and the
# standardized coordinates used by synthetic truth scenarios. `signal_signs`
# orients those coordinates so "high risk" matches the a priori quadrant:
# high L12/low observed X-ray luminosity for X-ray, high/high for WISE colours.
# The prevalence, sample size, and distance scale set the synthetic
# label/sensitivity process.
PLANES = {
    "xray": dict(
        columns=["L_12um_bestfit_1", "Lob"], target="is_megamaser_plus",
        dist_col="Lum_dis",                 # luminosity distance, Mpc
        means=(42.375, 41.632), stds=(1.371, 1.537), corr=0.737,
        signal_signs=(1.0, -1.0),
        prevalence=0.083, n=641,
        dist_mean=104.0, dist_std=82.0, sens_d50=170.0,
    ),
    "wise": dict(
        columns=["w1w2", "w2w3"], target="is_megamaser",
        dist_col="velocity",                # recession velocity, km/s
        means=(0.240, 2.972), stds=(0.324, 0.991), corr=0.388,
        signal_signs=(1.0, 1.0),
        prevalence=0.030, n=4400,
        dist_mean=7767.0, dist_std=4734.0, sens_d50=12000.0,
    ),
}

# Joint X-ray + WISE structure, measured on the 602 galaxies that have both
# (X-only). Column order matches maser_data's load_xray_with_wise. Note W1W2
# correlates with L12um at 0.82, so WISE colour is largely redundant with X-ray
# luminosity in the real data -- what the fusion comparison must contend with.
FUSED = dict(
    columns=["L_12um_bestfit_1", "Lob", "wise_w1w2", "wise_w2w3"],
    target="is_megamaser_plus", dist_col="Lum_dis",
    means=(42.392, 41.652, 0.500, 2.842),
    stds=(1.354, 1.526, 0.434, 1.055),
    corr=[[1.000, 0.741, 0.817, 0.568],
          [0.741, 1.000, 0.571, 0.158],
          [0.817, 0.571, 1.000, 0.543],
          [0.568, 0.158, 0.543, 1.000]],
    signal_signs=(1.0, -1.0, 1.0, 1.0),
    prevalence=0.083, n=602,
    dist_mean=104.0, dist_std=82.0, sens_d50=170.0,
)

BOX_THRESHOLD = 0.0


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def _sample_distance(spec, n, rng):
    """Right-skewed distances matching the real mean/std (lognormal)."""
    m, s = spec["dist_mean"], spec["dist_std"]
    sigma = np.sqrt(np.log(1 + (s / m) ** 2))
    mu = np.log(m) - 0.5 * sigma ** 2
    return rng.lognormal(mu, sigma, size=n)


@lru_cache(maxsize=None)
def _feature_pool(kind):
    """Real feature-only pools for label-blind jittered bootstrapping.

    Loaders do create labels internally, but this function immediately discards
    every label/target column and returns only the feature columns plus distance.
    Synthetic labels are always generated from the chosen scenario below.
    """
    if kind == "xray":
        spec = PLANES["xray"]
        cols = spec["columns"] + [spec["dist_col"]]
        return spec, load_xray_sample()[cols].dropna().reset_index(drop=True)
    if kind == "wise":
        spec = PLANES["wise"]
        cols = spec["columns"] + [spec["dist_col"]]
        df = load_wise()
        df = df[df["velocity"].notna() & (df["velocity"] < 21000)]
        return spec, df[cols].dropna().reset_index(drop=True)
    if kind == "fused":
        spec = FUSED
        cols = spec["columns"] + [spec["dist_col"]]
        df = load_xray_with_wise()
        df = df[df["wise_sep_arcsec"].notna()]
        return spec, df[cols].dropna().reset_index(drop=True)
    raise ValueError(f"unknown feature pool {kind!r}")


@lru_cache(maxsize=None)
def _gmm_feature_model(kind, max_components=8):
    """Fit a label-blind Gaussian mixture to features plus log-distance."""
    spec, pool = _feature_pool(kind)
    cols = spec["columns"]
    arr = pool[cols].to_numpy(dtype=float)
    # Distance proxies can include a few non-positive local velocities. For the
    # synthetic sensitivity model they are distances, so keep the GMM distance
    # coordinate nonnegative.
    dist = np.clip(pool[spec["dist_col"]].to_numpy(dtype=float), 0.0, None)
    log_dist = np.log1p(dist)
    Z = np.column_stack([arr, log_dist])

    center = Z.mean(axis=0)
    scale = Z.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    Zs = (Z - center) / scale

    n_max = max(1, min(max_components, len(pool)))
    best = None
    best_bic = np.inf
    for n_components in range(1, n_max + 1):
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=1e-4,
            random_state=0,
        )
        model.fit(Zs)
        bic = model.bic(Zs)
        if bic < best_bic:
            best = model
            best_bic = bic

    return spec, best, center, scale


def _sample_gmm(kind, n, rng):
    """Draw from the cached GMM using this dataset's RNG."""
    spec, model, center, scale = _gmm_feature_model(kind)
    components = rng.choice(model.n_components, size=n, p=model.weights_)
    Zs = np.empty((n, model.means_.shape[1]))

    for component in np.unique(components):
        rows = np.flatnonzero(components == component)
        Zs[rows] = rng.multivariate_normal(
            model.means_[component],
            model.covariances_[component],
            size=len(rows),
        )

    Z = Zs * scale + center
    X = Z[:, :-1]
    dist = np.expm1(Z[:, -1])
    return X, np.clip(dist, 0.0, None)


def _sample_features(kind, n, rng, feature_model, jitter):
    """Draw feature matrix and distance column from the requested X-only model."""
    if kind in PLANES:
        spec = PLANES[kind]
    elif kind == "fused":
        spec = FUSED
    else:
        raise ValueError(f"unknown feature kind {kind!r}")

    if feature_model == "gaussian":
        means, stds = np.array(spec["means"]), np.array(spec["stds"])
        corr = np.array(spec["corr"])
        if corr.ndim == 0:
            corr = np.array([[1.0, float(corr)], [float(corr), 1.0]])
        cov = corr * np.outer(stds, stds)
        X = rng.multivariate_normal(means, cov, size=n)
        dist = _sample_distance(spec, n, rng)
        return X, dist

    if feature_model == "bootstrap":
        spec, pool = _feature_pool(kind)
        take = rng.integers(0, len(pool), size=n)
        sampled = pool.iloc[take].reset_index(drop=True).copy()
        X = sampled[spec["columns"]].to_numpy(dtype=float)
        if jitter:
            scale = pool[spec["columns"]].std(ddof=0).to_numpy(dtype=float)
            X = X + rng.normal(0.0, jitter * scale, size=X.shape)
        dist = sampled[spec["dist_col"]].to_numpy(dtype=float)
        return X, dist

    if feature_model == "gmm":
        return _sample_gmm(kind, n, rng)

    raise ValueError("feature_model must be 'bootstrap', 'gmm', or 'gaussian'")


def _score(Xz, scenario, strength):
    """Latent log-odds shape on standardized risk coordinates Xz (n, 2).

    scenario chooses the TRUE boundary family:
      linear      : rises with z1 + z2            (a straight boundary)
      wedge       : rises with min(z1, z2)        (a soft corner)
      box         : a jump where both risk coordinates exceed BOX_THRESHOLD
      interaction : a genuine multiplicative z1*z2 term
      blob        : peaks at the centre           (a central island)
    `strength` scales the effect size (signal-to-noise of the boundary).
    """
    z1, z2 = Xz[:, 0], Xz[:, 1]
    if scenario == "linear":
        return strength * (z1 + z2) / np.sqrt(2)
    if scenario == "wedge":
        return strength * (np.minimum(z1, z2) - 0.5)
    if scenario == "box":
        return strength * ((z1 > BOX_THRESHOLD) & (z2 > BOX_THRESHOLD)).astype(float)
    if scenario == "interaction":
        return strength * (0.4 * z1 + 0.4 * z2 + 0.8 * z1 * z2)
    if scenario == "blob":
        return strength * (1.0 - (z1 ** 2 + z2 ** 2))
    raise ValueError(f"unknown scenario {scenario!r}")


def _calibrate_intercept(score, sens, target_prev):
    """Find b0 so that mean( sigmoid(b0+score) * sens ) == target_prev."""
    lo, hi = -60.0, 60.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.mean(_sigmoid(mid + score) * sens) < target_prev:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _risk_coordinates(Z, spec):
    """Orient standardized features so larger coordinates mean higher risk."""
    signs = np.array(spec["signal_signs"], dtype=float)
    return Z * signs


def _assemble(spec, X, score, dist, distance_noise, target_prev, rng):
    """Shared label-generation + DataFrame assembly for both generators."""
    n = X.shape[0]
    if distance_noise:
        sens = _sigmoid((spec["sens_d50"] - dist) / (0.25 * spec["sens_d50"]))
    else:
        sens = np.ones(n)
    b0 = _calibrate_intercept(score, sens, target_prev)
    p_true = _sigmoid(b0 + score)
    z_true = (rng.uniform(size=n) < p_true).astype(int)
    detected = (rng.uniform(size=n) < sens).astype(int)
    y = z_true * detected

    df = pd.DataFrame(X, columns=spec["columns"])
    df[spec["target"]] = y
    df[spec["dist_col"]] = dist
    df["z_true"] = z_true        # synthetic-only: the latent true label
    df["detected"] = detected    # synthetic-only: was a true maser seen
    df["p_true"] = p_true        # synthetic-only: P(latent maser) = sigmoid(b0+score)
    df["p_obs"] = p_true * sens  # synthetic-only: P(observed maser=1), after sensitivity
    return df


def make_dataset(plane="xray", scenario="wedge", n=None, strength=2.5,
                 distance_noise=False, prevalence=None, seed=0,
                 feature_model="bootstrap", jitter=0.03):
    """One synthetic catalog on a single 2-feature plane, as a DataFrame.

    Columns: the two real feature names, the real binary target name, the real
    distance column, plus `z_true`, `detected`, `p_true`, and `p_obs`
    (synthetic-only). With
    distance_noise=False, detected==1 and the observed target == z_true. With
    distance_noise=True, distant true masers can be missed (target=0), the
    maser=0-means-"not detected" problem in controllable form. Config is stored
    in df.attrs.
    """
    p = PLANES[plane]
    n = n or p["n"]
    target_prev = prevalence if prevalence is not None else p["prevalence"]
    rng = np.random.default_rng(seed)

    X, dist = _sample_features(plane, n, rng, feature_model, jitter)
    Xz = (X - np.array(p["means"])) / np.array(p["stds"])
    score = _score(_risk_coordinates(Xz, p), scenario, strength)

    df = _assemble(p, X, score, dist, distance_noise, target_prev, rng)
    df.attrs.update(plane=plane, scenario=scenario, strength=strength,
                    feature_model=feature_model, jitter=jitter,
                    signal_signs=p["signal_signs"],
                    distance_noise=distance_noise,
                    true_prevalence=round(df["z_true"].mean(), 4),
                    obs_prevalence=round(df[p["target"]].mean(), 4))
    return df


def make_fused_dataset(scenario="linear", wise_signal=0.0, n=None, strength=2.5,
                       distance_noise=False, prevalence=None, seed=0,
                       feature_model="bootstrap", jitter=0.03):
    """Joint X-ray + WISE catalog as a DataFrame (4 feature columns).

    The true boundary is built from the X-ray features (via `scenario`) PLUS a
    term on the part of the WISE colours ORTHOGONAL to the X-ray features,
    scaled by `wise_signal`:
      wise_signal = 0   -> "redundant": WISE mirrors X-ray, adds nothing.
      wise_signal > 0   -> "added-signal": WISE carries unique information.
    The controllable version of the fusion question (does WISE help?).
    """
    f = FUSED
    n = n or f["n"]
    target_prev = prevalence if prevalence is not None else f["prevalence"]
    rng = np.random.default_rng(seed)

    means, stds = np.array(f["means"]), np.array(f["stds"])
    X, dist = _sample_features("fused", n, rng, feature_model, jitter)
    Z = (X - means) / stds
    Zrisk = _risk_coordinates(Z, f)

    xray_shape = _score(Zrisk[:, :2], scenario, strength=1.0)
    # WISE residual orthogonal to the X-ray features (least squares in-sample),
    # so an X-ray-only model genuinely cannot access it.
    A = np.column_stack([np.ones(n), Z[:, 0], Z[:, 1]])
    resid = np.empty((n, 2))
    for j in (2, 3):
        beta, *_ = np.linalg.lstsq(A, Z[:, j], rcond=None)
        resid[:, j - 2] = Z[:, j] - A @ beta
    rstd = resid / (resid.std(0) + 1e-9)
    wise_term = (rstd * np.array(f["signal_signs"][2:])).sum(axis=1) / np.sqrt(2)
    score = strength * (xray_shape + wise_signal * wise_term)

    df = _assemble(f, X, score, dist, distance_noise, target_prev, rng)
    df.attrs.update(kind="fused", scenario=scenario, wise_signal=wise_signal,
                    strength=strength, feature_model=feature_model,
                    jitter=jitter,
                    signal_signs=f["signal_signs"],
                    true_prevalence=round(df["z_true"].mean(), 4),
                    obs_prevalence=round(df[f["target"]].mean(), 4))
    return df


if __name__ == "__main__":
    print("Synthetic dataset summaries (one draw each):")
    for plane in ("xray", "wise"):
        tgt = PLANES[plane]["target"]
        for sc in ("linear", "wedge", "box", "interaction", "blob"):
            df = make_dataset(plane=plane, scenario=sc, seed=0)
            print(f"  {plane:4s} {sc:11s}  n={len(df):5d}  "
                  f"model={df.attrs['feature_model']}  cols={list(df.columns)[:2]}"
                  f"  obs_prev={df.attrs['obs_prevalence']:.3f}")
        dn = make_dataset(plane=plane, scenario="wedge", distance_noise=True, seed=0)
        hidden = int(dn["z_true"].sum() - dn[tgt].sum())
        print(f"  {plane:4s} wedge+dist   true_prev={dn.attrs['true_prevalence']:.3f}"
              f"  obs_prev={dn.attrs['obs_prevalence']:.3f}  "
              f"(distance noise hides {hidden} of {int(dn['z_true'].sum())} true masers)")

    print("\n  fused (X-ray + WISE):")
    for ws in (0.0, 1.5):
        df = make_fused_dataset(scenario="linear", wise_signal=ws, seed=0)
        print(f"    wise_signal={ws}: cols={list(df.columns)[:4]}  "
              f"obs_prev={df.attrs['obs_prevalence']:.3f}")
