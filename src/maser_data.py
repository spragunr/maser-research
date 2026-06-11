"""Shared data loading for the maser ML project.

Every analysis script should get its data through this module so the whole
team works from the same cleaning and join decisions (rationale in
docs/RESEARCH_PLAN.md and docs/PREREGISTRATION.md):

  load_xray_sample()    -> 641 galaxies, X-ray + mid-IR (Kuo et al. 2020 sample,
                           MRK739E duplicate dropped)
  load_wise()           -> WISE photometry + maser labels for the full GBT
                           survey (Anca's AllWISE 6-arcsec cross-match), after
                           removing maser contamination and repeat pointings
                           from the nonmaser file (see its docstring)
  load_xray_with_wise() -> the X-ray sample with WISE columns attached by
                           coordinate match (602 of 641 rows match)

Run this module directly for a summary of all three:
  python -m src.maser_data
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
NA_VALUES = ["---", "-99.0", "-99"]
WISE_BANDS = ["w1", "w2", "w3", "w4"]


def _read_table(path):
    header = open(path).readline().lstrip("#").split()
    return pd.read_csv(path, sep=r"\s+", skiprows=1, names=header,
                       na_values=NA_VALUES)


def load_xray_sample():
    """X-ray sample: Table 1 + Table 2 merged on GBT name, one row per galaxy.

    MRK739E appears in both the Swift and XMM catalogs; the XMM row is dropped
    (Swift has catalog priority) because a duplicated galaxy leaks across CV
    folds. Note the two rows differ in capitalization and slightly in
    coordinates, so the drop keys on case-insensitive name + ref_xray.
    """
    t1 = _read_table(DATA_RAW / "Table_1_data.txt")
    t2 = _read_table(DATA_RAW / "table_2_data.txt")
    df = t1.merge(t2.drop(columns=["ra_mcp_1", "dec_mcp_1", "name_NED_2"]),
                  left_on="name_mcp", right_on="name_mcp_2_1", how="inner")

    upper = df["name_mcp"].str.upper()
    dup = upper.duplicated(keep=False) & (df["ref_xray"] == "xmm")
    df = df[~dup].reset_index(drop=True)

    # Derived: the paper's N_H proxy. Collinear with Lob and L_12um_bestfit_1 --
    # a linear model gets at most two of the three.
    df["Lob_minus_L12"] = df["Lob"] - df["L_12um_bestfit_1"]

    # Convenience binary targets (see docs/PREREGISTRATION.md for primary targets)
    df["is_maser"] = (df["maser"] > 0).astype(int)             # types 1-4
    df["is_megamaser_plus"] = (df["maser"] >= 2).astype(int)   # types 2-4
    df["is_disk"] = (df["maser"] == 4).astype(int)             # confirmed disk
    return df


def _dedupe_groups(ra, dec, radius_arcsec):
    """Group rows lying within radius of each other (union-find on sky pairs).

    Returns an integer group id per row; rows alone in their group are unique.
    """
    ra_r, dec_r = np.radians(ra), np.radians(dec)
    cos = (np.sin(dec_r[:, None]) * np.sin(dec_r[None, :])
           + np.cos(dec_r[:, None]) * np.cos(dec_r[None, :])
           * np.cos(ra_r[:, None] - ra_r[None, :]))
    sep = np.degrees(np.arccos(np.clip(cos, -1, 1))) * 3600
    parent = np.arange(len(ra))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in np.argwhere(np.triu(sep < radius_arcsec, k=1)):
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(ra))])


def load_wise(clean=True, radius_arcsec=6.0):
    """Full-survey AllWISE sample: WISE photometry + maser labels, one row per
    galaxy after cleaning.

    The raw nonmaser CSV is per-OBSERVATION, not per-galaxy, and is
    contaminated: ~84 known masers (incl. NGC4258) appear in it under alternate
    names -- calibration / offset / repeat GBT pointings (e.g. CALNGC3079,
    NGC5765B-OFF1). Same-galaxy rows can carry survey coordinates scattered by
    >6 arcsec, but they resolve to the SAME AllWISE counterpart, so cleaning
    keys on the WISE position (ra/dec), not the survey position (ra_01/dec_01).
    With clean=True (default, keep it on):
      * every nonmaser row whose WISE counterpart lies within 1 arcsec of a
        maser-file row's counterpart is dropped (label contamination);
      * remaining nonmaser rows are grouped by WISE counterpart and each group
        keeps its deepest observation (lowest rms1_01), since min-rms best
        describes how well the galaxy was searched.

    Column notes (harmonized across the two source CSVs):
      source        galaxy name (GBT survey naming, underscores)
      velocity      recession velocity km/s (vsys_01 / velo_01); NaN if flagged
      is_maser      1 / 0 (which file the row came from)
      maser_lum     H2O luminosity, L_sun; NaN where flagged (-999 / -10:
                    meaning of -10 unconfirmed -- ask Anca before using)
      maser_class   class_01 with -999 -> NaN; code key unconfirmed (ask Anca)
      w1w2, w2w3, w1w4   colors from wXmpro
      w3_lowsnr, w4_lowsnr   True where band SNR < 3: the magnitude is
                    effectively an upper limit. Asymmetric between classes
                    (masers are IR-bright), so an artifact risk -- prefer
                    W1-W2 / W2-W3 features, use these as quality flags.
    GBT sensitivity metadata (tsys_01, int_01, rms1_01, vlo1_01, vhi1_01,
    date_obs_01) exists for nonmasers only; NaN on maser rows.
    """
    m = pd.read_csv(DATA_RAW / "allwise_masers_cleaned_6arcs.csv")
    n = pd.read_csv(DATA_RAW / "allwise_nonmasers_cleaned_6arcs.csv")
    m = m.rename(columns={"sourcename_01": "source", "vsys_01": "velocity"})
    n = n.rename(columns={"source_01": "source", "velo_01": "velocity"})

    if clean:
        _, sep = crossmatch(n["ra"], n["dec"], m["ra"], m["dec"])
        n = n[sep > 1.0].reset_index(drop=True)
        group = _dedupe_groups(n["ra"].values, n["dec"].values, 1.0)
        n = (n.assign(_group=group)
               .sort_values("rms1_01")
               .drop_duplicates("_group")
               .drop(columns="_group")
               .sort_index()
               .reset_index(drop=True))

    m["is_maser"], n["is_maser"] = 1, 0
    df = pd.concat([m, n], ignore_index=True, sort=False)

    df.loc[df["velocity"] == -999, "velocity"] = np.nan
    df["maser_lum"] = df["lum_01"].where(df["lum_01"] >= 0)
    df["maser_class"] = df["class_01"].where(df["class_01"] != -999)

    df["w1w2"] = df["w1mpro"] - df["w2mpro"]
    df["w2w3"] = df["w2mpro"] - df["w3mpro"]
    df["w1w4"] = df["w1mpro"] - df["w4mpro"]
    df["w3_lowsnr"] = df["w3snr"] < 3
    df["w4_lowsnr"] = df["w4snr"] < 3
    return df


def crossmatch(ra1, dec1, ra2, dec2):
    """Nearest neighbour on the sky for each (ra1, dec1) among (ra2, dec2).

    All inputs in degrees. Returns (index_into_2, separation_arcsec), one entry
    per row of set 1. Brute force; fine at this scale (642 x 4650). Does not
    enforce one-to-one matching -- check for double assignments if that matters.
    """
    ra1, dec1 = np.radians(np.asarray(ra1)), np.radians(np.asarray(dec1))
    ra2, dec2 = np.radians(np.asarray(ra2)), np.radians(np.asarray(dec2))
    cossep = (np.sin(dec1[:, None]) * np.sin(dec2[None, :])
              + np.cos(dec1[:, None]) * np.cos(dec2[None, :])
              * np.cos(ra1[:, None] - ra2[None, :]))
    sep = np.degrees(np.arccos(np.clip(cossep, -1, 1))) * 3600
    idx = sep.argmin(axis=1)
    return idx, sep[np.arange(len(idx)), idx]


# WISE columns worth carrying onto the X-ray sample
_WISE_KEEP = (["source", "is_maser", "velocity", "w1w2", "w2w3", "w1w4",
               "w3_lowsnr", "w4_lowsnr"]
              + [f"{b}mpro" for b in WISE_BANDS]
              + [f"{b}snr" for b in WISE_BANDS])


def load_xray_with_wise(radius_arcsec=6.0):
    """X-ray sample with WISE columns attached by coordinate match.

    Adds (prefixed wise_): the columns in _WISE_KEEP, plus
      wise_sep_arcsec   match separation (NaN where no match within radius)
      label_disagree    True where the WISE dataset's any-maser label
                        contradicts the X-ray table's. 6 such galaxies --
                        unresolved provenance question (see docs/RESEARCH_PLAN.md);
                        exclude or investigate before using merged labels.
    Unmatched rows (39 at the default radius) keep their X-ray columns with
    WISE columns NaN.
    """
    xr = load_xray_sample()
    wise = load_wise()
    # Match against the AllWISE counterpart position (ra/dec) -- it is the
    # stable per-galaxy key; the survey positions (ra_01/dec_01) scatter.
    # Six masers stay unmatched at any sane radius (M33, Mrk1066, NGC1386,
    # IC342, NGC5128, NGC5256): big, nearby, extended galaxies where the GBT
    # pointing and the WISE source sit 8"-900" apart -- and where WISE
    # point-source photometry is unreliable anyway. Left unmatched on purpose.
    idx, sep = crossmatch(xr["ra_mcp_1"], xr["dec_mcp_1"],
                          wise["ra"], wise["dec"])
    ok = sep <= radius_arcsec

    attach = wise.iloc[idx][_WISE_KEEP].reset_index(drop=True)
    attach = attach.add_prefix("wise_").where(pd.Series(ok, name="ok"), np.nan)
    out = pd.concat([xr, attach], axis=1)
    out["wise_sep_arcsec"] = np.where(ok, sep, np.nan)
    out["label_disagree"] = ok & (out["wise_is_maser"] != out["is_maser"])
    return out


if __name__ == "__main__":
    xr = load_xray_sample()
    print(f"X-ray sample: {len(xr)} galaxies, "
          f"masers={xr.is_maser.sum()} (mega+={xr.is_megamaser_plus.sum()}, "
          f"disk={xr.is_disk.sum()})")

    wise = load_wise()
    print(f"WISE sample: {len(wise)} galaxies, masers={wise.is_maser.sum()}, "
          f"megamasers (lum>=10): {(wise.maser_lum >= 10).sum()}, "
          f"lum unknown: {(wise.is_maser == 1).sum() - wise.maser_lum.notna().sum()}")

    xw = load_xray_with_wise()
    matched = xw.wise_sep_arcsec.notna()
    print(f"X-ray+WISE: {matched.sum()}/{len(xw)} matched within 6 arcsec "
          f"({(matched & (xw.is_maser == 1)).sum()}/{xw.is_maser.sum()} masers)")
    double = xw.loc[matched, "wise_source"].duplicated().sum()
    print(f"WISE rows assigned to two X-ray rows: {double}")
    dis = xw[xw.label_disagree]
    print(f"\nLabel disagreements ({len(dis)}) -- X-ray maser type vs WISE label:")
    print(dis[["name_mcp", "maser", "wise_is_maser", "wise_source",
               "wise_sep_arcsec"]].to_string(index=False))
