"""
replicate kling/liebman/katz (2007) mto adult indices w/ causalpfn

process
- load the public-use adult cell-level .dta (this is the one that actually has mn_f_* adult indices)
- loads the pseudo-individual file; it uses ps_f_* names note: the adult mn_f_* stuff
  we care about for the kling table is cell-level
- for each outcome + contrast:
    * build T for the requested contrast (exp vs control, s8 vs control, or pooled treated vs control)
    * keep just the relevant cols
    * drop missing outcome rows
    * pick baseline covariates (mn_x_*, ps_x_*, x_f_site_*)
    * median-impute numeric covs
    * standardize X
    * fit causalpfn ate + cate
    * write to csv

notes / small annoyances
- i hard-pin threads to 1 because torch + blas likes to randomly nuke kernels on my machine otherwise
- causalpfn sometimes goes leaf=0 on tiny patches so the weak learner patch below makes it more stable

outputs
- results/kling_replication_causalpfn.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# stability stuff (worked, kernel no longer crashing)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch

# again so torch doesnt crash kernel
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from causalpfn import ATEEstimator, CATEEstimator
import causalpfn.causal_estimator as ce
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor


# causalpfn going leaf = 0 on small patches, this fixes it
def _safe_train_weak_learner(self, X, t, y):
    self.t_transformer = OneHotEncoder(sparse_output=False, categories="auto", drop="first")
    T = self.t_transformer.fit_transform(t.reshape(-1, 1))
    self._d_t = (T.shape[1],)
    feat_arr = np.concatenate((X, 1 - np.sum(T, axis=1).reshape(-1, 1), T), axis=1)
    min_leaf = max(1, int(X.shape[0] / 100))
    self.stratifier = GradientBoostingRegressor(
        n_estimators=100, max_depth=6, min_samples_leaf=min_leaf, random_state=111,
    )
    self.stratifier.fit(feat_arr, y)


ce.CausalEstimator._train_weak_learner = _safe_train_weak_learner


# paths
PATH_CELLS = "/Users/richardguo/csc494-spatialpfn/economics/mto/mto_aer_ad_puf_cells_20131025.dta"
PATH_IND   = "/Users/richardguo/csc494-spatialpfn/economics/mto/mto_aer_ad_puf_pseudo_20131025.dta"
OUTDIR     = Path("/Users/richardguo/csc494-spatialpfn/economics/mto/results")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTDIR / "kling_replication_causalpfn.csv"
DEVICE  = "cpu"

# adult indices (cell-level naming)
Y_CELL = ["mn_f_all_idx_fix_z_ad", "mn_f_ec_idx_z_ad", "mn_f_mh_idx_z_ad", "mn_f_ph_idx_fix_z_ad"]

# pseudo-individual PUF uses ps_f_* names for the same domains
Y_PSEUDO = ["ps_f_all_idx_fix_z_ad", "ps_f_ec_idx_z_ad", "ps_f_mh_idx_z_ad", "ps_f_ph_idx_fix_z_ad"]

# ITT contrasts to mirror paper (+ pooled treated as a bonus)
CONTRASTS = ["exp_vs_control", "s8_vs_control", "any_voucher_vs_control"]

# map y column -> domain label used for published table lookup
DOMAIN_BY_Y = {
    "mn_f_all_idx_fix_z_ad": "overall", "mn_f_ec_idx_z_ad": "economic",
    "mn_f_mh_idx_z_ad": "mental_health", "mn_f_ph_idx_fix_z_ad": "physical_health",
    "ps_f_all_idx_fix_z_ad": "overall", "ps_f_ec_idx_z_ad": "economic",
    "ps_f_mh_idx_z_ad": "mental_health", "ps_f_ph_idx_fix_z_ad": "physical_health",
}

# ITT estimates from Kling table 2 (effect sizes in SD units; SE in parentheses)
PUBLISHED_ITT = {
    ("exp_vs_control", "economic"): (0.017, 0.031), ("exp_vs_control", "physical_health"): (0.012, 0.024),
    ("exp_vs_control", "mental_health"): (0.079, 0.030), ("exp_vs_control", "overall"): (0.036, 0.020),
    ("s8_vs_control", "economic"): (0.037, 0.033), ("s8_vs_control", "physical_health"): (0.019, 0.026),
    ("s8_vs_control", "mental_health"): (0.029, 0.033), ("s8_vs_control", "overall"): (0.028, 0.022),
}


# helpers
def set_all_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_binary01(series):
    # stata sometimes gives weird categorical-ish encodings; this tries to robustly coerce "0/1"
    if str(series.dtype).startswith("category"):
        series = series.astype(str)
    if np.issubdtype(series.dtype, np.number):
        return pd.to_numeric(series, errors="coerce").astype(float)

    s_str = series.astype(str).str.strip()
    leading = s_str.str.extract(r"^([01])\s*=")[0]   # e.g. "1 = Experimental"
    leading2 = s_str.where(s_str.isin(["0", "1"]), np.nan)
    out = leading.fillna(leading2)
    return pd.to_numeric(out, errors="coerce").astype(float)


def build_t_and_keep(df, contrast):
    exp = to_binary01(df["ra_grp_exp"])
    s8  = to_binary01(df["ra_grp_s8"])
    ctl = to_binary01(df["ra_grp_control"])

    if contrast == "any_voucher_vs_control":
        keep = ((exp == 1) | (s8 == 1) | (ctl == 1))
        t = ((exp == 1) | (s8 == 1)).astype(float)
    elif contrast == "exp_vs_control":
        keep = ((exp == 1) | (ctl == 1))
        t = (exp == 1).astype(float)
    elif contrast == "s8_vs_control":
        keep = ((s8 == 1) | (ctl == 1))
        t = (s8 == 1).astype(float)
    else:
        print('failed at t')

    return t.to_numpy().astype(np.float32), keep.to_numpy().astype(bool)


def pick_x_baseline(df, y_col):
    # baseline covariates: mn_x_*, ps_x_*, x_f_site_* (+ ra_site if present)
    drop = set(["ra_group", y_col])
    x_cols = []
    for c in df.columns:
        if c in drop:
            continue
        if c.startswith("mn_x_") or c.startswith("ps_x_") or c.startswith("x_f_site_"):
            x_cols.append(c)
    if "ra_site" in df.columns and "ra_site" not in drop:
        x_cols.append("ra_site")

    X_df = df[x_cols].copy()

    cat_cols = [c for c in X_df.columns if X_df[c].dtype == "object" or str(X_df[c].dtype).startswith("category")]
    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

    for c in X_df.columns:
        if X_df[c].isna().any():
            if np.issubdtype(X_df[c].dtype, np.number):
                X_df[c] = X_df[c].fillna(X_df[c].median())
            else:
                X_df[c] = X_df[c].fillna("MISSING")

    return X_df.astype(np.float32).to_numpy()


def standardize(X, eps=1e-8):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd


def ate_diff_in_means(Y, T):
    t1 = (T == 1); t0 = (T == 0)
    if t1.sum() == 0 or t0.sum() == 0:
        return float("nan")
    return float(Y[t1].mean() - Y[t0].mean())


# main
def main():
    df_cells  = pd.read_stata(PATH_CELLS)
    df_pseudo = pd.read_stata(PATH_IND)

    needed_cols = {"ra_grp_exp", "ra_grp_s8", "ra_grp_control"}
    missing = needed_cols - set(df_cells.columns)
    if missing:
        print(f"cells missing required columns: {missing}")

    rows = []

    def run_level(df, y_cols, level):
        for y_col in y_cols:
            if y_col not in df.columns:
                continue

            for contrast in CONTRASTS:
                T_all, keep = build_t_and_keep(df, contrast)
                dfk = df.loc[keep].copy().reset_index(drop=True)
                T = T_all[keep]

                # drop missing outcomes
                dfk = dfk.dropna(subset=[y_col]).reset_index(drop=True)
                T = T[dfk.index.to_numpy()]
                Y = dfk[y_col].astype(np.float32).to_numpy()
                if len(Y) == 0:
                    continue

                X = standardize(pick_x_baseline(dfk, y_col=y_col))

                set_all_seeds(9)
                ate_model  = ATEEstimator(device=DEVICE)
                cate_model = CATEEstimator(device=DEVICE)
                ate_model.fit(X, T, Y); cate_model.fit(X, T, Y)

                ate_causalpfn = float(ate_model.estimate_ate())
                cate_hat = np.asarray(cate_model.estimate_cate(X), dtype=np.float32)

                domain = DOMAIN_BY_Y.get(y_col)
                published = PUBLISHED_ITT.get((contrast, domain))

                rows.append({
                    "level": level, "contrast": contrast, "y_col": y_col, "domain": domain,
                    "n_used": int(X.shape[0]), "p": int(X.shape[1]),
                    "ate_observed": ate_diff_in_means(Y, T), "ate_causalpfn": ate_causalpfn,
                    "cate_mean": float(cate_hat.mean()), "cate_std": float(cate_hat.std()),
                    "itt_published": float(published[0]) if published else np.nan,
                    "itt_published_se": float(published[1]) if published else np.nan,
                    "device": DEVICE, "seed": 9,
                })

    run_level(df_cells, Y_CELL, "cell")
    run_level(df_pseudo, Y_PSEUDO, "pseudo")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    print("\nsaved:", OUT_CSV)
    if len(out_df):
        print("\nsummary (per outcome/contrast/level):")
        print(
            out_df[
                [
                    "level","contrast","y_col","domain","n_used","p",
                    "ate_observed","ate_causalpfn","cate_mean","cate_std",
                    "itt_published","itt_published_se",
                ]
            ].to_string(index=False)
        )
    else:
        print("failed")


if __name__ == "__main__":
    main()
