"""
05_anchor_gated_calibration.py
─────────────────────────────────────────────────────────────────────────────
Anchor-constrained, gated post-hoc calibration of frozen TabPFN forecasts.

Calibrated distribution (gated geometric pool with temperature):

    log q*(k|x)  ∝  [ alpha(x) · log q0(k|x) + (1 − alpha(x)) · log qs(k|x) ] / T

    alpha(x) = sigmoid( gamma' phi(x) )

    phi(x)   = [ 1,
                 |elo_diff| (standardized),
                 KL(qs || q0),
                 excess favorite uncertainty U0 − Us,
                 EXTRA_GATE_COVARIATES... (standardized) ]
               where U(q) = 1 − q(reference favorite)

Objective (fit on chronologically out-of-fold rows):

    L(gamma, T) =  Σ_i w_i · [ −log q*(y_i | x_i) ]                     (outcome NLL)
                 + LAMBDA_ANCHOR · Σ_j a_j · hinge(τ_j − q*(fav_j|x_j))²    (anchors)

Anchors are one-sided lower bounds on the calibrated favorite probability,
weighted by a_j. The favorite is defined by the reference model (side with the
higher win probability). λ is chosen from a chronological-holdout λ path
scored on realized-outcome log-loss only — anchors influence fitting but are
never the yardstick.

Reference model (REFERENCE_MODEL switch):
  "dixon_coles"   — low-information structural arm (team identity + venue)
  "ordered_logit" — full-covariate parametric arm: proportional-odds model
                    m = beta'x with draw-band cutpoints (reference_models.py).
                    Continuous, interpretable, ordinal-structure-respecting.

Nesting properties:
  gamma = (g0, 0, ..., 0), T = 1     → global geometric pool (script 04)
  gamma0 → +inf, T = 1               → raw neural output (do nothing)
  LAMBDA_ANCHOR = 0                  → outcome-only calibration

Inputs
──────
  data/output/rolling_v2_oof_predictions_for_calibration.csv    (from script 03)
  data/interim/final_v3_training_engineered_features.csv        (from script 03;
        supplies reference + gate covariates, joined by date/home/away)
  data/raw/results.csv                                          (martj42 history;
        used by the Dixon–Coles arm)
  data/interim/expert_anchors.csv                               (optional)

Anchor CSV schema (data/interim/expert_anchors.csv):
  date, home_team, away_team, min_favorite_prob, anchor_weight
  - date matched on calendar day; team names in martj42 convention
  - min_favorite_prob in (0, 1); anchor_weight > 0 (1.0 = normal)
  - rows that match no OOF row are reported and skipped

Outputs
───────
  data/output/anchor_calibration_reliability_by_bin.csv    (Step-1 diagnostic)
  data/output/anchor_calibration_fit_summary.csv
  data/output/anchor_calibration_lambda_path.csv
  data/output/calibrated_submission_anchor_gated_r16.csv       (if PREDICTIONS_CSV exists)
  data/output/anchor_calibration_row_diagnostics.csv

Nothing here calls TabPFN. All reference fits are seconds each.
Depends on 01_dixon_coles_ratings.py and reference_models.py in the same
directory.
"""

import warnings
warnings.filterwarnings("ignore")

from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_dc = import_module("01_dixon_coles_ratings")
DixonColesModel = _dc.DixonColesModel
load_matches = _dc.load_matches

from reference_models import (
    OrderedLogitReference,
    build_ordered_logit_reference_probabilities,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent
RAW_DIR     = ROOT_DIR / "data" / "raw"
INTERIM_DIR = ROOT_DIR / "data" / "interim"
OUTPUT_DIR  = ROOT_DIR / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OOF_PREDICTIONS_CSV   = OUTPUT_DIR / "rolling_v2_oof_predictions_for_calibration.csv"
FINAL_V3_FEATURES_CSV = INTERIM_DIR / "final_v3_training_engineered_features.csv"
ANCHORS_CSV           = INTERIM_DIR / "expert_anchors.csv"          # optional

# Fixture application inputs (row-aligned pair exported by script 03).
PREDICTIONS_CSV         = OUTPUT_DIR / "round_of_16_candidate_probabilities_rolling_residual_v3.csv"
PREDICTION_FEATURES_CSV = OUTPUT_DIR / "round_of_16_candidate_features_rolling_residual_v3.csv"
DEFAULT_NEUTRAL         = 1

RELIABILITY_CSV           = OUTPUT_DIR / "anchor_calibration_reliability_by_bin.csv"
FIT_SUMMARY_CSV           = OUTPUT_DIR / "anchor_calibration_fit_summary.csv"
LAMBDA_PATH_CSV           = OUTPUT_DIR / "anchor_calibration_lambda_path.csv"
CALIBRATED_SUBMISSION_CSV = OUTPUT_DIR / "calibrated_submission_anchor_gated_r16.csv"
ROW_DIAGNOSTICS_CSV       = OUTPUT_DIR / "anchor_calibration_row_diagnostics.csv"

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_START         = pd.Timestamp("2014-01-01")
DC_HALF_LIFE_YEARS  = 1.5
CAL_HALF_LIFE_YEARS = 4.0        # recency weighting of OOF rows; None disables

# Reference model: "dixon_coles" or "ordered_logit".
REFERENCE_MODEL = "ordered_logit"

# Covariates for the ordered-logit reference: the full continuous set.
# Standardized inside the model; coefficients directly comparable.
REFERENCE_COVARIATES = [
    "elo_diff", "form5_diff", "form10_diff", "gd10_diff",
    "home_winrate", "away_winrate",
    "home_gf5", "away_gf5", "home_ga5", "away_ga5",
    "home_rest", "away_rest",
    "h2h_home_winrate", "h2h_draw_rate", "h2h_gd",
    "home_attack_rating", "home_defense_rating",
    "away_attack_rating", "away_defense_rating",
    "neutral", "importance",
]
OL_HALF_LIFE_YEARS = 1.5

# Gate covariates beyond the built-ins [1, |elo_diff|, KL, U0−Us]:
# columns named here are standardized (fit-population stats) and appended.
EXTRA_GATE_COVARIATES = ["form5_diff", "h2h_n"]

# Fit population: restrict to rows resembling the deployment target.
FIT_SUBSET_MIN_IMPORTANCE = 45.0        # None disables
FIT_SUBSET_NEUTRAL_ONLY   = True

# Ridge penalty on the gate SLOPES (intercept excluded): keeps gamma
# identifiable on the alpha≈1 plateau, where the likelihood is flat in
# gamma0 → ∞ and unpenalized slopes inflate into hard 0/1 gates that carve
# individual rows. The global-pool solution (slopes = 0) is unpenalized.
RIDGE_GAMMA = 1e-2

# Anchor penalty strength for the final fit. Inspect the λ path first
# (anchor_calibration_lambda_path.csv), then set and re-run.
LAMBDA_ANCHOR = 0.0
LAMBDA_GRID   = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]

# Chronological holdout for the λ path: fit on fold years < HOLDOUT_START_YEAR,
# evaluate outcome log-loss on fold years >= HOLDOUT_START_YEAR.
HOLDOUT_START_YEAR = 2024

# Reliability diagnostic bins on reference favorite probability.
RELIABILITY_BIN_EDGES = [1/3, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.00]

EPS         = 1e-12
PROB_COLS   = ["p_home_win", "p_draw", "p_away_win"]
CLASS_ORDER = ["home_win", "draw", "away_win"]

MIN_TRAIN_ROWS_REFERENCE = 500


# ─────────────────────────────────────────────────────────────────────────────
# Data assembly
# ─────────────────────────────────────────────────────────────────────────────

def required_covariate_columns() -> list[str]:
    return sorted(set(
        ["elo_diff"]
        + REFERENCE_COVARIATES
        + EXTRA_GATE_COVARIATES
    ))


def load_oof_with_features() -> pd.DataFrame:
    """OOF v2 predictions + reference/gate covariates by (date, home, away)."""
    oof = pd.read_csv(OOF_PREDICTIONS_CSV)
    oof["date"] = pd.to_datetime(oof["date"]).dt.normalize()

    feats = pd.read_csv(FINAL_V3_FEATURES_CSV)
    feats["date"] = pd.to_datetime(feats["date"]).dt.normalize()

    needed = required_covariate_columns()
    missing_cols = sorted(set(needed) - set(feats.columns))
    if missing_cols:
        raise ValueError(f"Feature table lacks required covariates: {missing_cols}")

    join_cols = ["date", "home_team", "away_team"]
    # neutral/importance already live on the OOF export; avoid duplicate columns.
    take = [c for c in needed if c not in oof.columns]
    feats = feats[join_cols + take].drop_duplicates(subset=join_cols)

    merged = oof.merge(feats, on=join_cols, how="left")

    incomplete = merged[needed].isna().any(axis=1)
    if incomplete.any():
        print(f"  WARNING: {int(incomplete.sum())} OOF rows lack covariates "
              "after join; dropped (conservative fit population).")
        merged = merged.loc[~incomplete].reset_index(drop=True)

    merged["neutral"] = merged["neutral"].fillna(1).astype(int)
    return merged


def build_dc_reference_probabilities(oof: pd.DataFrame,
                                     matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Chronological DC refits per fold year; reference never sees its fold."""
    ref = np.full((len(oof), 3), np.nan)

    for year in sorted(oof["fold_year"].unique()):
        cut = pd.Timestamp(year=int(year), month=1, day=1)
        train = matches[matches["date"] < cut]
        if len(train) < MIN_TRAIN_ROWS_REFERENCE:
            continue
        model = DixonColesModel(half_life_years=DC_HALF_LIFE_YEARS).fit(
            train, reference_date=cut
        )
        idx = np.where(oof["fold_year"].to_numpy() == year)[0]
        for i in idx:
            row = oof.iloc[i]
            p = model.match_probabilities(
                row["home_team"], row["away_team"], neutral=bool(row["neutral"])
            )
            ref[i] = [p["p_home_win"], p["p_draw"], p["p_away_win"]]
        print(f"  DC reference fold {int(year)}: fit on {len(train):,}, "
              f"scored {len(idx):,}")

    keep = np.isfinite(ref).all(axis=1)
    if (~keep).sum():
        print(f"  Dropped {(~keep).sum()} rows without a DC reference.")
    return ref, keep


# ─────────────────────────────────────────────────────────────────────────────
# Calibration family
# ─────────────────────────────────────────────────────────────────────────────

def build_phi(neural: np.ndarray, ref: np.ndarray,
              elo_diff: np.ndarray,
              extra: np.ndarray | None,
              scales: dict | None = None) -> tuple[np.ndarray, dict, np.ndarray]:
    """
    Gate covariates phi(x) = [1, |elo_diff|/s, KL(qs||q0), U0 − Us, extras/s_k].

    `extra` is an (n, m) array of EXTRA_GATE_COVARIATES values (or None).
    `scales` (fit-population statistics) must be reused at application time:
        {"elo_scale": float, "extra_mu": array, "extra_sd": array}

    Returns (phi, scales_used, favorite_index).
    """
    abs_elo = np.abs(elo_diff)

    if scales is None:
        scales = {"elo_scale": float(abs_elo.std()) or 1.0}
        if extra is not None:
            mu = extra.mean(axis=0)
            sd = extra.std(axis=0)
            sd[sd == 0] = 1.0
            scales["extra_mu"], scales["extra_sd"] = mu, sd

    q0 = np.clip(neural, EPS, 1.0)
    qs = np.clip(ref, EPS, 1.0)

    kl = (qs * (np.log(qs) - np.log(q0))).sum(axis=1)

    # Reference favorite: whichever WIN side qs rates higher (draw excluded).
    fav_idx = np.where(qs[:, 0] >= qs[:, 2], 0, 2)
    u0 = 1.0 - q0[np.arange(len(q0)), fav_idx]
    us = 1.0 - qs[np.arange(len(qs)), fav_idx]

    columns = [
        np.ones(len(q0)),
        abs_elo / scales["elo_scale"],
        kl,
        u0 - us,
    ]
    if extra is not None:
        columns.append((extra - scales["extra_mu"]) / scales["extra_sd"])

    phi = np.column_stack(columns)
    return phi, scales, fav_idx


def gated_pool(neural: np.ndarray, ref: np.ndarray, phi: np.ndarray,
               gamma: np.ndarray, temperature: float) -> np.ndarray:
    """q*(k|x) for the gated geometric pool."""
    alpha = 1.0 / (1.0 + np.exp(-(phi @ gamma)))
    log_p = (
        alpha[:, None] * np.log(np.clip(neural, EPS, 1.0))
        + (1.0 - alpha)[:, None] * np.log(np.clip(ref, EPS, 1.0))
    ) / temperature
    log_p -= log_p.max(axis=1, keepdims=True)
    p = np.exp(log_p)
    return p / p.sum(axis=1, keepdims=True)


def gate_values(phi: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(phi @ gamma)))


# ─────────────────────────────────────────────────────────────────────────────
# Objective and fitting
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(neural, ref, phi, y_index, weights,
                   anchor_rows, anchor_tau, anchor_w, fav_idx, lam,
                   ridge_gamma=RIDGE_GAMMA):
    """f(theta) = weighted outcome NLL + ridge on gate slopes
    + lam * anchor hinge² penalty. theta = [gamma(d), log T]."""
    n = len(neural)
    d = phi.shape[1]
    w_norm = weights / weights.sum()

    def objective(theta):
        gamma = theta[:d]
        temperature = np.exp(theta[d])
        pooled = gated_pool(neural, ref, phi, gamma, temperature)

        chosen = np.clip(pooled[np.arange(n), y_index], EPS, 1.0)
        nll = float(-(w_norm * np.log(chosen)).sum())

        ridge = ridge_gamma * float(np.sum(gamma[1:] ** 2))   # slopes only

        penalty = 0.0
        if lam > 0 and len(anchor_rows):
            p_fav = pooled[anchor_rows, fav_idx[anchor_rows]]
            gap = np.maximum(0.0, anchor_tau - p_fav)
            penalty = float((anchor_w * gap ** 2).sum() / max(anchor_w.sum(), 1.0))

        return nll + ridge + lam * penalty

    return objective


def fit_calibration(neural, ref, phi, y_index, weights,
                    anchor_rows, anchor_tau, anchor_w, fav_idx,
                    lam, label="fit"):
    d = phi.shape[1]
    obj = make_objective(neural, ref, phi, y_index, weights,
                         anchor_rows, anchor_tau, anchor_w, fav_idx, lam)
    # Warm start at the do-nothing corner: alpha ≈ sigmoid(3) ≈ 0.95, T = 1.
    theta0 = np.zeros(d + 1)
    theta0[0] = 3.0
    res = minimize(obj, theta0, method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-9,
                            "maxiter": 20000, "maxfev": 30000})
    gamma = res.x[:d]
    temperature = float(np.exp(res.x[d]))
    print(f"  [{label}] gamma = {np.round(gamma, 4)}  T = {temperature:.4f}  "
          f"objective = {res.fun:.6f}  converged = {res.success}")
    return gamma, temperature


def outcome_log_loss(neural, ref, phi, gamma, temperature, y_index,
                     weights=None) -> float:
    pooled = gated_pool(neural, ref, phi, gamma, temperature)
    chosen = np.clip(pooled[np.arange(len(pooled)), y_index], EPS, 1.0)
    if weights is None:
        return float(-np.log(chosen).mean())
    return float(-(weights * np.log(chosen)).sum() / weights.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Anchors
# ─────────────────────────────────────────────────────────────────────────────

def load_anchors(oof: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match anchor rows to OOF rows by (date, home, away).
    Returns (row_indices, tau, weights); empty arrays when no anchor file.
    """
    empty = (np.array([], dtype=int), np.array([]), np.array([]))
    if not ANCHORS_CSV.exists():
        print("  No anchors file found; fitting with outcome supervision only.")
        return empty

    anchors = pd.read_csv(ANCHORS_CSV)
    anchors["date"] = pd.to_datetime(anchors["date"]).dt.normalize()

    key = oof.reset_index()[["index", "date", "home_team", "away_team"]]
    merged = anchors.merge(key, on=["date", "home_team", "away_team"], how="left")

    unmatched = merged["index"].isna()
    if unmatched.any():
        print(f"  WARNING: {int(unmatched.sum())} anchor rows matched no OOF row; skipped:")
        print(merged.loc[unmatched, ["date", "home_team", "away_team"]]
              .to_string(index=False))
    merged = merged[~unmatched]

    tau = merged["min_favorite_prob"].to_numpy(dtype=float)
    if ((tau <= 0) | (tau >= 1)).any():
        raise ValueError("min_favorite_prob must lie strictly in (0, 1).")

    w = merged.get("anchor_weight", pd.Series(1.0, index=merged.index))
    print(f"  Anchors matched to OOF rows: {len(merged)}")
    return (merged["index"].to_numpy(dtype=int), tau,
            w.to_numpy(dtype=float))


# ─────────────────────────────────────────────────────────────────────────────
# Step-1 diagnostic: reliability by reference-favorite-strength bin
# ─────────────────────────────────────────────────────────────────────────────

def reliability_by_bin(neural, ref, y_index, fav_idx) -> pd.DataFrame:
    """
    Within bins of the reference favorite probability, compare the neural
    model's stated favorite probability against the realized favorite win
    rate (with binomial standard errors). Also reports draw calibration.
    neural_gap > 0 beyond ~2×SE in strong-favorite bins = the timidity motif.
    """
    qs_fav = ref[np.arange(len(ref)), fav_idx]
    q0_fav = neural[np.arange(len(neural)), fav_idx]
    fav_won = (y_index == fav_idx).astype(float)
    drew = (y_index == 1).astype(float)
    q0_draw = neural[:, 1]

    bins = pd.cut(qs_fav, RELIABILITY_BIN_EDGES, include_lowest=True)
    rows = []
    for b, idx in pd.Series(np.arange(len(qs_fav))).groupby(bins).groups.items():
        idx = np.asarray(idx)
        if len(idx) == 0:
            continue
        n = len(idx)
        rate = fav_won[idx].mean()
        se = np.sqrt(max(rate * (1 - rate), EPS) / n)
        rows.append({
            "reference_fav_bin": str(b),
            "n": n,
            "mean_reference_fav_prob": qs_fav[idx].mean(),
            "mean_neural_fav_prob": q0_fav[idx].mean(),
            "realized_fav_win_rate": rate,
            "binomial_se": se,
            "neural_gap": rate - q0_fav[idx].mean(),
            "mean_neural_draw_prob": q0_draw[idx].mean(),
            "realized_draw_rate": drew[idx].mean(),
        })
    table = pd.DataFrame(rows)
    table.to_csv(RELIABILITY_CSV, index=False)

    print("\n  Reliability by reference-favorite-strength bin")
    print(table.round(4).to_string(index=False))
    print(f"  Saved → {RELIABILITY_CSV}")
    return table


# ─────────────────────────────────────────────────────────────────────────────
# λ path: does anchor supervision help held-out outcome log-loss?
# ─────────────────────────────────────────────────────────────────────────────

def lambda_path(neural, ref, phi, y_index, weights, fold_years,
                anchor_rows, anchor_tau, anchor_w, fav_idx) -> pd.DataFrame:
    train_mask = fold_years < HOLDOUT_START_YEAR
    test_mask = ~train_mask
    if train_mask.sum() < 500 or test_mask.sum() < 200:
        print("  λ path skipped: insufficient rows on one side of the holdout split.")
        return pd.DataFrame()

    tr = np.where(train_mask)[0]
    anchor_in_train = np.isin(anchor_rows, tr)
    a_rows = anchor_rows[anchor_in_train]
    a_tau = anchor_tau[anchor_in_train]
    a_w = anchor_w[anchor_in_train]
    pos = {g: k for k, g in enumerate(tr)}
    a_rows_local = np.array([pos[r] for r in a_rows], dtype=int)

    te = np.where(test_mask)[0]
    ll_raw_holdout = float(-np.log(np.clip(
        neural[te][np.arange(len(te)), y_index[te]], EPS, 1.0)).mean())

    rows = []
    for lam in LAMBDA_GRID:
        gamma, temperature = fit_calibration(
            neural[tr], ref[tr], phi[tr], y_index[tr], weights[tr],
            a_rows_local, a_tau, a_w, fav_idx[tr],
            lam, label=f"lambda={lam}",
        )
        ll = outcome_log_loss(neural[te], ref[te], phi[te],
                              gamma, temperature, y_index[te])
        record = {"lambda": lam, "holdout_log_loss": ll,
                  "holdout_log_loss_raw_neural": ll_raw_holdout,
                  "temperature": temperature,
                  "n_train": len(tr), "n_holdout": len(te),
                  "n_anchors_in_train": len(a_rows_local)}
        for k, g in enumerate(gamma):
            record[f"gamma_{k}"] = g
        rows.append(record)

    path = pd.DataFrame(rows)
    path.to_csv(LAMBDA_PATH_CSV, index=False)
    print(f"\n  λ path (holdout = fold years ≥ {HOLDOUT_START_YEAR}):")
    print(path.round(5).to_string(index=False))
    print(f"  Saved → {LAMBDA_PATH_CSV}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Application to new predictions
# ─────────────────────────────────────────────────────────────────────────────

def apply_to_predictions(params: dict, scales: dict, gamma: np.ndarray,
                         reference_predict_fn) -> None:
    """
    reference_predict_fn(preds_df, feats_df) -> (n, 3) reference probabilities
    in CLASS_ORDER. Supplied by main according to REFERENCE_MODEL.
    """
    if not PREDICTIONS_CSV.exists():
        print(f"  Predictions file not found: {PREDICTIONS_CSV} — skipping application.")
        return

    preds = pd.read_csv(PREDICTIONS_CSV)
    feats = pd.read_csv(PREDICTION_FEATURES_CSV)
    if len(preds) != len(feats):
        raise ValueError("Prediction and feature CSVs are not row-aligned.")

    needed = required_covariate_columns()
    missing_cols = sorted(set(needed) - set(feats.columns))
    if missing_cols:
        raise ValueError(f"Fixture feature CSV lacks covariates: {missing_cols}")

    neutral = preds["neutral"] if "neutral" in preds.columns else DEFAULT_NEUTRAL
    preds = preds.assign(neutral=neutral)

    neural = preds[PROB_COLS].to_numpy(dtype=float)
    neural = neural / neural.sum(axis=1, keepdims=True)

    ref = reference_predict_fn(preds, feats)

    extra = (feats[EXTRA_GATE_COVARIATES].to_numpy(dtype=float)
             if EXTRA_GATE_COVARIATES else None)
    phi, _, fav_idx = build_phi(
        neural, ref, feats["elo_diff"].to_numpy(dtype=float),
        extra, scales=scales,
    )
    pooled = gated_pool(neural, ref, phi, gamma, params["temperature"])
    alpha = gate_values(phi, gamma)

    diag = preds.copy()
    for k, col in enumerate(PROB_COLS):
        diag[f"raw_{col}"] = neural[:, k]
        diag[f"reference_{col}"] = ref[:, k]
        diag[f"calibrated_{col}"] = pooled[:, k]
    diag["alpha_gate"] = alpha
    diag["reference_favorite"] = np.where(fav_idx == 0,
                                          diag["home_team"], diag["away_team"])
    diag["reference_model"] = REFERENCE_MODEL
    diag.to_csv(ROW_DIAGNOSTICS_CSV, index=False)

    submission = preds[["date", "home_team", "away_team"]].copy()
    submission[PROB_COLS] = pooled
    if not np.allclose(submission[PROB_COLS].sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Calibrated rows do not sum to 1.")
    submission.to_csv(CALIBRATED_SUBMISSION_CSV, index=False)

    print(f"  Saved → {CALIBRATED_SUBMISSION_CSV}")
    print(f"  Saved → {ROW_DIAGNOSTICS_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print(f"Anchor-constrained gated calibration  (reference = {REFERENCE_MODEL})")
    print("=" * 65)

    oof = load_oof_with_features()
    print(f"  OOF rows with covariates: {len(oof):,}")

    # ── Reference probabilities (chronological per-fold refits) ──────────────
    if REFERENCE_MODEL == "ordered_logit":
        history = pd.read_csv(FINAL_V3_FEATURES_CSV)
        ref, keep, ref_models = build_ordered_logit_reference_probabilities(
            oof, history, REFERENCE_COVARIATES,
            half_life_years=OL_HALF_LIFE_YEARS,
            min_train_rows=MIN_TRAIN_ROWS_REFERENCE,
        )
        latest_ref_model = ref_models[max(ref_models)]
        print("\n  Ordered-logit coefficients (latest fold refit; standardized):")
        print(latest_ref_model.coefficient_table().round(4).to_string(index=False))
    elif REFERENCE_MODEL == "dixon_coles":
        matches = load_matches(train_start=TRAIN_START)
        ref, keep = build_dc_reference_probabilities(oof, matches)
        latest_ref_model = None
    else:
        raise ValueError(f"Unknown REFERENCE_MODEL: {REFERENCE_MODEL!r}")

    oof = oof.loc[keep].reset_index(drop=True)
    ref = ref[keep]

    # ── Fit-population subset (tournament-like rows) ─────────────────────────
    subset = np.ones(len(oof), dtype=bool)
    if FIT_SUBSET_MIN_IMPORTANCE is not None:
        subset &= oof["importance"].to_numpy() >= FIT_SUBSET_MIN_IMPORTANCE
    if FIT_SUBSET_NEUTRAL_ONLY:
        subset &= oof["neutral"].to_numpy() == 1
    oof = oof.loc[subset].reset_index(drop=True)
    ref = ref[subset]
    print(f"\n  Fit-population rows: {len(oof):,}")

    neural = oof[PROB_COLS].to_numpy(dtype=float)
    neural = neural / neural.sum(axis=1, keepdims=True)
    y_index = oof["outcome"].map(
        {c: k for k, c in enumerate(CLASS_ORDER)}
    ).to_numpy(dtype=int)

    if CAL_HALF_LIFE_YEARS is not None:
        yrs_ago = (oof["date"].max() - oof["date"]).dt.days.to_numpy() / 365.25
        weights = np.exp(-np.log(2.0) * yrs_ago / CAL_HALF_LIFE_YEARS)
    else:
        weights = np.ones(len(oof))

    extra = (oof[EXTRA_GATE_COVARIATES].to_numpy(dtype=float)
             if EXTRA_GATE_COVARIATES else None)
    phi, scales, fav_idx = build_phi(
        neural, ref, oof["elo_diff"].to_numpy(dtype=float), extra
    )
    gate_terms = (["intercept", "abs_elo_diff", "kl_ref_neural", "excess_unc"]
                  + list(EXTRA_GATE_COVARIATES))
    print(f"  Gate covariates phi(x): {gate_terms}")

    # ── Step 1: conditional diagnostic ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 1 — reliability by favorite-strength bin (raw neural)")
    print("=" * 65)
    reliability_by_bin(neural, ref, y_index, fav_idx)

    # ── Anchors ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Anchors")
    print("=" * 65)
    anchor_rows, anchor_tau, anchor_w = load_anchors(oof)

    # ── λ path on chronological holdout ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("λ path — does anchor supervision help held-out log-loss?")
    print("=" * 65)
    lambda_path(neural, ref, phi, y_index, weights,
                oof["fold_year"].to_numpy(), anchor_rows, anchor_tau,
                anchor_w, fav_idx)

    # ── Final fit on all fit-population rows at LAMBDA_ANCHOR ────────────────
    print("\n" + "=" * 65)
    print(f"Final fit (LAMBDA_ANCHOR = {LAMBDA_ANCHOR})")
    print("=" * 65)
    gamma, temperature = fit_calibration(
        neural, ref, phi, y_index, weights,
        anchor_rows, anchor_tau, anchor_w, fav_idx,
        LAMBDA_ANCHOR, label="final",
    )

    ll_raw = float(-np.log(np.clip(
        neural[np.arange(len(neural)), y_index], EPS, 1.0)).mean())
    ll_ref = float(-np.log(np.clip(
        ref[np.arange(len(ref)), y_index], EPS, 1.0)).mean())
    ll_cal = outcome_log_loss(neural, ref, phi, gamma, temperature, y_index)
    alpha = gate_values(phi, gamma)
    print(f"\n  unweighted log-loss raw        : {ll_raw:.4f}")
    print(f"  unweighted log-loss reference  : {ll_ref:.4f}")
    print(f"  unweighted log-loss calibrated : {ll_cal:.4f}")
    print(f"  gate alpha(x): min={alpha.min():.3f} "
          f"median={np.median(alpha):.3f} max={alpha.max():.3f}")

    params = {
        "reference_model": REFERENCE_MODEL,
        "temperature": temperature,
        "lambda_anchor": LAMBDA_ANCHOR,
        "n_fit_rows": len(oof), "n_anchors": len(anchor_rows),
        "log_loss_raw": ll_raw, "log_loss_reference": ll_ref,
        "log_loss_calibrated": ll_cal,
        "gate_terms": " | ".join(gate_terms),
    }
    for k, g in enumerate(gamma):
        params[f"gamma_{k}"] = g
    pd.DataFrame([params]).to_csv(FIT_SUMMARY_CSV, index=False)
    print(f"  Saved → {FIT_SUMMARY_CSV}")

    # ── Application to next-round predictions ────────────────────────────────
    print("\n" + "=" * 65)
    print("Applying to prediction CSV")
    print("=" * 65)

    if REFERENCE_MODEL == "ordered_logit":
        def reference_predict_fn(preds_df, feats_df):
            return latest_ref_model.predict_proba_frame(feats_df)
    else:
        current_dc = DixonColesModel(half_life_years=DC_HALF_LIFE_YEARS).fit(
            load_matches(train_start=TRAIN_START)
        )
        def reference_predict_fn(preds_df, feats_df):
            return np.array([
                list(current_dc.match_probabilities(
                    r.home_team, r.away_team, neutral=bool(r.neutral)).values())
                for r in preds_df.itertuples()
            ])

    apply_to_predictions(params, scales, gamma, reference_predict_fn)
