"""
reference_models.py
─────────────────────────────────────────────────────────────────────────────
Parametric, interpretable reference models for post-hoc calibration.

OrderedLogitReference
─────────────────────
Proportional-odds (ordered logit) model on the full covariate vector.
This is the latent-margin construction generalized beyond a single Elo axis:

    m_i = beta' x_i                       (latent match margin, continuous in x)
    P(away_win) = sigmoid(c1 - m_i)
    P(draw)     = sigmoid(c2 - m_i) - sigmoid(c1 - m_i)
    P(home_win) = 1 - sigmoid(c2 - m_i)

with cutpoints c1 < c2 forming the draw band. Every beta is a log-odds effect
on the latent margin; (c2 - c1) is the width of the draw region. The model is
continuous, parametric, respects the ordinal structure away < draw < home,
and is estimable in seconds by weighted MLE.

Fitting conventions (mirroring the Dixon–Coles reference in the pipeline):
- Covariates are standardized with training-fold statistics (stored, reused
  at prediction time), so coefficients are directly comparable in magnitude.
- Optional exponential time-decay weights with a configurable half-life,
  measured back from a reference date.
- Cutpoints parameterized as (c1, log gap) to enforce c1 < c2 without
  constrained optimization.
- Small L2 penalty on beta for stability with correlated covariates.

Intended use: refit chronologically per fold year (fit strictly before the
fold, decay measured from the fold start), exactly like the DC reference,
so reference probabilities never see their own outcomes.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

EPS = 1e-12
CLASS_ORDER = ["home_win", "draw", "away_win"]   # column order used everywhere


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class OrderedLogitReference:
    """Weighted proportional-odds model over {away_win, draw, home_win}."""

    def __init__(self, covariates: list[str],
                 half_life_years: float | None = 1.5,
                 l2_penalty: float = 1e-3):
        self.covariates = list(covariates)
        self.half_life_years = half_life_years
        self.l2_penalty = l2_penalty
        self.beta_: np.ndarray | None = None
        self.c1_: float | None = None
        self.c2_: float | None = None
        self.mu_x_: np.ndarray | None = None
        self.sd_x_: np.ndarray | None = None
        self.ref_date_: pd.Timestamp | None = None
        self.n_fit_rows_: int | None = None

    # ── internals ────────────────────────────────────────────────────────────
    def _design(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.covariates].to_numpy(dtype=float)
        if not np.isfinite(X).all():
            raise ValueError("OrderedLogitReference received non-finite covariates.")
        return (X - self.mu_x_) / self.sd_x_

    @staticmethod
    def _class_probs(m: np.ndarray, c1: float, c2: float) -> np.ndarray:
        p_away = _sigmoid(c1 - m)
        p_away_or_draw = _sigmoid(c2 - m)
        p_draw = np.clip(p_away_or_draw - p_away, EPS, 1.0)
        p_home = np.clip(1.0 - p_away_or_draw, EPS, 1.0)
        p_away = np.clip(p_away, EPS, 1.0)
        probs = np.column_stack([p_home, p_draw, p_away])   # CLASS_ORDER
        return probs / probs.sum(axis=1, keepdims=True)

    # ── fitting ──────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, outcome_col: str = "outcome",
            date_col: str = "date",
            reference_date: pd.Timestamp | None = None) -> "OrderedLogitReference":
        """
        df must contain the covariate columns, an outcome column with values
        in CLASS_ORDER, and a date column when time decay is enabled.
        """
        work = df[df[outcome_col].isin(CLASS_ORDER)].copy()
        self.n_fit_rows_ = len(work)
        if self.n_fit_rows_ < 10 * (len(self.covariates) + 2):
            raise ValueError(
                f"OrderedLogitReference: {self.n_fit_rows_} rows is too few for "
                f"{len(self.covariates)} covariates."
            )

        X_raw = work[self.covariates].to_numpy(dtype=float)
        if not np.isfinite(X_raw).all():
            raise ValueError("Non-finite covariates in the fitting data.")
        self.mu_x_ = X_raw.mean(axis=0)
        self.sd_x_ = X_raw.std(axis=0)
        self.sd_x_[self.sd_x_ == 0] = 1.0
        X = (X_raw - self.mu_x_) / self.sd_x_

        y = work[outcome_col].map({c: k for k, c in enumerate(CLASS_ORDER)}) \
                .to_numpy(dtype=int)

        if self.half_life_years is not None:
            self.ref_date_ = (
                pd.Timestamp(reference_date) if reference_date is not None
                else pd.to_datetime(work[date_col]).max()
            )
            yrs = (self.ref_date_ - pd.to_datetime(work[date_col])) \
                      .dt.days.to_numpy() / 365.25
            w = np.exp(-np.log(2.0) * yrs / self.half_life_years)
        else:
            w = np.ones(len(work))
        w = w / w.sum()

        k = len(self.covariates)

        def unpack(theta):
            beta = theta[:k]
            c1 = theta[k]
            c2 = c1 + np.exp(theta[k + 1])     # enforces c1 < c2
            return beta, c1, c2

        def neg_log_lik(theta):
            beta, c1, c2 = unpack(theta)
            probs = self._class_probs(X @ beta, c1, c2)
            chosen = probs[np.arange(len(y)), y]
            nll = -(w * np.log(chosen)).sum()
            return nll + self.l2_penalty * np.sum(beta ** 2)

        theta0 = np.zeros(k + 2)
        theta0[k] = -0.6                         # symmetric-ish draw band init
        theta0[k + 1] = np.log(1.2)

        res = minimize(neg_log_lik, theta0, method="L-BFGS-B",
                       options={"maxiter": 3000})
        if not res.success:
            print(f"  OrderedLogitReference WARNING: {res.message}")

        self.beta_, self.c1_, self.c2_ = unpack(res.x)
        return self

    # ── prediction ───────────────────────────────────────────────────────────
    def predict_proba_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Returns an (n, 3) array in CLASS_ORDER = [home_win, draw, away_win]."""
        m = self._design(df) @ self.beta_
        return self._class_probs(m, self.c1_, self.c2_)

    def coefficient_table(self) -> pd.DataFrame:
        """Interpretable summary: standardized log-odds effects + draw band."""
        rows = [{"term": c, "beta_std": b}
                for c, b in zip(self.covariates, self.beta_)]
        rows.append({"term": "cutpoint_c1", "beta_std": self.c1_})
        rows.append({"term": "cutpoint_c2", "beta_std": self.c2_})
        rows.append({"term": "draw_band_width", "beta_std": self.c2_ - self.c1_})
        return pd.DataFrame(rows)


def build_ordered_logit_reference_probabilities(
        oof: pd.DataFrame,
        history: pd.DataFrame,
        covariates: list[str],
        half_life_years: float = 1.5,
        min_train_rows: int = 500,
        fold_col: str = "fold_year") -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Chronological refits per fold year, mirroring the DC reference builder:
    for each fold year Y, fit on history rows dated strictly before Y-01-01
    (decay measured from the fold start) and score the OOF rows in that fold.

    `history` must be the engineered feature table (played rows with outcome
    and the covariate columns). `oof` must carry the same covariate columns —
    join them from the feature table beforehand — plus fold_col.

    Returns (ref_probs, keep_mask, fitted_models_by_year).
    """
    ref = np.full((len(oof), 3), np.nan)
    models = {}

    history = history[history["outcome"].isin(CLASS_ORDER)].copy()
    history["date"] = pd.to_datetime(history["date"])

    for year in sorted(oof[fold_col].unique()):
        cut = pd.Timestamp(year=int(year), month=1, day=1)
        train = history[history["date"] < cut]
        train = train.dropna(subset=covariates)
        if len(train) < min_train_rows:
            continue

        model = OrderedLogitReference(
            covariates=covariates, half_life_years=half_life_years
        ).fit(train, reference_date=cut)
        models[int(year)] = model

        idx = np.where(oof[fold_col].to_numpy() == year)[0]
        fold_rows = oof.iloc[idx]
        ref[idx] = model.predict_proba_frame(fold_rows)
        print(f"  OL reference fold {int(year)}: fit on {len(train):,}, "
              f"scored {len(idx):,}")

    keep = np.isfinite(ref).all(axis=1)
    if (~keep).sum():
        print(f"  Dropped {(~keep).sum()} rows without an ordered-logit reference.")
    return ref, keep, models
