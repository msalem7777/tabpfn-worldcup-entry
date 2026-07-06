"""
01_dixon_coles_ratings.py
─────────────────────────────────────────────────────────────────────────────
Fits a time-decayed Dixon–Coles (1997) model on the martj42 international
results history and exports opponent-adjusted team ratings.

Model
─────
Goals for the home side i vs away side j (independent Poisson with a
low-score dependence correction):

    log lambda_home = mu + home_adv * (1 - neutral) + attack_i + defense_j
    log lambda_away = mu +                            attack_j + defense_i

    P(x, y) = tau(x, y; lambda_h, lambda_a, rho) * Pois(x; lambda_h) * Pois(y; lambda_a)

tau adjusts only the 0-0 / 1-0 / 0-1 / 1-1 cells (Dixon & Coles eq. 4.3).
Each match's log-likelihood is weighted by exp(-XI * days_ago / 365.25),
so XI = ln(2) / half_life_years.

Identification: sum(attack) = 0 and sum(defense) = 0 (enforced by centering
inside the objective).

Sign conventions in the export
──────────────────────────────
    attack_rating  = attack_i          (higher = better attack)
    defense_rating = -defense_i        (higher = better defense; defense_i is
                                        "goals conceded inflation" in the model)

These are z-scale log-rate parameters, not goals-per-match, so they replace —
not mix with — the old script-02 ratings. Script 03 consumes them through
team_ratings.csv unchanged.

Outputs
───────
  data/interim/team_ratings.csv          (team, attack_rating, defense_rating,
                                          composite_overall_rating)
  data/interim/dixon_coles_params.csv    (mu, home_adv, rho, xi, ref_date,
                                          n_matches, train_start)

The module also exposes dc_match_probabilities(...) for direct H/D/A
probabilities of any fixture — usable as the reference model in calibration.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent
RAW_DIR     = ROOT_DIR / "data" / "raw"
INTERIM_DIR = ROOT_DIR / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV       = RAW_DIR / "results.csv"          # martj42 file already downloaded by 03
TEAM_RATINGS_CSV  = INTERIM_DIR / "team_ratings.csv"
DC_PARAMS_CSV     = INTERIM_DIR / "dixon_coles_params.csv"

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_START      = pd.Timestamp("2018-01-01")  # matches before this are ignored
REFERENCE_DATE   = None                        # None → max date in data; decay is measured back from here
HALF_LIFE_YEARS  = 1.5                         # weight halves every 18 months
MAX_GOALS        = 10                          # scoreline grid truncation for probabilities
MIN_TEAM_MATCHES = 8                           # teams with fewer weighted appearances are pooled less reliably; reported, not dropped
L2_PENALTY       = 1e-3                        # ridge on attack/defense params: shrinks sparse-schedule teams toward average

# Name harmonization: Sofascore display names → martj42 names (mirror of script 03)
OUR_TO_MARTJ42 = {
    "Republic of South Africa":     "South Africa",
    "USA":                          "United States",
    "Cote d'Ivoire":                "Ivory Coast",
    "Cabo Verde":                   "Cape Verde",
    "Democratic Republic of Congo": "DR Congo",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_matches(results_csv: Path = RESULTS_CSV,
                 train_start: pd.Timestamp = TRAIN_START) -> pd.DataFrame:
    """Load played martj42 matches from train_start onward."""
    df = pd.read_csv(results_csv)
    df["date"] = pd.to_datetime(df["date"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)

    df = df[
        df["home_score"].notna()
        & df["away_score"].notna()
        & (df["date"] >= train_start)
    ].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Dixon–Coles likelihood
# ─────────────────────────────────────────────────────────────────────────────

def _tau_log(x: np.ndarray, y: np.ndarray,
             lam_h: np.ndarray, lam_a: np.ndarray, rho: float) -> np.ndarray:
    """log tau(x, y) for the four low-score cells; 0 elsewhere."""
    tau = np.ones_like(lam_h)
    m00 = (x == 0) & (y == 0)
    m10 = (x == 1) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho
    tau[m10] = 1.0 + lam_a[m10] * rho
    tau[m01] = 1.0 + lam_h[m01] * rho
    tau[m11] = 1.0 - rho
    # Guard: tau must stay positive for a valid likelihood.
    return np.log(np.clip(tau, 1e-10, None))


class DixonColesModel:
    """Time-decayed Dixon–Coles fit on international results."""

    def __init__(self, half_life_years: float = HALF_LIFE_YEARS,
                 l2_penalty: float = L2_PENALTY):
        self.half_life_years = half_life_years
        self.xi = np.log(2.0) / half_life_years
        self.l2_penalty = l2_penalty
        self.teams_: list[str] | None = None
        self.attack_: dict[str, float] | None = None
        self.defense_: dict[str, float] | None = None
        self.mu_: float | None = None
        self.home_adv_: float | None = None
        self.rho_: float | None = None
        self.ref_date_: pd.Timestamp | None = None
        self.n_matches_: int | None = None

    # ── Fitting ──────────────────────────────────────────────────────────────
    def fit(self, matches: pd.DataFrame,
            reference_date: pd.Timestamp | None = None) -> "DixonColesModel":
        df = matches.reset_index(drop=True)
        self.ref_date_ = (
            pd.Timestamp(reference_date) if reference_date is not None
            else df["date"].max()
        )
        self.n_matches_ = len(df)

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.teams_ = teams
        t_index = {t: k for k, t in enumerate(teams)}
        n = len(teams)

        hi = df["home_team"].map(t_index).to_numpy()
        ai = df["away_team"].map(t_index).to_numpy()
        x = df["home_score"].to_numpy(dtype=float)
        y = df["away_score"].to_numpy(dtype=float)
        not_neutral = (1 - df["neutral"].to_numpy()).astype(float)

        days_ago = (self.ref_date_ - df["date"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-self.xi * days_ago / 365.25)

        # Parameter vector: [attack(n), defense(n), mu, home_adv, rho]
        def unpack(theta):
            att = theta[:n]
            dfn = theta[n:2 * n]
            att = att - att.mean()   # centering enforces identification
            dfn = dfn - dfn.mean()
            mu, home_adv, rho = theta[2 * n], theta[2 * n + 1], theta[2 * n + 2]
            return att, dfn, mu, home_adv, rho

        def neg_log_lik(theta):
            att, dfn, mu, home_adv, rho = unpack(theta)
            log_lh = mu + home_adv * not_neutral + att[hi] + dfn[ai]
            log_la = mu + att[ai] + dfn[hi]
            lam_h = np.exp(np.clip(log_lh, -10, 4))
            lam_a = np.exp(np.clip(log_la, -10, 4))

            ll = (
                x * np.log(lam_h) - lam_h
                + y * np.log(lam_a) - lam_a
                + _tau_log(x, y, lam_h, lam_a, rho)
            )
            penalty = self.l2_penalty * (np.sum(att ** 2) + np.sum(dfn ** 2))
            return -(w * ll).sum() / w.sum() + penalty

        theta0 = np.zeros(2 * n + 3)
        theta0[2 * n] = np.log(max(x.mean(), 0.1))   # mu init at log mean goals
        theta0[2 * n + 1] = 0.25                     # home_adv init (log-rate)
        theta0[2 * n + 2] = -0.05                    # rho init

        res = minimize(
            neg_log_lik, theta0, method="L-BFGS-B",
            bounds=[(None, None)] * (2 * n) + [(None, None), (None, None), (-0.9, 0.9)],
            options={"maxiter": 2000, "maxfun": 200000},
        )
        if not res.success:
            print(f"  WARNING: optimizer did not report convergence: {res.message}")

        att, dfn, mu, home_adv, rho = unpack(res.x)
        self.attack_ = dict(zip(teams, att))
        self.defense_ = dict(zip(teams, dfn))
        self.mu_, self.home_adv_, self.rho_ = float(mu), float(home_adv), float(rho)
        return self

    # ── Prediction ──────────────────────────────────────────────────────────
    def _rates(self, home_team: str, away_team: str, neutral: bool) -> tuple[float, float]:
        att, dfn = self.attack_, self.defense_
        # Unseen teams get league-average parameters (0 by centering).
        ah, dh = att.get(home_team, 0.0), dfn.get(home_team, 0.0)
        aa, da = att.get(away_team, 0.0), dfn.get(away_team, 0.0)
        adv = 0.0 if neutral else self.home_adv_
        lam_h = np.exp(self.mu_ + adv + ah + da)
        lam_a = np.exp(self.mu_ + aa + dh)
        return float(lam_h), float(lam_a)

    def scoreline_matrix(self, home_team: str, away_team: str,
                         neutral: bool = True, max_goals: int = MAX_GOALS) -> np.ndarray:
        """Joint P(home goals = x, away goals = y) grid, tau-corrected, renormalized."""
        lam_h, lam_a = self._rates(home_team, away_team, neutral)
        gx = poisson.pmf(np.arange(max_goals + 1), lam_h)
        gy = poisson.pmf(np.arange(max_goals + 1), lam_a)
        grid = np.outer(gx, gy)

        rho = self.rho_
        grid[0, 0] *= max(1.0 - lam_h * lam_a * rho, 1e-10)
        grid[1, 0] *= max(1.0 + lam_a * rho, 1e-10)
        grid[0, 1] *= max(1.0 + lam_h * rho, 1e-10)
        grid[1, 1] *= max(1.0 - rho, 1e-10)
        return grid / grid.sum()

    def match_probabilities(self, home_team: str, away_team: str,
                            neutral: bool = True) -> dict[str, float]:
        """90-minute H/D/A probabilities."""
        grid = self.scoreline_matrix(home_team, away_team, neutral)
        p_home = float(np.tril(grid, -1).sum())   # x > y
        p_draw = float(np.trace(grid))
        p_away = float(np.triu(grid, 1).sum())    # y > x
        return {"p_home_win": p_home, "p_draw": p_draw, "p_away_win": p_away}


def dc_match_probabilities(model: DixonColesModel, home_team: str, away_team: str,
                           neutral: bool = True) -> dict[str, float]:
    """Convenience wrapper matching the calibration script's expected callable."""
    return model.match_probabilities(home_team, away_team, neutral=neutral)


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_ratings(model: DixonColesModel, matches: pd.DataFrame) -> pd.DataFrame:
    """Write team_ratings.csv in the schema script 03 already consumes."""
    counts = pd.concat([matches["home_team"], matches["away_team"]]).value_counts()

    rows = []
    for team in model.teams_:
        attack = model.attack_[team]
        defense = -model.defense_[team]  # flip sign: higher = better defense
        rows.append({
            "team": team,
            "attack_rating": attack,
            "defense_rating": defense,
            "composite_overall_rating": attack + defense,
            "n_matches_in_window": int(counts.get(team, 0)),
            "sparse_schedule_flag": int(counts.get(team, 0) < MIN_TEAM_MATCHES),
        })

    out = (
        pd.DataFrame(rows)
        .sort_values("composite_overall_rating", ascending=False)
        .reset_index(drop=True)
    )
    out.to_csv(TEAM_RATINGS_CSV, index=False)

    pd.DataFrame([{
        "mu": model.mu_,
        "home_adv": model.home_adv_,
        "rho": model.rho_,
        "xi": model.xi,
        "half_life_years": model.half_life_years,
        "ref_date": model.ref_date_.date(),
        "train_start": TRAIN_START.date(),
        "n_matches": model.n_matches_,
    }]).to_csv(DC_PARAMS_CSV, index=False)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Half-life validation (rolling log-loss; run only when tuning)
# ─────────────────────────────────────────────────────────────────────────────

def rolling_log_loss(matches: pd.DataFrame, half_life_years: float,
                     validation_years: list[int]) -> float:
    """
    Chronological validation: for each year Y, fit on matches before Y-01-01
    and score H/D/A log-loss on matches in year Y. Mirrors the rolling design
    of script 03's residual folds.
    """
    losses, n_total = 0.0, 0
    for year in validation_years:
        cut = pd.Timestamp(year=year, month=1, day=1)
        train = matches[matches["date"] < cut]
        valid = matches[(matches["date"] >= cut) & (matches["date"] < cut + pd.DateOffset(years=1))]
        if len(train) < 500 or valid.empty:
            continue

        model = DixonColesModel(half_life_years=half_life_years).fit(
            train, reference_date=cut
        )
        for r in valid.itertuples():
            p = model.match_probabilities(r.home_team, r.away_team,
                                          neutral=bool(r.neutral))
            if r.home_score > r.away_score:
                chosen = p["p_home_win"]
            elif r.home_score < r.away_score:
                chosen = p["p_away_win"]
            else:
                chosen = p["p_draw"]
            losses += -np.log(max(chosen, 1e-12))
            n_total += 1

    return losses / max(n_total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Dixon–Coles team ratings")
    print("=" * 65)

    matches = load_matches()
    print(f"  Matches: {len(matches):,}  ({matches['date'].min().date()} → {matches['date'].max().date()})")
    print(f"  Teams  : {len(set(matches['home_team']) | set(matches['away_team']))}")
    print(f"  Half-life: {HALF_LIFE_YEARS} years  (xi = {np.log(2)/HALF_LIFE_YEARS:.4f}/yr)")

    model = DixonColesModel().fit(matches, reference_date=REFERENCE_DATE)
    print(f"\n  mu = {model.mu_:.4f}   home_adv = {model.home_adv_:.4f} (log-rate)   rho = {model.rho_:.4f}")

    ratings = export_ratings(model, matches)
    print(f"\n  Saved → {TEAM_RATINGS_CSV}")
    print(f"  Saved → {DC_PARAMS_CSV}")
    print("\n  Top 20 by composite:")
    print(ratings.head(20).to_string(index=False))
