"""
02_dixon_coles_rating_history.py
─────────────────────────────────────────────────────────────────────────────
Time-varying Dixon–Coles ratings via checkpointed refits.

Motivation
──────────
A single Dixon–Coles fit produces ratings *as of* one reference date. Joining
that snapshot onto historical training rows leaks future team strength into
the past. This module instead refits the model at a grid of checkpoint dates,
each fit using only matches strictly before the checkpoint (decay measured
from the checkpoint), and stores the full rating history.

Script 03 then joins ratings as-of each match date with a backward asof-merge,
so a 2015 row sees only 2015 knowledge and a 2026 fixture sees the latest fit.

Outputs
───────
  data/interim/team_rating_history.csv
      as_of_date, team, attack_rating, defense_rating, n_matches_seen

  data/interim/team_ratings.csv   (unchanged schema: the LATEST checkpoint,
      so existing consumers keep working)

Depends on 01_dixon_coles_ratings.py in the same directory.
"""

import warnings
warnings.filterwarnings("ignore")

from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

_dc = import_module("01_dixon_coles_ratings")
DixonColesModel = _dc.DixonColesModel
load_matches = _dc.load_matches
export_ratings = _dc.export_ratings

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent
INTERIM_DIR = ROOT_DIR / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

RATING_HISTORY_CSV = INTERIM_DIR / "team_rating_history.csv"

# ── Config ────────────────────────────────────────────────────────────────────
HISTORY_TRAIN_START   = pd.Timestamp("2010-01-01")  # matches available to the earliest checkpoints
FIRST_CHECKPOINT      = pd.Timestamp("2014-01-01")  # aligns with TRAIN_START in script 03
CHECKPOINT_FREQ       = "QS"                        # quarterly (quarter-start); "MS" for monthly
FINAL_CHECKPOINT      = None                        # None → day after last played match
MIN_TRAIN_MATCHES     = 500                         # skip checkpoints with too little history
DC_HALF_LIFE_YEARS    = 0.25


def build_checkpoint_grid(matches: pd.DataFrame) -> list[pd.Timestamp]:
    """Quarterly checkpoints from FIRST_CHECKPOINT through the end of data."""
    last = (
        pd.Timestamp(FINAL_CHECKPOINT) if FINAL_CHECKPOINT is not None
        else matches["date"].max() + pd.Timedelta(days=1)
    )
    grid = list(pd.date_range(FIRST_CHECKPOINT, last, freq=CHECKPOINT_FREQ))
    # Always include the very latest state so current fixtures get fresh ratings.
    if not grid or grid[-1] < last:
        grid.append(last)
    return grid


def fit_rating_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Refit Dixon–Coles at each checkpoint; return the long rating table."""
    checkpoints = build_checkpoint_grid(matches)
    records = []

    for cp in checkpoints:
        train = matches[matches["date"] < cp]
        if len(train) < MIN_TRAIN_MATCHES:
            print(f"  {cp.date()}  skipped (only {len(train)} matches before checkpoint)")
            continue

        model = DixonColesModel(half_life_years=DC_HALF_LIFE_YEARS).fit(
            train, reference_date=cp
        )
        counts = pd.concat([train["home_team"], train["away_team"]]).value_counts()

        for team in model.teams_:
            records.append({
                "as_of_date":     cp,
                "team":           team,
                "attack_rating":  model.attack_[team],
                "defense_rating": -model.defense_[team],   # higher = better, matching 01
                "n_matches_seen": int(counts.get(team, 0)),
            })

        print(f"  {cp.date()}  fit on {len(train):,} matches, {len(model.teams_)} teams")

    history = pd.DataFrame(records).sort_values(["team", "as_of_date"]).reset_index(drop=True)
    history.to_csv(RATING_HISTORY_CSV, index=False)
    print(f"\n  Saved → {RATING_HISTORY_CSV}  ({len(history):,} rows, "
          f"{history['as_of_date'].nunique()} checkpoints)")
    return history


if __name__ == "__main__":
    print("=" * 65)
    print("Dixon–Coles time-varying rating history")
    print("=" * 65)

    matches = load_matches(train_start=HISTORY_TRAIN_START)
    print(f"  Matches: {len(matches):,}  ({matches['date'].min().date()} → "
          f"{matches['date'].max().date()})\n")

    history = fit_rating_history(matches)

    # Also refresh the flat latest-snapshot file so existing consumers
    # (and quick inspection) stay in sync with the newest checkpoint.
    latest_model = DixonColesModel(half_life_years=DC_HALF_LIFE_YEARS).fit(matches)
    export_ratings(latest_model, matches)
    print("  Refreshed latest-snapshot team_ratings.csv / dixon_coles_params.csv")
