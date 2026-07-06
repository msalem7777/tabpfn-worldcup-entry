"""
predict_round16.py
─────────────────────────────────────────────────────────────────────────────
Fits v3 with rolling historical residual-bias features and writes Round-of-16 submission probabilities:

  v1 (Baseline) : ELO + form + H2H features, TabPFN
  v2            : v1 features + our attack/defense ratings
  v3            : v2 features + rolling historical residual bias against baseline

All three models are evaluated on the same rows (those where augmented
features are available) so comparisons are fair.

Evaluation: accuracy and log-loss on played WC group stage matches.

Rolling residual design
───────────────────────
Residual-bias features are generated out-of-sample through chronological
annual folds. For each fold, a baseline TabPFN model is fit on earlier matches
and predicts the held-out fold. Team-level residual-bias features for a match
are then the team's prior expanding mean residuals before that match. This
avoids filling residual-bias NaNs with zeros and gives the final v3 model a
non-constant historical residual context.
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product
from sklearn.metrics import accuracy_score, log_loss
from tabpfn_client import TabPFNClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "data" / "raw"
INTERIM_DIR = ROOT_DIR / "data" / "interim"
OUTPUT_DIR = ROOT_DIR / "data" / "output"

for _dir in (RAW_DIR, INTERIM_DIR, OUTPUT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RAW_URL     = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
DATA        = RAW_DIR / "results.csv"
RATINGS     = INTERIM_DIR / "team_ratings.csv"
CHARACTERISTICS = INTERIM_DIR / "team_characteristics.csv"

# Time-aware team-style eras.
#
# This file should contain one row per team/style era with:
#   team, style_period_start, style_period_end, keyword_triplet
#
# Example keyword_triplet:
#   compact|efficient|structured
#
# Important:
# - These are match-date covariates.
# - They must not be converted into static team-level ratings.
# - Missing coverage is encoded as an explicit "unknown" category, not dropped.
STYLE_ERAS = INTERIM_DIR / "team_style_keyword_eras_round32_time_aware.csv"

# Keep style keyword covariates disabled for the current calibration case study.
#
# Why:
# - The current post-hoc calibration experiment should isolate the effect of
#   transitive strength/reference-model calibration.
# - Style keywords can be added back later as neural-network covariates after
#   the Elo/reference-calibration mechanism is stable.
# - configure_style_features(USE_STYLE_KEYWORDS) below already centralizes the
#   feature-list update, so setting this to False removes the style columns from
#   AUGMENTED_V3_FEATURES consistently.
USE_STYLE_KEYWORDS = True

STYLE_UNKNOWN_TOKEN = "unknown"
STYLE_HOME_FEATURE = "home_style_triplet_code"
STYLE_AWAY_FEATURE = "away_style_triplet_code"

TRAIN_START = pd.Timestamp("2014-01-01")
MAX_TRAIN   = None  # None = use all complete pre-WC rows from TRAIN_START onward
HOME_ADV    = 65.0

# TabPFN client / thinking-mode settings.
#
# Quota design for this variant:
# - Rolling residual-bias generation uses chronological baseline fits.
# - Those rolling baseline fits are regular/client mode by default to avoid
#   spending many thinking-fit quota calls.
# - The final v3 model fit can use thinking mode.
# - Round-of-16 prediction uses one batched predict_proba call and does not
#   consume thinking fits.
ROLLING_RESIDUAL_BASELINE_THINKING_MODE = False
FINAL_V3_THINKING_MODE = False
FINAL_V3_THINKING_EFFORT = "high"
FINAL_V3_THINKING_METRIC = "log_loss"
FINAL_V3_THINKING_TIMEOUT_S = None  # set to e.g. 300 to cap optimization wall-clock time

# Rolling residual-bias settings.
# Annual folds keep the number of baseline fits manageable while still making
# residual predictions out-of-sample in calendar time.
ROLLING_RESIDUAL_FIRST_VALIDATION_YEAR = 2015
ROLLING_RESIDUAL_MIN_TRAIN_ROWS = 500
ROLLING_RESIDUAL_PREDICTIONS_CSV = OUTPUT_DIR / "rolling_residual_oof_predictions.csv"
ROLLING_RESIDUAL_LONG_CSV = OUTPUT_DIR / "rolling_team_residual_events.csv"
ROLLING_RESIDUAL_LATEST_CSV = OUTPUT_DIR / "rolling_team_residual_bias_latest.csv"

# Competition submission output.
#
# The competition upload for the current Round of 16 must have exactly these
# columns, with 90-minute match outcome probabilities, not knockout advancement
# probabilities:
#   date,home_team,away_team,p_home_win,p_draw,p_away_win
#
# If a Round-of-16 bracket slot is still unresolved in BRACKET_R16, fill this
# dictionary once the official team is known. The script will only write the
# upload-ready CSV when every placeholder slot has a concrete full country name.
ROUND16_SUBMISSION_SLOT_ASSIGNMENT = {
    # Final Round-of-16 bracket is now fully known, so no placeholder-slot
    # assignment is needed. BRACKET_R16 below contains only concrete country names.
}

ROUND16_SUBMISSION_CSV = OUTPUT_DIR / "round_of_16_submission_rolling_residual_v3.csv"
ROUND16_CANDIDATE_PROBABILITIES_CSV = OUTPUT_DIR / "round_of_16_candidate_probabilities_rolling_residual_v3.csv"

# Separate candidate-feature export aligned row-for-row with
# ROUND16_CANDIDATE_PROBABILITIES_CSV.
#
# This is intentionally not the competition submission file. It is for
# post-hoc calibration, diagnostics, and reference-model case studies.
ROUND16_CANDIDATE_FEATURES_CSV = OUTPUT_DIR / "round_of_16_candidate_features_rolling_residual_v3.csv"

# Explicit engineered-feature exports. These are diagnostic/reproducibility
# artifacts, not competition uploads. They let us inspect exactly what the
# baseline Prior-Labs-style feature builder created and what the final v3 model
# consumed after ratings and rolling residual-bias features are added.
BASELINE_ENGINEERED_FEATURES_CSV = INTERIM_DIR / "martj42_baseline_engineered_features.csv"
FINAL_V3_TRAINING_FEATURES_CSV = INTERIM_DIR / "final_v3_training_engineered_features.csv"
ROUND16_ENGINEERED_FEATURES_CSV = INTERIM_DIR / "round16_engineered_features_rolling_residual_v3.csv"
FEATURE_SET_DICTIONARY_CSV = INTERIM_DIR / "feature_set_dictionary.csv"

# Second OOF stream for calibration fitting: same rolling folds, but the
# model uses v2 features (baseline + DC ratings). This is the closest honest
# proxy for v3's sharpness (v3 = v2 + bias features, which cannot be used
# here without circularity), so the pooling weight fit on these predictions
# transfers to v3 far better than baseline-only OOF.
CALIBRATION_OOF_FEATURES_CSV = OUTPUT_DIR / "rolling_v2_oof_predictions_for_calibration.csv"

SUBMISSION_COLUMNS = [
    "date", "home_team", "away_team", "p_home_win", "p_draw", "p_away_win",
]

# Round-of-16 only script: no Monte Carlo settings are needed.

# Knockout games cannot end in a draw.
#
# "split_evenly":
#   p_home_adv = p_home_win + 0.5 * p_draw
#   p_away_adv = p_away_win + 0.5 * p_draw
#
# "renormalize_no_draw":
#   p_home_adv = p_home_win / (p_home_win + p_away_win)
#   p_away_adv = p_away_win / (p_home_win + p_away_win)
#
# I recommend "split_evenly" here because a model that assigns high draw
# probability is expressing uncertainty, and knockout advancement after extra
# time/penalties should preserve that uncertainty rather than discard it.
BRACKET_DRAW_POLICY = "split_evenly"

# Optional probability flattening/sharpening for sampling.
#
# 1.0 = use the model probabilities directly.
# >1.0 = flatter sampling, more upset exploration.
# <1.0 = sharper sampling, more favorite-heavy brackets.
#
# The final bracket score still uses the original model probabilities, so this
# only affects exploration, not ranking.
BRACKET_SAMPLING_TEMPERATURE = 1.0

# ── Name mapping: our Sofascore names → martj42 names ────────────────────────
OUR_TO_MARTJ42 = {
    "Republic of South Africa":      "South Africa",
    "USA":                           "United States",
    "Cote d'Ivoire":                 "Ivory Coast",
    "Cabo Verde":                    "Cape Verde",
    "Democratic Republic of Congo":  "DR Congo",
}

# All 36 World Cup teams in martj42 naming convention
WC_TEAMS = {
    "Mexico", "South Africa", "South Korea", "Switzerland",
    "Canada", "Bosnia and Herzegovina", "Brazil", "Morocco",
    "Scotland", "United States", "Australia", "Paraguay",
    "Germany", "Ivory Coast", "Ecuador", "Netherlands",
    "Japan", "Sweden", "Belgium", "Egypt", "Iran", "Spain",
    "Cape Verde", "France", "Norway", "Senegal", "Argentina",
    "Algeria", "Austria", "Colombia", "Portugal", "DR Congo",
    "Uzbekistan", "England", "Ghana", "Croatia",
}

# ── 2026 knockout bracket from the supplied image/text ────────────────────────
#
# Important:
# - Use martj42 names here, not Sofascore names.
# - The supplied text says "Cabo Verde"; martj42 uses "Cape Verde".
# - These match IDs intentionally encode bracket position, not calendar order.
BRACKET_R16 = [
    # Round of 16 fixtures derived from the completed Round-of-32 results:
    #   M73 Canada d. South Africa (0-1)      M74 Paraguay d. Germany (pens 4-3)
    #   M75 Morocco d. Netherlands (pens 3-2) M76 Brazil d. Japan (2-1)
    #   M77 France d. Sweden (3-0)            M78 Norway d. Ivory Coast (2-1)
    #   M79 Mexico d. Ecuador (2-0)           M80 England d. DR Congo (2-1)
    #   M81 United States d. Bosnia (2-0)     M82 Belgium d. Senegal (3-2 aet)
    #   M83 Portugal d. Croatia (2-1)         M84 Spain d. Austria (3-0)
    #   M85 Switzerland d. Algeria (2-0)      M86 Argentina d. Cape Verde (3-2 aet)
    #   M87 Colombia d. Ghana (1-0)           M88 Egypt d. Australia (pens 4-2)
    # Home team = winner of the first match id in each derived pair.
    {"match_id": "M89", "date": "2026-07-04", "home_team": "Paraguay",      "away_team": "France"},
    {"match_id": "M90", "date": "2026-07-04", "home_team": "Canada",        "away_team": "Morocco"},
    {"match_id": "M91", "date": "2026-07-05", "home_team": "Brazil",        "away_team": "Norway"},
    {"match_id": "M92", "date": "2026-07-05", "home_team": "Mexico",        "away_team": "England"},
    {"match_id": "M93", "date": "2026-07-06", "home_team": "Portugal",      "away_team": "Spain"},
    {"match_id": "M94", "date": "2026-07-06", "home_team": "United States", "away_team": "Belgium"},
    {"match_id": "M95", "date": "2026-07-07", "home_team": "Argentina",     "away_team": "Egypt"},
    {"match_id": "M96", "date": "2026-07-07", "home_team": "Switzerland",   "away_team": "Colombia"},
]

# Candidate teams for unresolved bracket slots.
# The updated Round-of-16 bracket is fully resolved, so this remains empty.
BRACKET_SLOT_CANDIDATES = {}


BRACKET_DERIVED_ROUNDS = {
    # Round of 16 is now the concrete root round (BRACKET_R16 above), so the
    # first derived layer is the Quarterfinal.
    "Quarterfinal": [
        {"match_id": "M97",  "date": "2026-07-09", "from": ("M89", "M90")},
        {"match_id": "M98",  "date": "2026-07-10", "from": ("M93", "M94")},
        {"match_id": "M99",  "date": "2026-07-11", "from": ("M91", "M92")},
        {"match_id": "M100", "date": "2026-07-11", "from": ("M95", "M96")},
    ],
    "Semifinal": [
        {"match_id": "M101", "date": "2026-07-14", "from": ("M97", "M98")},
        {"match_id": "M102", "date": "2026-07-15", "from": ("M99", "M100")},
    ],
    "Final": [
        {"match_id": "M104", "date": "2026-07-19", "from": ("M101", "M102")},
    ],
}

BRACKET_THIRD_PLACE = {
    "match_id": "M103",
    "date": "2026-07-18",
    "from_losers": ("M101", "M102"),
}

# ── Feature sets ──────────────────────────────────────────────────────────────
BASELINE_FEATURES = [
    "elo_diff", "home_elo", "away_elo",
    "form5_diff", "form10_diff", "home_form5", "away_form5",
    "home_winrate", "away_winrate",
    "home_gf5", "away_gf5", "home_ga5", "away_ga5", "gd10_diff",
    "home_streak", "away_streak", "home_rest", "away_rest",
    "home_played", "away_played",
    "h2h_n", "h2h_home_winrate", "h2h_draw_rate", "h2h_gd",
    "neutral", "importance",
]

AUG_FEATURES = [
    "home_attack_rating", "home_defense_rating",
    "away_attack_rating", "away_defense_rating",
]

BIAS_FEATURES = [
    "home_attack_bias", "home_defense_bias",
    "away_attack_bias", "away_defense_bias",
]

AUGMENTED_FEATURES = BASELINE_FEATURES + AUG_FEATURES

# Style features are added only to the selected v3 model, not to the residual
# baseline and not to the first-layer ratings model.
#
# This preserves the existing rolling-residual design:
#   v1 baseline      : martj42 ELO/form/H2H only
#   v2 ratings       : v1 + attack/defense ratings
#   v3 residual      : v2 + rolling residual-bias features
#   v3 + style toggle: v3 + time-aware style-triplet categories
STYLE_FEATURES: list = []
AUGMENTED_V3_FEATURES: list = []


def configure_style_features(use_style_keywords: bool) -> None:
    """
    Central switch for adding/removing style keyword covariates.

    Keeping this centralized prevents partial toggles where the columns are
    joined onto the dataframe but accidentally excluded from the model, or the
    reverse: expected by the model but never created.
    """
    global STYLE_FEATURES, AUGMENTED_V3_FEATURES

    STYLE_FEATURES = (
        [STYLE_HOME_FEATURE, STYLE_AWAY_FEATURE]
        if use_style_keywords
        else []
    )

    AUGMENTED_V3_FEATURES = (
        BASELINE_FEATURES
        + AUG_FEATURES
        + BIAS_FEATURES
        + STYLE_FEATURES
    )


configure_style_features(USE_STYLE_KEYWORDS)

# v4 feature list is built dynamically from team_characteristics.csv columns.
# Populated by load_team_characteristics() at runtime.
CHAR_FEATURES: list = []          # filled in after file is loaded
AUGMENTED_V4_FEATURES: list = []  # = BASELINE_FEATURES + CHAR_FEATURES

# Convenience alias used when filling NaNs for non-WC teams
BIAS_COLS = BIAS_FEATURES

# Canonical outcome labels used throughout this script.
#
# We need this because FinetunedTabPFNClassifier officially supports:
#   - predict(X)
#   - predict_proba(X)
#
# but does not necessarily expose:
#   - classes_
#
# For string labels, sklearn-style encoders usually use np.unique(y), which gives:
#   ["away_win", "draw", "home_win"]
#
# We validate this assumption against clf.predict(X) before using the probability
# columns for log-loss or bracket probabilities.
OUTCOME_CLASSES = np.array(["away_win", "draw", "home_win"])


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Prior Labs feature engineering (verbatim from predict.py)
# ─────────────────────────────────────────────────────────────────────────────

def importance(t):
    t = t.lower()
    if "world cup" in t and "qual" not in t: return 60.0
    if "confederations" in t:                return 50.0
    if any(k in t for k in ["uefa euro", "copa am", "african cup", "asian cup",
                             "gold cup", "nations league", "oceania nations"]): return 45.0
    if "qualif" in t:   return 35.0
    if "friendly" in t: return 20.0
    return 30.0


def load_data(refresh=False):
    if refresh or not os.path.exists(DATA):
        df = pd.read_csv(RAW_URL)
        df.to_csv(DATA, index=False)
    else:
        df = pd.read_csv(DATA)
    df["date"]       = pd.to_datetime(df["date"])
    df               = df.sort_values("date").reset_index(drop=True)
    df["neutral"]    = df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["outcome"]    = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] < df["away_score"]],
        ["home_win", "away_win"], default="draw")
    df.loc[df["home_score"].isna(), "outcome"] = np.nan
    df["importance"] = df["tournament"].apply(importance)
    return df


def _compute_team_feats(team, elo, res):
    r = res[team]
    if not r:
        return elo[team], 1.3, 1.3, 0.33, 1.0, 1.0, 0.0, 0.0, 0
    last5, last10 = r[-5:], r[-10:]
    streak = 0
    for p, *_ in reversed(r):
        if p < 1: break
        streak += 1
    return (elo[team],
            np.mean([p for p, *_ in last5]),
            np.mean([p for p, *_ in last10]),
            np.mean([w for *_, w in last10]),
            np.mean([g for _, g, _, _ in last5]),
            np.mean([a for _, _, a, _ in last5]),
            np.mean([g - a for _, g, a, _ in last10]),
            streak, len(r))


def _compute_h2h_feats(home, away, h2h):
    m = h2h[tuple(sorted((home, away)))]
    if not m: return 0, 0.5, 0.25, 0.0
    n = len(m)
    return (n,
            sum(w == home for _, _, w in m) / n,
            sum(w == "draw" for _, _, w in m) / n,
            np.mean([g if h == home else -g for h, g, _ in m]))


def build_features(df):
    """Build features for all rows. Returns (feats_df, state_snapshot).

    state_snapshot captures ELO, form history, last dates and H2H after all
    played matches are processed. It is passed to make_single_fixture_features_fast
    so knockout predictions never need to replay history.
    """
    elo       = defaultdict(lambda: 1500.0)
    res       = defaultdict(list)
    last_date = {}
    h2h       = defaultdict(list)

    rows = []
    for r in df.itertuples():
        h, a, adj = r.home_team, r.away_team, HOME_ADV * (1 - r.neutral)
        he, hf5, hf10, hwr, hgf, hga, hgd, hstk, hn = _compute_team_feats(h, elo, res)
        ae, af5, af10, awr, agf, aga, agd, astk, an  = _compute_team_feats(a, elo, res)
        nm, h2h_wr, h2h_dr, h2h_gd = _compute_h2h_feats(h, a, h2h)
        rows.append({
            "elo_diff":       he + adj - ae,
            "home_elo":       he,  "away_elo":  ae,
            "form5_diff":     hf5 - af5,  "form10_diff": hf10 - af10,
            "home_form5":     hf5, "away_form5": af5,
            "home_winrate":   hwr, "away_winrate": awr,
            "home_gf5":       hgf, "away_gf5": agf,
            "home_ga5":       hga, "away_ga5": aga,
            "gd10_diff":      hgd - agd,
            "home_streak":    hstk, "away_streak": astk,
            "home_rest":      min((r.date - last_date[h]).days, 90) if h in last_date else 30,
            "away_rest":      min((r.date - last_date[a]).days, 90) if a in last_date else 30,
            "home_played":    hn,  "away_played": an,
            "h2h_n":          nm,  "h2h_home_winrate": h2h_wr,
            "h2h_draw_rate":  h2h_dr, "h2h_gd": h2h_gd,
        })

        if not np.isnan(r.home_score):
            gd    = r.home_score - r.away_score
            exp   = 1 / (1 + 10 ** ((ae - he - adj) / 400))
            s     = 1.0 if gd > 0 else (0.0 if gd < 0 else 0.5)
            g     = 1.0 if abs(gd) <= 1 else (1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8)
            delta = r.importance * g * (s - exp)
            elo[h] += delta; elo[a] -= delta
            res[h].append((3 if gd > 0 else (1 if gd == 0 else 0), r.home_score, r.away_score, gd > 0))
            res[a].append((3 if gd < 0 else (1 if gd == 0 else 0), r.away_score, r.home_score, gd < 0))
            last_date[h] = last_date[a] = r.date
            h2h[tuple(sorted((h, a)))].append((h, gd, h if gd > 0 else (a if gd < 0 else "draw")))

    # Snapshot captured after all played matches — used by knockout feature builder
    state_snapshot = {
        "elo":       dict(elo),
        "res":       {k: list(v) for k, v in res.items()},
        "last_date": dict(last_date),
        "h2h":       {k: list(v) for k, v in h2h.items()},
    }

    return df.join(pd.DataFrame(rows, index=df.index)), state_snapshot


def make_single_fixture_features_fast(ratings, state_snapshot, home_team, away_team,
                                      match_date, characteristics=None, stat_cols=None,
                                      style_eras_by_team=None, style_encoder=None):
    """
    Builds one knockout fixture feature row directly from the pre-computed state
    snapshot — no history replay.

    This replaces the original make_single_fixture_features which called
    build_features on the full dataset for every single knockout prediction,
    making 50k-simulation Monte Carlo runs take hours.
    """
    elo       = defaultdict(lambda: 1500.0, state_snapshot["elo"])
    res       = defaultdict(list, {k: list(v) for k, v in state_snapshot["res"].items()})
    last_date = dict(state_snapshot["last_date"])
    h2h       = defaultdict(list, {k: list(v) for k, v in state_snapshot["h2h"].items()})

    match_ts  = pd.Timestamp(match_date)

    he, hf5, hf10, hwr, hgf, hga, hgd, hstk, hn = _compute_team_feats(home_team, elo, res)
    ae, af5, af10, awr, agf, aga, agd, astk, an  = _compute_team_feats(away_team, elo, res)
    nm, h2h_wr, h2h_dr, h2h_gd                   = _compute_h2h_feats(home_team, away_team, h2h)

    # All knockout matches are at neutral venues — no home advantage
    adj = 0.0

    row = {
        "elo_diff":           he + adj - ae,
        "home_elo":           he,  "away_elo":  ae,
        "form5_diff":         hf5 - af5,  "form10_diff": hf10 - af10,
        "home_form5":         hf5, "away_form5": af5,
        "home_winrate":       hwr, "away_winrate": awr,
        "home_gf5":           hgf, "away_gf5": agf,
        "home_ga5":           hga, "away_ga5": aga,
        "gd10_diff":          hgd - agd,
        "home_streak":        hstk, "away_streak": astk,
        "home_rest":          min((match_ts - last_date[home_team]).days, 90) if home_team in last_date else 30,
        "away_rest":          min((match_ts - last_date[away_team]).days, 90) if away_team in last_date else 30,
        "home_played":        hn,  "away_played": an,
        "h2h_n":              nm,  "h2h_home_winrate": h2h_wr,
        "h2h_draw_rate":      h2h_dr, "h2h_gd": h2h_gd,
        "neutral":            1,
        "importance":         60.0,  # FIFA World Cup knockout
        "home_team":          home_team,
        "away_team":          away_team,
        # ratings and bias
        "home_attack_rating":  ratings.get(home_team, {}).get("attack_rating",         np.nan),
        "home_defense_rating": ratings.get(home_team, {}).get("defense_rating",        np.nan),
        "home_attack_bias":    ratings.get(home_team, {}).get("attack_residual_bias",  np.nan),
        "home_defense_bias":   ratings.get(home_team, {}).get("defense_residual_bias", np.nan),
        "away_attack_rating":  ratings.get(away_team, {}).get("attack_rating",         np.nan),
        "away_defense_rating": ratings.get(away_team, {}).get("defense_rating",        np.nan),
        "away_attack_bias":    ratings.get(away_team, {}).get("attack_residual_bias",  np.nan),
        "away_defense_bias":   ratings.get(away_team, {}).get("defense_residual_bias", np.nan),
    }

    # Join time-aware style keyword triplets if enabled.
    #
    # This must happen here as well as in join_style_keywords(...), because
    # Round-of-16 fixtures are generated directly from the state snapshot rather
    # than pulled from the historical feature dataframe.
    if STYLE_FEATURES:
        if style_eras_by_team is None or style_encoder is None:
            raise ValueError(
                "Style keyword features are enabled, but fixture generation did "
                "not receive style_eras_by_team and style_encoder."
            )

        for side, team in [("home", home_team), ("away", away_team)]:
            triplet = lookup_style_triplet(
                style_eras_by_team=style_eras_by_team,
                team=team,
                match_date=match_ts,
            )
            row[f"{side}_style_triplet"] = triplet
            row[f"{side}_style_triplet_code"] = encode_style_triplet(
                keyword_triplet=triplet,
                style_encoder=style_encoder,
            )

    fixture_df = pd.DataFrame([row])

    # Join team characteristic averages if provided (for v4)
    if characteristics is not None and stat_cols is not None:
        for side, team in [("home", home_team), ("away", away_team)]:
            for stat in stat_cols:
                fixture_df[f"{side}_{stat}"] = characteristics.get(team, {}).get(stat, np.nan)

    return fixture_df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load and join our team ratings
# ─────────────────────────────────────────────────────────────────────────────

RATING_HISTORY = INTERIM_DIR / "team_rating_history.csv"

def load_rating_history():
    """Long table of checkpointed DC ratings: as_of_date, team, attack/defense."""
    hist = pd.read_csv(RATING_HISTORY)
    hist["as_of_date"] = pd.to_datetime(hist["as_of_date"])
    hist["team"] = hist["team"].replace(OUR_TO_MARTJ42)
    hist = hist.sort_values("as_of_date").reset_index(drop=True)
    print(f"  Loaded rating history: {len(hist):,} rows, "
          f"{hist['as_of_date'].nunique()} checkpoints, {hist['team'].nunique()} teams")
    return hist


def join_ratings(feats, rating_history):
    """
    As-of join: each match row receives the most recent checkpoint rating
    strictly before its date, per team. Rows earlier than the first checkpoint
    (or teams absent from history) get NaN and are filtered later by
    require_complete_features, same as before.
    """
    feats = feats.copy().sort_values("date")

    for side in ["home", "away"]:
        side_hist = rating_history.rename(columns={
            "team":           f"{side}_team",
            "attack_rating":  f"{side}_attack_rating",
            "defense_rating": f"{side}_defense_rating",
        })[["as_of_date", f"{side}_team",
            f"{side}_attack_rating", f"{side}_defense_rating"]]

        feats = pd.merge_asof(
            feats,
            side_hist.sort_values("as_of_date"),
            left_on="date", right_on="as_of_date",
            by=f"{side}_team",
            direction="backward",
            allow_exact_matches=False,   # checkpoint must strictly predate the match
        ).drop(columns=["as_of_date"])

    # Bias columns still start as NaN; rolling residual step populates them.
    for col in BIAS_FEATURES:
        feats[col] = np.nan

    feats = feats.sort_index()
    wc_rows  = feats["home_team"].isin(WC_TEAMS) | feats["away_team"].isin(WC_TEAMS)
    coverage = feats.loc[wc_rows, "home_attack_rating"].notna().mean()
    print(f"  As-of rating coverage across WC team matches: {coverage:.1%}")
    return feats

def latest_ratings_dict(rating_history):
    """
    Latest-checkpoint snapshot as {team: {"attack_rating", "defense_rating"}} —
    the dict shape the fixture builder, bias enrichment, and slot candidate
    pool expect. Used ONLY for future fixtures; historical rows get their
    ratings from the as-of join.
    """
    latest = (
        rating_history.sort_values("as_of_date")
        .groupby("team", as_index=False)
        .tail(1)
    )
    return {
        r.team: {"attack_rating": float(r.attack_rating),
                 "defense_rating": float(r.defense_rating)}
        for r in latest.itertuples()
    }

# ─────────────────────────────────────────────────────────────────────────────
# 2a. Load and join time-aware team style keywords
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_style_triplet(value):
    """
    Normalizes a style triplet into the exact categorical token used by the model.

    Missing or blank values become STYLE_UNKNOWN_TOKEN so the model keeps the row
    but can distinguish "no style evidence available" from a real style era.
    """
    if pd.isna(value):
        return STYLE_UNKNOWN_TOKEN

    token = str(value).strip().lower()

    if not token:
        return STYLE_UNKNOWN_TOKEN

    return token


def load_style_keyword_eras():
    """
    Loads the time-aware team style era table.

    Returns:
    - style_eras_by_team:
        dict team -> DataFrame of that team's style eras
    - style_encoder:
        dict keyword_triplet -> stable integer code

    The encoder is intentionally deterministic and file-wide:
    - "unknown" is always 0.
    - Real triplets are sorted alphabetically.
    - This avoids run-to-run code changes when row order changes.
    """
    if not STYLE_FEATURES:
        print("  Style keyword features: disabled")
        return {}, {STYLE_UNKNOWN_TOKEN: 0}

    if not STYLE_ERAS.exists():
        raise FileNotFoundError(
            f"Style keyword eras file not found: {STYLE_ERAS}\n"
            "Expected file: data/interim/team_style_keyword_eras_round32_time_aware.csv\n"
            "Run/copy the time-aware style-era export before enabling style keywords, "
            "or pass --disable-style-keywords."
        )

    eras = pd.read_csv(STYLE_ERAS)
    eras["team"] = eras["team"].replace(OUR_TO_MARTJ42)

    required = {
        "team",
        "style_period_start",
        "style_period_end",
        "keyword_triplet",
    }
    missing = sorted(required - set(eras.columns))

    if missing:
        raise ValueError(
            f"Style eras file is missing required column(s): {missing}\n"
            f"Found columns: {list(eras.columns)}"
        )

    eras["style_period_start"] = pd.to_datetime(
        eras["style_period_start"],
        errors="coerce",
    )
    eras["style_period_end"] = pd.to_datetime(
        eras["style_period_end"],
        errors="coerce",
    ).fillna(pd.Timestamp("2100-12-31"))

    eras["keyword_triplet"] = eras["keyword_triplet"].map(_normalize_style_triplet)

    bad_dates = eras["style_period_start"].isna() | eras["style_period_end"].isna()
    if bad_dates.any():
        bad_rows = eras.loc[bad_dates, ["team", "style_period_start", "style_period_end"]]
        raise ValueError(
            "Style eras file contains unparsable style period dates:\n"
            f"{bad_rows.to_string(index=False)}"
        )

    style_eras_by_team = {
        team: group.sort_values("style_period_start").reset_index(drop=True)
        for team, group in eras.groupby("team", sort=False)
    }

    real_triplets = sorted(
        token
        for token in eras["keyword_triplet"].dropna().unique()
        if token != STYLE_UNKNOWN_TOKEN
    )

    style_encoder = {STYLE_UNKNOWN_TOKEN: 0}
    for token in real_triplets:
        style_encoder[token] = len(style_encoder)

    print(
        "  Loaded style keyword eras: "
        f"{len(eras)} eras, {len(style_eras_by_team)} teams, "
        f"{len(style_encoder)} encoded triplet categories including '{STYLE_UNKNOWN_TOKEN}'"
    )

    return style_eras_by_team, style_encoder


def lookup_style_triplet(style_eras_by_team, team, match_date):
    """
    Returns the style keyword triplet active for a team on a match date.

    This is the key anti-leakage rule:
    - no current/future team identity is backfilled into old matches
    - each row gets the style era active at that row's date
    """
    if not STYLE_FEATURES:
        return STYLE_UNKNOWN_TOKEN

    team = OUR_TO_MARTJ42.get(team, team)
    eras = style_eras_by_team.get(team)

    if eras is None or len(eras) == 0:
        return STYLE_UNKNOWN_TOKEN

    match_ts = pd.Timestamp(match_date).normalize()

    active = eras[
        (eras["style_period_start"] <= match_ts)
        & (eras["style_period_end"] >= match_ts)
    ]

    if active.empty:
        return STYLE_UNKNOWN_TOKEN

    return _normalize_style_triplet(active.iloc[-1]["keyword_triplet"])


def encode_style_triplet(keyword_triplet, style_encoder):
    """
    Converts a keyword triplet into a stable numeric category for TabPFN.

    Unseen categories fall back to "unknown" instead of creating new codes at
    prediction time.
    """
    token = _normalize_style_triplet(keyword_triplet)
    return int(style_encoder.get(token, style_encoder[STYLE_UNKNOWN_TOKEN]))


def join_style_keywords(feats, style_eras_by_team, style_encoder):
    """
    Adds time-aware style-triplet audit columns and numeric model features.

    Model features:
    - home_style_triplet_code
    - away_style_triplet_code

    Audit columns:
    - home_style_triplet
    - away_style_triplet

    Missing coverage is encoded as "unknown" rather than NaN so the feature never
    causes historical rows to be dropped by require_complete_features(...).
    """
    if not STYLE_FEATURES:
        return feats

    feats = feats.copy()

    for side in ["home", "away"]:
        team_col = f"{side}_team"
        triplet_col = f"{side}_style_triplet"
        code_col = f"{side}_style_triplet_code"

        feats[triplet_col] = [
            lookup_style_triplet(
                style_eras_by_team=style_eras_by_team,
                team=team,
                match_date=match_date,
            )
            for team, match_date in zip(feats[team_col], feats["date"])
        ]

        feats[code_col] = feats[triplet_col].map(
            lambda token: encode_style_triplet(token, style_encoder)
        )

    known_home = feats["home_style_triplet"].ne(STYLE_UNKNOWN_TOKEN).mean()
    known_away = feats["away_style_triplet"].ne(STYLE_UNKNOWN_TOKEN).mean()

    print(
        "  Style keyword coverage across all feature rows: "
        f"home={known_home:.1%}, away={known_away:.1%}"
    )

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# 2b. Load and join team characteristics (per-team average match stats)
# ─────────────────────────────────────────────────────────────────────────────

def load_team_characteristics():
    """
    Loads team_characteristics.csv — per-team averages of ~36 match-level stats
    exported by the retired Sofascore pipeline (see README: inactive v4 path).

    Returns a dict: martj42_name -> {stat_name: value, ...}
    Also populates the global CHAR_FEATURES and AUGMENTED_V4_FEATURES lists.
    """
    global CHAR_FEATURES, AUGMENTED_V4_FEATURES

    tc = pd.read_csv(CHARACTERISTICS)
    tc["team"] = tc["team"].replace(OUR_TO_MARTJ42)

    # Stat columns = everything except 'team'
    stat_cols = [c for c in tc.columns if c != "team"]

    characteristics = tc.set_index("team")[stat_cols].to_dict(orient="index")

    # Build home_/away_ prefixed feature names
    CHAR_FEATURES       = [f"home_{c}" for c in stat_cols] + [f"away_{c}" for c in stat_cols]
    AUGMENTED_V4_FEATURES = BASELINE_FEATURES + CHAR_FEATURES

    print(f"  Loaded characteristics for {len(characteristics)} teams "
          f"({len(stat_cols)} stats × 2 sides = {len(CHAR_FEATURES)} features)")
    return characteristics, stat_cols


def join_characteristics(feats, characteristics, stat_cols):
    """
    Joins per-team characteristic averages onto each fixture row as
    home_{stat} and away_{stat} columns.
    Missing teams get NaN (treated as no information by TabPFN).
    """
    feats = feats.copy()
    for side in ["home", "away"]:
        col = f"{side}_team"
        for stat in stat_cols:
            feats[f"{side}_{stat}"] = feats[col].map(
                lambda t, s=stat: characteristics.get(t, {}).get(s, np.nan)
            )

    wc_rows  = feats["home_team"].isin(WC_TEAMS) | feats["away_team"].isin(WC_TEAMS)
    coverage = feats.loc[wc_rows, f"home_{stat_cols[0]}"].notna().mean()
    print(f"  Characteristic coverage across WC team matches: {coverage:.1%}")
    return feats




def make_tabpfn_classifier(*, use_thinking_mode, random_state=42):
    """
    Creates a TabPFN client classifier with a single centralized policy for
    regular vs thinking-mode fits.

    This script variant can use thinking mode both for the residual-generating
    baseline model and for the final v3 model, controlled by separate flags.
    """
    kwargs = {
        "ignore_pretraining_limits": True,
        "random_state": random_state,
    }

    if use_thinking_mode:
        kwargs.update({
            "thinking_mode": True,
            "thinking_effort": FINAL_V3_THINKING_EFFORT,
            "thinking_metric": FINAL_V3_THINKING_METRIC,
        })

        if FINAL_V3_THINKING_TIMEOUT_S is not None:
            kwargs["thinking_timeout_s"] = FINAL_V3_THINKING_TIMEOUT_S

    return TabPFNClassifier(**kwargs)


def compute_baseline_residual_bias(feats, ratings):
    """
    Computes per-team residual bias using ONLY already-played WC 2026 matches.
    Bias = how systematically the baseline over/underestimates each team in
    THIS tournament (2-3 matches per team — small but captures tournament form).

    attack_residual_bias  : mean(I(team won) − p_baseline(team wins))
    defense_residual_bias : mean(I(team didn't lose) − p_baseline(team doesn't lose))
    """
    WC_START  = pd.Timestamp("2026-06-11")
    played    = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)].copy()
    pre_wc    = played[played["date"] < WC_START]
    wc_played = played[played["date"] >= WC_START].copy()

    print(f"  Pre-WC training rows : {len(pre_wc)}")
    print(f"  WC played matches    : {len(wc_played)}")

    # Train baseline on pre-WC data to get uncontaminated predictions.
    # In this variant, the residual-generating baseline uses the same
    # thinking-mode family as the final v3 model, so the residual features are
    # generated and consumed within a consistent thinking-mode pipeline.
    print(
        "  Residual baseline fit uses TabPFN client thinking mode: "
        f"enabled={RESIDUAL_BASELINE_THINKING_MODE}, "
        f"effort={FINAL_V3_THINKING_EFFORT}, "
        f"metric={FINAL_V3_THINKING_METRIC}, "
        f"timeout_s={FINAL_V3_THINKING_TIMEOUT_S}"
    )
    clf = make_tabpfn_classifier(
        use_thinking_mode=RESIDUAL_BASELINE_THINKING_MODE,
        random_state=42,
    )
    clf.fit(pre_wc[BASELINE_FEATURES].values, pre_wc["outcome"].values)
    classes = list(clf.classes_)

    wc_our = wc_played[
        wc_played["home_team"].isin(WC_TEAMS) |
        wc_played["away_team"].isin(WC_TEAMS)
    ].copy()
    print(f"  WC matches involving WC teams: {len(wc_our)}")

    proba            = clf.predict_proba(wc_our[BASELINE_FEATURES].values)
    wc_our["p_home_win"] = proba[:, classes.index("home_win")]
    wc_our["p_draw"]     = proba[:, classes.index("draw")]
    wc_our["p_away_win"] = proba[:, classes.index("away_win")]

    attack_bias, defense_bias = {}, {}
    for team in WC_TEAMS:
        home_rows = wc_our[wc_our["home_team"] == team]
        away_rows = wc_our[wc_our["away_team"] == team]

        if len(home_rows) + len(away_rows) == 0:
            attack_bias[team] = defense_bias[team] = np.nan
            continue

        home_win_surp = (home_rows["outcome"] == "home_win").astype(float) - home_rows["p_home_win"]
        away_win_surp = (away_rows["outcome"] == "away_win").astype(float) - away_rows["p_away_win"]
        attack_bias[team] = pd.concat([home_win_surp, away_win_surp]).mean()

        home_not_loss = (home_rows["outcome"] != "away_win").astype(float) \
                      - (home_rows["p_home_win"] + home_rows["p_draw"])
        away_not_loss = (away_rows["outcome"] != "home_win").astype(float) \
                      - (away_rows["p_away_win"] + away_rows["p_draw"])
        defense_bias[team] = pd.concat([home_not_loss, away_not_loss]).mean()

    bias_df = pd.DataFrame({
        "attack_residual_bias":  attack_bias,
        "defense_residual_bias": defense_bias,
    }).rename_axis("team").reset_index()

    print("\n  WC residual bias by team (sorted by attack surprise):")
    print(bias_df.dropna().sort_values("attack_residual_bias", ascending=False).to_string(index=False))
    print(f"\n  Teams with no WC matches (NaN): {bias_df['attack_residual_bias'].isna().sum()}")
    return bias_df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Model training helpers
# ─────────────────────────────────────────────────────────────────────────────


def require_complete_features(df, feature_cols, label):
    """
    Drops rows that are missing any feature required by a model variant.

    This is intentionally used for the expanded ratings-based models.
    If a model's feature set depends on team ratings or WC residual-bias fields,
    then matches involving teams without those ratings should not be silently
    imputed into the expanded-model training/evaluation set.
    """
    before = len(df)

    keep_mask = df[feature_cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    filtered = df.loc[keep_mask].copy()

    after = len(filtered)
    dropped = before - after

    print(f"  {label}: kept {after} / {before} rows; dropped {dropped} rows with missing required features")

    if filtered.empty:
        missing_counts = df[feature_cols].isna().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

        raise ValueError(
            f"{label}: no rows remain after dropping missing required features.\n"
            f"Missing counts:\n{missing_counts.to_string()}"
        )

    return filtered


def assert_no_missing_features(df, feature_cols, label):
    """
    Defensive check before fitting or predicting.

    Expanded models should never receive rows with missing rating/bias features.
    """
    feature_frame = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    if feature_frame.isna().any().any():
        missing_counts = feature_frame.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

        raise ValueError(
            f"{label}: missing values remain in required features.\n"
            f"Missing counts:\n{missing_counts.to_string()}"
        )


def Xf(df, cols, label):
    """
    Converts a DataFrame slice into a finite numeric model matrix.

    This does not impute. It fails if missing values remain.
    """
    assert_no_missing_features(df, cols, label)

    X = df[cols].to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError(f"{label}: non-finite values found after conversion to numpy.")

    return X


def get_probability_classes(clf):
    """
    Returns the class order corresponding to columns of clf.predict_proba(X).

    We search several common fitted-classifier attributes first. This keeps the
    code robust across TabPFN / sklearn-style wrappers.

    If no class metadata is available, we fall back to OUTCOME_CLASSES:
        ["away_win", "draw", "home_win"]

    That fallback is appropriate here because this script owns the target label
    construction and always uses these three string labels.
    """
    candidate_attrs = [
        "classes_",
        "classes",
        "class_names_",
        "class_names",
    ]

    for attr in candidate_attrs:
        if hasattr(clf, attr):
            value = getattr(clf, attr)

            if value is not None:
                return np.asarray(value)

    nested_encoder_attrs = [
        "label_encoder_",
        "target_encoder_",
        "_label_encoder",
        "_target_encoder",
    ]

    for attr in nested_encoder_attrs:
        if hasattr(clf, attr):
            encoder = getattr(clf, attr)

            if hasattr(encoder, "classes_"):
                return np.asarray(encoder.classes_)

    return OUTCOME_CLASSES.copy()

def resolve_probability_classes(clf, proba, label):
    """
    Resolves the class labels corresponding to predict_proba columns.

    Important:
    - TabPFNClassifier usually exposes clf.classes_.
    - FinetunedTabPFNClassifier officially documents predict() and
      predict_proba(), but may not expose classes_.
    - We do NOT require predict(X) to equal argmax(predict_proba(X)).
      Some wrappers/configurations may produce hard predictions through a path
      that is not exactly equivalent to taking the argmax of predict_proba.

    Therefore:
    1. Prefer class metadata from the fitted classifier when available.
    2. Fall back to OUTCOME_CLASSES for the known three-class football outcome.
    3. Only fail if the number of probability columns is inconsistent.
    """
    probability_classes = get_probability_classes(clf)

    if proba.shape[1] != len(probability_classes):
        raise ValueError(
            f"{label}: predict_proba returned {proba.shape[1]} columns, "
            f"but resolved probability_classes has {len(probability_classes)} "
            f"entries: {probability_classes}"
        )

    return probability_classes


def print_draw_probability_diagnostics(clf, X, y, label):
    """
    Shows whether the model is assigning meaningful probability mass to draws.

    This is diagnostic only. It helps distinguish:
    - no draws because p_draw is genuinely low
    - no draws because class columns are mislabeled
    - no draws because hard predictions came from clf.predict()
    """
    proba = clf.predict_proba(X)

    if not np.isfinite(proba).all():
        print(f"  {label}: cannot diagnose draw probabilities; proba has NaN/Inf.")
        return

    classes = resolve_probability_classes(clf, proba, label)

    if "draw" not in classes:
        print(f"  {label}: no 'draw' class found in classes={classes}")
        return

    draw_idx = list(classes).index("draw")
    p_draw = proba[:, draw_idx]
    argmax_pred = classes[proba.argmax(axis=1)]

    print(f"\n  Draw diagnostics for {label}:")
    print(f"    actual draw rate          : {np.mean(np.asarray(y) == 'draw'):.1%}")
    print(f"    argmax draw prediction rate: {np.mean(argmax_pred == 'draw'):.1%}")
    print(f"    mean p_draw               : {np.mean(p_draw):.4f}")
    print(f"    median p_draw             : {np.median(p_draw):.4f}")
    print(f"    max p_draw                : {np.max(p_draw):.4f}")

    draw_rows = np.asarray(y) == "draw"
    if draw_rows.any():
        print(f"    mean p_draw on true draws  : {np.mean(p_draw[draw_rows]):.4f}")

def report_predict_argmax_disagreement(clf, X, label, n_check=100):
    """
    Diagnostic only.

    Reports how often clf.predict(X) differs from argmax(predict_proba(X)).
    This should not be used as a correctness assertion.
    """
    X_check = X[:min(len(X), n_check)]

    if len(X_check) == 0:
        print(f"  {label}: no rows available for predict/argmax diagnostic.")
        return

    pred = np.asarray(clf.predict(X_check))
    proba = clf.predict_proba(X_check)

    probability_classes = resolve_probability_classes(
        clf=clf,
        proba=proba,
        label=label,
    )

    argmax_pred = probability_classes[proba.argmax(axis=1)]
    disagreement_rate = np.mean(pred != argmax_pred)

    print(
        f"  {label}: predict vs proba-argmax disagreement = "
        f"{disagreement_rate:.1%} on {len(X_check)} checked rows"
    )

def train_tabpfn(X_train, y_train, *, use_thinking_mode=False):
    clf = make_tabpfn_classifier(
        use_thinking_mode=use_thinking_mode,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, X_test, y_test, label):
    """
    Evaluates a fitted classifier using accuracy and multiclass log-loss.

    Hard labels:
        use clf.predict(X_test)

    Probabilities:
        use clf.predict_proba(X_test)

    Probability column names:
        use fitted classifier metadata when available; otherwise fall back to
        OUTCOME_CLASSES.

    We intentionally do not assert that predict(X) equals
    argmax(predict_proba(X)), because that is not guaranteed by the documented
    fine-tuning interface.
    """
    # Call predict_proba only once. On CUDA, TabPFN v3 can fail on the second
    # immediate inference pass for the same fitted model, especially with larger
    # in-context training sets. Deriving the hard label from the probability
    # argmax is also the same decision rule used by the fixture/bracket code.
    proba = clf.predict_proba(X_test)

    if not np.isfinite(proba).all():
        raise ValueError(
            f"{label}: predict_proba returned NaN or Inf values. "
            "This usually means the model received invalid inputs or fine-tuning became unstable."
        )

    probability_classes = resolve_probability_classes(
        clf=clf,
        proba=proba,
        label=label,
    )

    pred = probability_classes[proba.argmax(axis=1)]

    acc = accuracy_score(y_test, pred)
    ll  = log_loss(y_test, proba, labels=probability_classes)

    print(f"  {label:20s}  accuracy={acc:.1%}  log-loss={ll:.4f}")
    return acc, ll, pred, proba, probability_classes


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Rolling residual-bias feature generation
# ─────────────────────────────────────────────────────────────────────────────

def _predict_named_probabilities(clf, X, label):
    """
    Returns a DataFrame with home_win/draw/away_win probabilities in canonical
    named columns, independent of the classifier's internal class order.
    """
    proba = clf.predict_proba(X)

    if not np.isfinite(proba).all():
        raise ValueError(f"{label}: predict_proba returned NaN or Inf values.")

    classes = resolve_probability_classes(
        clf=clf,
        proba=proba,
        label=label,
    )

    out = pd.DataFrame(index=np.arange(proba.shape[0]))
    for class_name in ["home_win", "draw", "away_win"]:
        out[f"p_{class_name}"] = [
            get_class_probability(row, classes, class_name)
            for row in proba
        ]

    row_sums = out[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
    out["p_home_win"] = out["p_home_win"] / row_sums
    out["p_draw"] = out["p_draw"] / row_sums
    out["p_away_win"] = out["p_away_win"] / row_sums
    return out


def compute_rolling_residual_bias_features(feats):
    """
    Adds pre-match rolling residual-bias features to the match-level feature table.

    Design:
    1. Keep only played rows from TRAIN_START onward with complete rating features.
    2. For each validation calendar year, fit a baseline TabPFN model on all
       earlier eligible rows and predict that year's eligible rows.
    3. Convert out-of-sample probabilities into team-level residual events.
    4. For each team, compute prior expanding mean residuals using shift(1), so
       the bias attached to a match only uses earlier residual events.
    5. Return the enriched feature frame and a latest-bias table for future
       Round-of-16 fixtures.

    This keeps residual-bias features historically populated without using
    same-row outcomes or filling missing residuals with artificial zeros.
    """
    print("\n" + "=" * 65)
    print("Computing rolling out-of-sample residual-bias features...")
    print("=" * 65)

    feats = feats.copy()
    for col in BIAS_COLS:
        feats[col] = np.nan

    played = feats[
        feats["outcome"].notna()
        & (feats["date"] >= TRAIN_START)
    ].copy()

    eligible = require_complete_features(
        played,
        AUGMENTED_FEATURES,
        "rolling residual eligible rows with attack/defense ratings",
    )
    eligible = eligible.sort_values("date").copy()
    eligible["_source_index"] = eligible.index
    eligible["_year"] = eligible["date"].dt.year

    years = [
        int(y) for y in sorted(eligible["_year"].unique())
        if int(y) >= ROLLING_RESIDUAL_FIRST_VALIDATION_YEAR
    ]

    prediction_frames = []
    calibration_frames = []
    fit_count = 0

    print(f"  Eligible rated rows       : {len(eligible):,}")
    print(f"  Validation years          : {years}")
    print(f"  Rolling baseline thinking : {ROLLING_RESIDUAL_BASELINE_THINKING_MODE}")
    print(f"  Minimum train rows/fold   : {ROLLING_RESIDUAL_MIN_TRAIN_ROWS:,}")

    for year in years:
        train_fold = eligible[eligible["date"] < pd.Timestamp(year=year, month=1, day=1)].copy()
        valid_fold = eligible[eligible["_year"] == year].copy()

        if valid_fold.empty:
            continue

        if len(train_fold) < ROLLING_RESIDUAL_MIN_TRAIN_ROWS:
            print(
                f"  Skipping {year}: train rows={len(train_fold):,} "
                f"< {ROLLING_RESIDUAL_MIN_TRAIN_ROWS:,}"
            )
            continue

        print(
            f"  Rolling fold {year}: train={len(train_fold):,} "
            f"valid={len(valid_fold):,}"
        )

        clf = train_tabpfn(
            Xf(train_fold, BASELINE_FEATURES, f"rolling residual baseline train {year}"),
            train_fold["outcome"].values,
            use_thinking_mode=ROLLING_RESIDUAL_BASELINE_THINKING_MODE,
        )
        fit_count += 1

        probs = _predict_named_probabilities(
            clf,
            Xf(valid_fold, BASELINE_FEATURES, f"rolling residual baseline valid {year}"),
            f"rolling residual baseline valid {year}",
        )

        fold_out = valid_fold[[
            "_source_index", "date", "home_team", "away_team", "outcome"
        ]].reset_index(drop=True)
        fold_out = pd.concat([fold_out, probs.reset_index(drop=True)], axis=1)
        fold_out["fold_year"] = year
        prediction_frames.append(fold_out)

        # ── Second stream: v2-feature model for calibration fitting ─────────
        clf_v2 = train_tabpfn(
            Xf(train_fold, AUGMENTED_FEATURES, f"calibration v2 train {year}"),
            train_fold["outcome"].values,
            use_thinking_mode=ROLLING_RESIDUAL_BASELINE_THINKING_MODE,
        )
        fit_count += 1

        probs_v2 = _predict_named_probabilities(
            clf_v2,
            Xf(valid_fold, AUGMENTED_FEATURES, f"calibration v2 valid {year}"),
            f"calibration v2 valid {year}",
        )

        fold_v2 = valid_fold[[
            "_source_index", "date", "home_team", "away_team", "outcome",
            "neutral", "importance",
        ]].reset_index(drop=True)
        fold_v2 = pd.concat([fold_v2, probs_v2.reset_index(drop=True)], axis=1)
        fold_v2["fold_year"] = year
        calibration_frames.append(fold_v2)

    if not prediction_frames:
        raise RuntimeError(
            "No rolling residual folds produced predictions. Lower "
            "ROLLING_RESIDUAL_MIN_TRAIN_ROWS or check rating coverage."
        )

    oof = pd.concat(prediction_frames, ignore_index=True)
    oof = oof.sort_values(["date", "_source_index"]).reset_index(drop=True)
    oof.to_csv(ROLLING_RESIDUAL_PREDICTIONS_CSV, index=False)

    cal_oof = pd.concat(calibration_frames, ignore_index=True)
    cal_oof = cal_oof.sort_values(["date", "_source_index"]).reset_index(drop=True)
    cal_oof.to_csv(CALIBRATION_OOF_FEATURES_CSV, index=False)
    print(f"  Saved v2 calibration OOF   → {CALIBRATION_OOF_FEATURES_CSV}")

    print(f"  Rolling baseline fits run : {fit_count}")
    print(f"  OOF residual predictions  : {len(oof):,}")
    print(f"  Saved → {ROLLING_RESIDUAL_PREDICTIONS_CSV}")

    residual_records = []

    for _, row in oof.iterrows():
        home_win_actual = 1.0 if row["outcome"] == "home_win" else 0.0
        draw_actual = 1.0 if row["outcome"] == "draw" else 0.0
        away_win_actual = 1.0 if row["outcome"] == "away_win" else 0.0

        residual_records.append({
            "_source_index": int(row["_source_index"]),
            "date": row["date"],
            "side": "home",
            "team": row["home_team"],
            "attack_residual": home_win_actual - row["p_home_win"],
            "defense_residual": (home_win_actual + draw_actual) - (row["p_home_win"] + row["p_draw"]),
        })

        residual_records.append({
            "_source_index": int(row["_source_index"]),
            "date": row["date"],
            "side": "away",
            "team": row["away_team"],
            "attack_residual": away_win_actual - row["p_away_win"],
            "defense_residual": (away_win_actual + draw_actual) - (row["p_away_win"] + row["p_draw"]),
        })

    long = pd.DataFrame(residual_records)
    long = long.sort_values(["team", "date", "_source_index", "side"]).reset_index(drop=True)

    grouped = long.groupby("team", group_keys=False)
    long["attack_residual_bias"] = grouped["attack_residual"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    long["defense_residual_bias"] = grouped["defense_residual"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    long["latest_attack_residual_bias"] = grouped["attack_residual"].transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    long["latest_defense_residual_bias"] = grouped["defense_residual"].transform(
        lambda s: s.expanding(min_periods=1).mean()
    )

    long.to_csv(ROLLING_RESIDUAL_LONG_CSV, index=False)
    print(f"  Saved → {ROLLING_RESIDUAL_LONG_CSV}")

    home_bias = long[long["side"].eq("home")][[
        "_source_index", "attack_residual_bias", "defense_residual_bias"
    ]].rename(columns={
        "attack_residual_bias": "home_attack_bias",
        "defense_residual_bias": "home_defense_bias",
    })

    away_bias = long[long["side"].eq("away")][[
        "_source_index", "attack_residual_bias", "defense_residual_bias"
    ]].rename(columns={
        "attack_residual_bias": "away_attack_bias",
        "defense_residual_bias": "away_defense_bias",
    })

    bias_by_match = home_bias.merge(away_bias, on="_source_index", how="outer")

    for _, row in bias_by_match.iterrows():
        idx = int(row["_source_index"])
        feats.loc[idx, "home_attack_bias"] = row["home_attack_bias"]
        feats.loc[idx, "home_defense_bias"] = row["home_defense_bias"]
        feats.loc[idx, "away_attack_bias"] = row["away_attack_bias"]
        feats.loc[idx, "away_defense_bias"] = row["away_defense_bias"]

    latest = (
        long.sort_values(["team", "date", "_source_index", "side"])
            .groupby("team")
            .tail(1)[[
                "team", "date",
                "latest_attack_residual_bias",
                "latest_defense_residual_bias",
            ]]
            .rename(columns={
                "latest_attack_residual_bias": "attack_residual_bias",
                "latest_defense_residual_bias": "defense_residual_bias",
                "date": "latest_residual_date",
            })
            .reset_index(drop=True)
    )
    latest.to_csv(ROLLING_RESIDUAL_LATEST_CSV, index=False)
    print(f"  Saved → {ROLLING_RESIDUAL_LATEST_CSV}")

    complete_v3_rows = feats[AUGMENTED_V3_FEATURES].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    played_complete_v3 = complete_v3_rows & feats["outcome"].notna() & (feats["date"] >= TRAIN_START)

    print("\n  Rolling residual feature coverage:")
    print(f"    played rows from TRAIN_START : {int(((feats['outcome'].notna()) & (feats['date'] >= TRAIN_START)).sum()):,}")
    print(f"    v3-complete played rows      : {int(played_complete_v3.sum()):,}")
    for col in BIAS_COLS:
        print(f"    {col:>18s} non-missing: {int(feats[col].notna().sum()):,}")

    return feats, latest


# ─────────────────────────────────────────────────────────────────────────────
# 4b.  Engineered-feature exports
# ─────────────────────────────────────────────────────────────────────────────

def _ordered_existing_columns(df, columns):
    """
    Return columns that exist in df, preserving order and removing duplicates.

    This keeps feature exports stable while avoiding accidental duplicate columns
    when metadata columns such as neutral/importance also appear in feature sets.
    """
    out = []
    seen = set()
    for col in columns:
        if col in df.columns and col not in seen:
            out.append(col)
            seen.add(col)
    return out


def _feature_source(feature_name):
    """Describe the source of one feature for the feature dictionary CSV."""
    if feature_name in BASELINE_FEATURES:
        return "martj42_history_elo_form_h2h"
    if feature_name in AUG_FEATURES:
        return "sofascore_first_layer_team_ratings"
    if feature_name in BIAS_FEATURES:
        return "rolling_oof_baseline_residual_bias"
    if feature_name in STYLE_FEATURES:
        return "time_aware_team_style_keyword_triplet"
    if feature_name in CHAR_FEATURES:
        return "sofascore_team_characteristics"
    return "unknown"


def export_feature_set_dictionary():
    """
    Write a compact dictionary of all model feature sets used by this script.

    The baseline feature set is the engineered-feature set inherited from the
    attached predict.py script: ELO, recent form, head-to-head, rest, neutral,
    and tournament importance. v2 adds first-layer attack/defense ratings. v3
    adds rolling out-of-fold residual-bias features.
    """
    rows = []
    feature_sets = {
        "baseline_v1": BASELINE_FEATURES,
        "ratings_v2": AUGMENTED_FEATURES,
        "rolling_residual_v3": AUGMENTED_V3_FEATURES,
    }
    if AUGMENTED_V4_FEATURES:
        feature_sets["characteristics_v4"] = AUGMENTED_V4_FEATURES

    for set_name, features in feature_sets.items():
        for position, feature_name in enumerate(features, start=1):
            rows.append({
                "feature_set": set_name,
                "position": position,
                "feature_name": feature_name,
                "source": _feature_source(feature_name),
            })

    dictionary = pd.DataFrame(rows)
    dictionary.to_csv(FEATURE_SET_DICTIONARY_CSV, index=False)
    print(f"  Saved feature dictionary → {FEATURE_SET_DICTIONARY_CSV}")
    return dictionary


def build_round16_engineered_feature_frame(ratings, state_snapshot, slot_candidate_pool,
                                           style_eras_by_team=None, style_encoder=None):
    """
    Build the Round-of-16 feature rows before prediction.

    This uses the same fixture feature path as precompute_round16_probability_cache,
    but it does not call TabPFN. It exists so we can save the exact feature matrix
    that will be fed into the final v3 model for Round-of-16 inference.
    """
    matchups = collect_possible_round16_matchups(slot_candidate_pool)
    fixture_frames = []

    for matchup in matchups.itertuples(index=False):
        fixture = make_single_fixture_features_fast(
            ratings=ratings,
            state_snapshot=state_snapshot,
            home_team=matchup.home_team,
            away_team=matchup.away_team,
            match_date=matchup.date,
            characteristics=None,
            stat_cols=None,
            style_eras_by_team=style_eras_by_team,
            style_encoder=style_encoder,
        )

        fixture.insert(0, "round", matchup.round)
        fixture.insert(0, "match_id", matchup.match_id)
        fixture.insert(2, "date", matchup.date)
        fixture_frames.append(fixture)

    if not fixture_frames:
        raise ValueError("No Round-of-16 fixture rows were generated.")

    return pd.concat(fixture_frames, ignore_index=True)


def export_engineered_feature_tables(feats, ratings, state_snapshot, slot_candidate_pool,
                                     style_eras_by_team=None, style_encoder=None):
    """
    Save the important intermediate feature tables used by the final model.

    Outputs:
    1. martj42_baseline_engineered_features.csv
       All historical martj42 rows with the original predict.py engineered
       baseline features.

    2. final_v3_training_engineered_features.csv
       The complete pre-WC training rows consumed by the final rolling-residual
       v3 model: baseline features + first-layer ratings + rolling residual bias.

    3. round16_engineered_features_rolling_residual_v3.csv
       The actual Round-of-16 feature rows consumed at prediction time.

    4. feature_set_dictionary.csv
       Human-readable feature list and provenance by model version.
    """
    metadata_cols = [
        "date", "home_team", "away_team", "home_score", "away_score",
        "outcome", "tournament", "city", "country", "neutral", "importance",
        "home_style_triplet", "away_style_triplet",
    ]

    baseline_cols = _ordered_existing_columns(feats, metadata_cols + BASELINE_FEATURES)
    baseline_export = feats[baseline_cols].copy()
    baseline_export.to_csv(BASELINE_ENGINEERED_FEATURES_CSV, index=False)
    print(
        f"  Saved baseline engineered features → {BASELINE_ENGINEERED_FEATURES_CSV} "
        f"({baseline_export.shape[0]:,} rows × {baseline_export.shape[1]:,} cols)"
    )

    wc_start = pd.Timestamp("2026-06-11")
    played_pre_wc = feats[
        feats["outcome"].notna()
        & (feats["date"] >= TRAIN_START)
        & (feats["date"] < wc_start)
    ].copy()

    complete_v3_mask = (
        played_pre_wc[AUGMENTED_V3_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
    final_train_export = played_pre_wc.loc[complete_v3_mask].copy()
    final_train_cols = _ordered_existing_columns(
        final_train_export,
        metadata_cols + AUGMENTED_V3_FEATURES,
    )
    final_train_export = final_train_export[final_train_cols]
    final_train_export.to_csv(FINAL_V3_TRAINING_FEATURES_CSV, index=False)
    print(
        f"  Saved final v3 training features → {FINAL_V3_TRAINING_FEATURES_CSV} "
        f"({final_train_export.shape[0]:,} rows × {final_train_export.shape[1]:,} cols)"
    )

    round16_export = build_round16_engineered_feature_frame(
        ratings=ratings,
        state_snapshot=state_snapshot,
        slot_candidate_pool=slot_candidate_pool,
        style_eras_by_team=style_eras_by_team,
        style_encoder=style_encoder,
    )
    # Include raw human-readable style triplets in the diagnostic Round-of-16
    # feature export, but keep only the numeric *_style_triplet_code columns in
    # AUGMENTED_V3_FEATURES for model fitting/prediction.
    #
    # This makes the CSV auditable:
    #   home_style_triplet       = e.g. "creative|structured|wide"
    #   home_style_triplet_code  = numeric category sent to TabPFN
    #
    # _ordered_existing_columns(...) keeps this safe when style keywords are
    # disabled: missing audit columns are simply omitted from the export.
    round16_style_audit_cols = [
        "home_style_triplet",
        "away_style_triplet",
    ]

    round16_cols = _ordered_existing_columns(
        round16_export,
        (
            ["match_id", "round", "date", "home_team", "away_team"]
            + round16_style_audit_cols
            + AUGMENTED_V3_FEATURES
        ),
    )
    round16_export = round16_export[round16_cols]
    round16_export.to_csv(ROUND16_ENGINEERED_FEATURES_CSV, index=False)
    print(
        f"  Saved Round-of-16 engineered features → {ROUND16_ENGINEERED_FEATURES_CSV} "
        f"({round16_export.shape[0]:,} rows × {round16_export.shape[1]:,} cols)"
    )

    export_feature_set_dictionary()

    return {
        "baseline": baseline_export,
        "final_v3_train": final_train_export,
        "round16": round16_export,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Comparison: four models on WC 2026 group stage
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(feats, ratings):
    WC_START = pd.Timestamp("2026-06-11")

    played = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)].copy()
    pre_wc = played[played["date"] < WC_START]
    test   = played[played["date"] >= WC_START].copy()

    # Strict v3 policy: do not fill residual-bias NaNs with 0.0.
    # Missing residual-bias values are kept as NaN so v3 can be trained/evaluated
    # only on rows where the augmented bias signal is genuinely available.
    pre_wc = pre_wc.copy()
    test   = test.copy()

    # The shared training set is defined by v2's requirements: both teams must
    # have attack/defense ratings. All models train on exactly these rows
    # so comparisons are fair.
    train = require_complete_features(
        pre_wc, AUGMENTED_FEATURES, "shared train (v2 rows)"
    )

    print(f"\n  Train : {len(train)} matches (up to {train['date'].max().date()})")
    print(f"  Test  : {len(test)} WC 2026 matches\n")

    def yf(df): return df["outcome"].values

    test_base = require_complete_features(test, BASELINE_FEATURES,     "v1 test")
    test_aug  = require_complete_features(test, AUGMENTED_FEATURES,    "v2 test")
    test_v3   = require_complete_features(test, AUGMENTED_V3_FEATURES, "v3 test")
    test_v4   = (require_complete_features(test, AUGMENTED_V4_FEATURES, "v4 test")
                 if AUGMENTED_V4_FEATURES else None)

    results = test[["date", "home_team", "away_team", "outcome"]].copy()
    metrics = {}
    model_artifacts = {}

    # ── v1: Baseline ──────────────────────────────────────────────────────────
    print("── v1  Baseline (ELO + form + H2H) ──")
    clf1 = train_tabpfn(Xf(train, BASELINE_FEATURES, "v1 train"), yf(train))
    acc, ll, pred, _, _ = evaluate(clf1, Xf(test_base, BASELINE_FEATURES, "v1 test"), yf(test_base), "v1 Baseline")
    results.loc[test_base.index, "v1_pred"]    = pred
    results.loc[test_base.index, "v1_correct"] = pred == test_base["outcome"].values
    metrics["v1"] = (acc, ll)
    model_artifacts["v1"] = {
        "label": "v1 Baseline", "clf": clf1, "features": BASELINE_FEATURES,
        "accuracy": acc, "log_loss": ll, "n_eval_rows": len(test_base),
    }

    # ── v2: + attack/defense ratings ──────────────────────────────────────────
    print("\n── v2  + attack/defense ratings ──")
    clf2 = train_tabpfn(Xf(train, AUGMENTED_FEATURES, "v2 train"), yf(train))
    acc, ll, pred, _, _ = evaluate(clf2, Xf(test_aug, AUGMENTED_FEATURES, "v2 test"), yf(test_aug), "v2 + ratings")
    results.loc[test_aug.index, "v2_pred"]    = pred
    results.loc[test_aug.index, "v2_correct"] = pred == test_aug["outcome"].values
    metrics["v2"] = (acc, ll)
    model_artifacts["v2"] = {
        "label": "v2 + ratings", "clf": clf2, "features": AUGMENTED_FEATURES,
        "accuracy": acc, "log_loss": ll, "n_eval_rows": len(test_aug),
    }

    # ── v3: + ratings + WC residual bias ──────────────────────────────────────
    print("\n── v3  + ratings + WC residual bias ──")
    train_v3 = require_complete_features(
        train,
        AUGMENTED_V3_FEATURES,
        "v3 train rolling residual augmented rows",
    )
    clf3 = train_tabpfn(Xf(train_v3, AUGMENTED_V3_FEATURES, "v3 rolling-residual train"), yf(train_v3))
    acc, ll, pred, _, _ = evaluate(clf3, Xf(test_v3, AUGMENTED_V3_FEATURES, "v3 rolling-residual test"), yf(test_v3), "v3 strict + bias")
    results.loc[test_v3.index, "v3_pred"]    = pred
    results.loc[test_v3.index, "v3_correct"] = pred == test_v3["outcome"].values
    metrics["v3"] = (acc, ll)
    model_artifacts["v3"] = {
        "label": "v3 + bias", "clf": clf3, "features": AUGMENTED_V3_FEATURES,
        "accuracy": acc, "log_loss": ll, "n_eval_rows": len(test_v3),
    }

    # ── v4: + full team characteristic averages ────────────────────────────────
    if AUGMENTED_V4_FEATURES and test_v4 is not None:
        print("\n── v4  + team characteristic averages ──")
        train_v4 = require_complete_features(train, AUGMENTED_V4_FEATURES, "v4 train")
        clf4 = train_tabpfn(Xf(train_v4, AUGMENTED_V4_FEATURES, "v4 train"), yf(train_v4))
        acc, ll, pred, _, _ = evaluate(clf4, Xf(test_v4, AUGMENTED_V4_FEATURES, "v4 test"), yf(test_v4), "v4 + chars")
        results.loc[test_v4.index, "v4_pred"]    = pred
        results.loc[test_v4.index, "v4_correct"] = pred == test_v4["outcome"].values
        metrics["v4"] = (acc, ll)
        model_artifacts["v4"] = {
            "label": "v4 + chars", "clf": clf4, "features": AUGMENTED_V4_FEATURES,
            "accuracy": acc, "log_loss": ll, "n_eval_rows": len(test_v4),
        }

    # ── Deltas vs baseline ────────────────────────────────────────────────────
    print("\n  Model comparison vs v1 baseline:")
    for ver in [v for v in ["v2", "v3", "v4"] if v in metrics]:
        da = metrics[ver][0] - metrics["v1"][0]
        dl = metrics[ver][1] - metrics["v1"][1]
        print(f"  {ver}: Δ accuracy={da:+.1%} ({'better' if da > 0 else 'worse'})  "
              f"Δ log-loss={dl:+.4f} ({'better' if dl < 0 else 'worse'})")

    # ── Matches where any model differed ──────────────────────────────────────
    pred_cols = [c for c in ["v1_pred","v2_pred","v3_pred","v4_pred"] if c in results.columns]
    complete_rows = results[pred_cols].notna().all(axis=1)
    changed = results[complete_rows & results[pred_cols].nunique(axis=1).gt(1)]
    print(f"\n  Predictions changed by augmentation: {len(changed)} / {len(results)}")
    if len(changed):
        corr_cols = [c.replace("pred", "correct") for c in pred_cols if c.replace("pred","correct") in results.columns]
        print(changed[["date","home_team","away_team","outcome"] + pred_cols + corr_cols].to_string(index=False))

    return results, metrics, model_artifacts


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Predict upcoming fixtures
# ─────────────────────────────────────────────────────────────────────────────

def predict_upcoming(feats, model_artifacts):
    TODAY  = pd.Timestamp.now().normalize()
    played = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)]
    future = feats[feats["home_score"].isna() & (feats["date"] > TODAY)].sort_values("date")

    if future.empty:
        print("  No upcoming fixtures found.")
        return None

    clf1 = model_artifacts["v1"]["clf"]
    clf2 = model_artifacts["v2"]["clf"]
    clf3 = model_artifacts["v3"]["clf"]
    clf4 = model_artifacts.get("v4", {}).get("clf")

    def get_probs(clf, X, label):
        """Returns probability-argmax predictions and named probability columns."""
        proba = clf.predict_proba(X)
        if not np.isfinite(proba).all():
            raise ValueError(f"{label}: predict_proba returned NaN or Inf values.")
        probability_classes = resolve_probability_classes(clf=clf, proba=proba, label=label)
        pred = probability_classes[proba.argmax(axis=1)]
        cols = {c: proba[:, i] for i, c in enumerate(probability_classes)}
        return pred, cols

    future_all = future.copy()
    # Strict v3 policy: do not fill residual-bias NaNs with 0.0 for future rows.
    # Missing augmented features should fail through require_complete_features(...)
    # rather than silently becoming "neutral" bias values.
    # Filter to rows where all features across all active models are available
    filter_features = AUGMENTED_V4_FEATURES if (AUGMENTED_V4_FEATURES and clf4) else AUGMENTED_V3_FEATURES
    future_all = require_complete_features(future_all, filter_features, "upcoming shared")

    pred1, c1 = get_probs(clf1, Xf(future_all, BASELINE_FEATURES,     "v1 upcoming"), "v1 upcoming")
    pred2, c2 = get_probs(clf2, Xf(future_all, AUGMENTED_FEATURES,    "v2 upcoming"), "v2 upcoming")
    pred3, c3 = get_probs(clf3, Xf(future_all, AUGMENTED_V3_FEATURES, "v3 upcoming"), "v3 upcoming")

    pairs = [("v1", pred1, c1), ("v2", pred2, c2), ("v3", pred3, c3)]
    if clf4 and AUGMENTED_V4_FEATURES:
        pred4, c4 = get_probs(clf4, Xf(future_all, AUGMENTED_V4_FEATURES, "v4 upcoming"), "v4 upcoming")
        pairs.append(("v4", pred4, c4))

    out = future_all[["date", "home_team", "away_team"]].copy()
    for ver, pred, cols in pairs:
        out[f"{ver}_predicted"]  = pred
        out[f"{ver}_p_home_win"] = cols.get("home_win", np.nan)
        out[f"{ver}_p_draw"]     = cols.get("draw",     np.nan)
        out[f"{ver}_p_away_win"] = cols.get("away_win", np.nan)

    print(f"\n  Upcoming fixtures ({len(out)}):")
    ver_keys = [p[0] for p in pairs]
    for r in out.itertuples():
        preds = [getattr(r, f"{v}_predicted") for v in ver_keys]
        agree = "✓" if len(set(preds)) == 1 else "≠"
        pred_str = "  ".join(f"{v}:{getattr(r, f'{v}_predicted'):<9}" for v in ver_keys)
        print(f"  {r.date.date()}  {r.home_team:>25} vs {r.away_team:<25}  {pred_str}{agree}")

    out.to_csv(OUTPUT_DIR / "upcoming_predictions.csv", index=False)
    print("\n  Saved → upcoming_predictions.csv")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Knockout bracket projection
# ─────────────────────────────────────────────────────────────────────────────

def choose_best_model_by_log_loss(model_artifacts):
    """Selects v3 explicitly for bracket projection."""
    if "v3" not in model_artifacts:
        raise KeyError("model_artifacts must contain a fitted v3 model for bracket projection.")

    best_key = "v3"
    best = model_artifacts[best_key]

    print("\n  Bracket model:")
    print(f"  {best_key}: {best['label']}  "
          f"accuracy={best['accuracy']:.1%}  "
          f"log-loss={best['log_loss']:.4f}  "
          f"n_eval={best['n_eval_rows']}")

    return best_key, best


def make_single_fixture_features(df, ratings, home_team, away_team, match_date):
    """
    Builds one hypothetical fixture row using the same feature pipeline as the
    training/evaluation script.

    Important design choice:
    - We only use played matches from df.
    - We append exactly one future knockout fixture with missing scores.
    - build_features computes pre-match features before any result update.

    This avoids contaminating the feature state with simulated future winners.
    """
    played_df = df[df["home_score"].notna() & df["away_score"].notna()].copy()

    fixture = pd.DataFrame([{
        "date": pd.Timestamp(match_date),
        "home_team": home_team,
        "away_team": away_team,
        "home_score": np.nan,
        "away_score": np.nan,
        "tournament": "FIFA World Cup",
        "city": "",
        "country": "",
        "neutral": 1,
    }])

    tmp = pd.concat([played_df, fixture], ignore_index=True, sort=False)
    tmp = tmp.sort_values("date").reset_index(drop=True)

    tmp["date"]       = pd.to_datetime(tmp["date"])
    tmp["neutral"]    = tmp["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    tmp["home_score"] = pd.to_numeric(tmp["home_score"], errors="coerce")
    tmp["away_score"] = pd.to_numeric(tmp["away_score"], errors="coerce")
    tmp["outcome"]    = np.select(
        [tmp["home_score"] > tmp["away_score"], tmp["home_score"] < tmp["away_score"]],
        ["home_win", "away_win"], default="draw")
    tmp.loc[tmp["home_score"].isna(), "outcome"] = np.nan
    tmp["importance"] = tmp["tournament"].apply(importance)

    tmp_feats, _ = build_features(tmp)
    tmp_feats = join_ratings(tmp_feats, ratings)

    return tmp_feats.iloc[[-1]].copy()


def get_class_probability(proba_row, classes, class_name):
    """
    Safely extracts a class probability from a classifier output row.

    TabPFN should have all three classes here, but this helper avoids brittle
    index assumptions.
    """
    if class_name not in classes:
        return 0.0

    class_idx = list(classes).index(class_name)
    return float(proba_row[class_idx])


def convert_match_probs_to_advancement_probs(p_home_win, p_draw, p_away_win,
                                             draw_policy=BRACKET_DRAW_POLICY):
    """
    Converts 3-way match probabilities into 2-way knockout advancement
    probabilities.

    The classifier predicts:
    - home_win
    - draw
    - away_win

    The bracket needs:
    - home advances
    - away advances
    """
    if draw_policy == "split_evenly":
        p_home_adv = p_home_win + 0.5 * p_draw
        p_away_adv = p_away_win + 0.5 * p_draw

    elif draw_policy == "renormalize_no_draw":
        denom = p_home_win + p_away_win

        if denom <= 0:
            p_home_adv = 0.5
            p_away_adv = 0.5
        else:
            p_home_adv = p_home_win / denom
            p_away_adv = p_away_win / denom

    else:
        raise ValueError(
            f"Unknown draw_policy={draw_policy!r}. "
            "Use 'split_evenly' or 'renormalize_no_draw'."
        )

    denom = p_home_adv + p_away_adv

    if denom <= 0:
        return 0.5, 0.5

    return p_home_adv / denom, p_away_adv / denom


def apply_sampling_temperature(p_home, p_away, temperature=BRACKET_SAMPLING_TEMPERATURE):
    """
    Applies temperature only to the sampling probabilities.

    The bracket's reported log probability still uses the original model
    probabilities. This lets us explore diverse brackets without changing the
    scoring rule.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    if temperature == 1.0:
        return p_home, p_away

    logits = np.log(np.array([p_home, p_away], dtype=float).clip(1e-12, 1.0))
    logits = logits / temperature
    logits = logits - logits.max()

    probs = np.exp(logits)
    probs = probs / probs.sum()

    return float(probs[0]), float(probs[1])


def make_probability_cache_key(model_artifact, home_team, away_team, match_date):
    """
    Centralized cache key for knockout matchup probabilities.

    The key intentionally includes date because rest-day features are date-dependent.
    """
    return (
        model_artifact["label"],
        tuple(model_artifact["features"]),
        home_team,
        away_team,
        str(pd.Timestamp(match_date).date()),
        BRACKET_DRAW_POLICY,
    )


def predict_knockout_match(model_artifact, df, ratings, state_snapshot, home_team, away_team,
                           match_date, prob_cache, characteristics=None, stat_cols=None):
    """
    Returns cached knockout advancement probabilities for one matchup.

    Bracket simulation must not call TabPFN directly. All possible knockout
    matchups are predicted in one batched predict_proba call before Monte Carlo
    starts. If this function misses the cache, the precomputation step did not
    cover a matchup that the simulator can generate.
    """
    cache_key = make_probability_cache_key(
        model_artifact=model_artifact,
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
    )

    if cache_key not in prob_cache:
        raise KeyError(
            "Missing precomputed probability for bracket matchup: "
            f"{home_team} vs {away_team} on {pd.Timestamp(match_date).date()}. "
            "This means collect_possible_bracket_matchups() did not enumerate "
            "a matchup that simulate_one_bracket() generated."
        )

    return prob_cache[cache_key]

def sample_match_winner(match_probs, rng):
    """
    Samples one knockout winner using model-derived advancement probabilities.

    Returns:
    - winner
    - loser
    - chosen model probability, before temperature
    - chosen sampling probability, after temperature
    """
    home_team = match_probs["home_team"]
    away_team = match_probs["away_team"]

    p_home_model = match_probs["p_home_adv"]
    p_away_model = match_probs["p_away_adv"]

    p_home_sample, p_away_sample = apply_sampling_temperature(
        p_home=p_home_model,
        p_away=p_away_model,
        temperature=BRACKET_SAMPLING_TEMPERATURE,
    )

    home_wins = rng.random() < p_home_sample

    if home_wins:
        return home_team, away_team, p_home_model, p_home_sample

    return away_team, home_team, p_away_model, p_away_sample


def record_bracket_match(records, bracket_rank, round_name, match_id, match_probs,
                         winner, loser, chosen_model_prob, chosen_sample_prob):
    """
    Appends a single match result to the long-form bracket records list.
    """
    records.append({
        "bracket_rank": bracket_rank,
        "round": round_name,
        "match_id": match_id,
        "home_team": match_probs["home_team"],
        "away_team": match_probs["away_team"],
        "p_home_win_90": match_probs["p_home_win_90"],
        "p_draw_90": match_probs["p_draw_90"],
        "p_away_win_90": match_probs["p_away_win_90"],
        "p_home_adv": match_probs["p_home_adv"],
        "p_away_adv": match_probs["p_away_adv"],
        "winner": winner,
        "loser": loser,
        "chosen_model_prob": chosen_model_prob,
        "chosen_sample_prob": chosen_sample_prob,
    })


def build_full_bracket_summary_columns(match_records):
    """
    Converts one simulated bracket into wide, full-bracket summary columns.

    The compact summary columns only show champion / runner-up / third / fourth.
    These wide columns preserve the entire populated bracket for each top-ranked
    Monte Carlo bracket: every match's home side, away side, winner, loser, and
    chosen model probability.
    """
    row = {}
    path_parts = []

    for record in match_records:
        match_id = record["match_id"]
        match_probs = record["match_probs"]

        row[f"{match_id}_round"] = record["round"]
        row[f"{match_id}_home"] = match_probs["home_team"]
        row[f"{match_id}_away"] = match_probs["away_team"]
        row[f"{match_id}_winner"] = record["winner"]
        row[f"{match_id}_loser"] = record["loser"]
        row[f"{match_id}_chosen_model_prob"] = record["chosen_model_prob"]
        row[f"{match_id}_p_home_adv"] = match_probs["p_home_adv"]
        row[f"{match_id}_p_away_adv"] = match_probs["p_away_adv"]

        path_parts.append(
            f"{match_id}: {match_probs['home_team']} vs {match_probs['away_team']} -> {record['winner']}"
        )

    row["full_bracket_path"] = " | ".join(path_parts)
    return row



def get_explicit_r32_teams():
    """
    Returns teams already explicitly placed in the Round of 16.

    Placeholder slots such as 2J, 2K, 1K, 3IK, 3AIJ, and 3GJ are excluded.
    """
    return {
        team
        for match in BRACKET_R16
        for team in (match["home_team"], match["away_team"])
        if team not in BRACKET_SLOT_CANDIDATES
    }


def build_slot_candidate_pool(ratings):
    """
    Builds the actual candidate pool used for unresolved bracket slots.

    A candidate is excluded if:
    - it is already explicitly present in the Round of 16, or
    - it has no attack/defense rating, which means v3 cannot score matchups
      involving that team without introducing missing augmented features.

    The global BRACKET_SLOT_CANDIDATES remains the editable source of truth;
    this function just makes the runtime pool internally consistent.
    """
    explicit_teams = get_explicit_r32_teams()
    pool = {}
    excluded_rows = []

    for slot, candidates in BRACKET_SLOT_CANDIDATES.items():
        clean_candidates = []

        for team in candidates:
            reason = None
            if team in explicit_teams:
                reason = "already_explicit_in_r32"
            elif team not in ratings:
                reason = "missing_team_ratings"

            if reason is None:
                clean_candidates.append(team)
            else:
                excluded_rows.append({"slot": slot, "team": team, "reason": reason})

        # Deduplicate while preserving order.
        clean_candidates = list(dict.fromkeys(clean_candidates))

        if not clean_candidates:
            raise ValueError(
                f"No usable candidates remain for unresolved slot {slot!r}. "
                "Edit BRACKET_SLOT_CANDIDATES or make sure team_ratings.csv "
                "contains the candidate teams."
            )

        pool[slot] = clean_candidates

    if excluded_rows:
        excluded_df = pd.DataFrame(excluded_rows)
        print("\n  Excluded unresolved-slot candidates:")
        print(excluded_df.to_string(index=False))

    print("\n  Runtime unresolved-slot candidate pool:")
    for slot, candidates in pool.items():
        print(f"    {slot}: {', '.join(candidates)}")

    return pool


def iter_valid_slot_assignments(slot_candidate_pool):
    """
    Enumerates all valid unresolved-slot assignments.

    A valid assignment does not reuse a team across unresolved slots and does
    not duplicate a team already explicitly present in the Round of 16.
    """
    explicit_teams = get_explicit_r32_teams()
    slot_names = list(slot_candidate_pool)

    for combo in product(*(slot_candidate_pool[slot] for slot in slot_names)):
        assignment = dict(zip(slot_names, combo))
        chosen = list(assignment.values())

        if len(set(chosen)) != len(chosen):
            continue

        if any(team in explicit_teams for team in chosen):
            continue

        yield assignment


def sample_bracket_slot_assignment(rng, slot_candidate_pool):
    """
    Samples one valid assignment for unresolved bracket slots.

    Slot probabilities are uniform by default because this script is not
    simulating the remaining group-stage games. The important point is that the
    missing future bracket parts remain uncertain; they are not treated as old
    known wins.
    """
    valid_assignments = list(iter_valid_slot_assignments(slot_candidate_pool))

    if not valid_assignments:
        raise RuntimeError(
            "Could not build any valid unresolved-slot assignment. "
            "Check BRACKET_SLOT_CANDIDATES for impossible overlaps."
        )

    idx = int(rng.integers(0, len(valid_assignments)))
    return valid_assignments[idx]


def resolve_bracket_team(team, slot_assignment):
    """Returns a real team name for either an explicit team or a placeholder slot."""
    return slot_assignment.get(team, team)


def resolve_r16_match(match, slot_assignment):
    """Resolves placeholder teams in one Round-of-16 match dictionary."""
    resolved = dict(match)
    resolved["home_team"] = resolve_bracket_team(match["home_team"], slot_assignment)
    resolved["away_team"] = resolve_bracket_team(match["away_team"], slot_assignment)
    resolved["home_slot"] = match["home_team"]
    resolved["away_slot"] = match["away_team"]
    return resolved


def collect_possible_bracket_matchups(slot_candidate_pool):
    """
    Collects every matchup/date that can be requested by the bracket simulator.

    This intentionally happens before Monte Carlo. The model can then score all
    possible requested matchups in one batched predict_proba call, and the Monte
    Carlo loop only reads probabilities from a dictionary.
    """
    valid_assignments = list(iter_valid_slot_assignments(slot_candidate_pool))

    if not valid_assignments:
        raise RuntimeError("No valid unresolved-slot assignments were found.")

    matchup_rows = []
    possible_entrants_by_match = {}

    def add_matchup(match_id, round_name, match_date, home_team, away_team):
        if home_team == away_team:
            return

        matchup_rows.append({
            "match_id": match_id,
            "round": round_name,
            "date": match_date,
            "home_team": home_team,
            "away_team": away_team,
        })

    # Round of 16: enumerate concrete matchups under all valid future slot assignments.
    for match in BRACKET_R16:
        entrants = set()

        for assignment in valid_assignments:
            resolved = resolve_r16_match(match, assignment)
            home_team = resolved["home_team"]
            away_team = resolved["away_team"]

            add_matchup(
                match_id=match["match_id"],
                round_name="Round of 16",
                match_date=match["date"],
                home_team=home_team,
                away_team=away_team,
            )

            entrants.add(home_team)
            entrants.add(away_team)

        possible_entrants_by_match[match["match_id"]] = entrants

    # Derived rounds: the home side is the winner of source_a and the away side
    # is the winner of source_b. Any entrant to a source match can be its winner.
    for round_name, round_matches in BRACKET_DERIVED_ROUNDS.items():
        for match in round_matches:
            source_a, source_b = match["from"]
            home_candidates = possible_entrants_by_match[source_a]
            away_candidates = possible_entrants_by_match[source_b]

            for home_team in sorted(home_candidates):
                for away_team in sorted(away_candidates):
                    add_matchup(
                        match_id=match["match_id"],
                        round_name=round_name,
                        match_date=match["date"],
                        home_team=home_team,
                        away_team=away_team,
                    )

            possible_entrants_by_match[match["match_id"]] = set(home_candidates) | set(away_candidates)

    # Third-place match: semifinal losers can be any semifinal entrants.
    third_source_a, third_source_b = BRACKET_THIRD_PLACE["from_losers"]
    home_candidates = possible_entrants_by_match[third_source_a]
    away_candidates = possible_entrants_by_match[third_source_b]

    for home_team in sorted(home_candidates):
        for away_team in sorted(away_candidates):
            add_matchup(
                match_id=BRACKET_THIRD_PLACE["match_id"],
                round_name="Third place",
                match_date=BRACKET_THIRD_PLACE["date"],
                home_team=home_team,
                away_team=away_team,
            )

    matchups = pd.DataFrame(matchup_rows).drop_duplicates().reset_index(drop=True)

    print(f"\n  Valid unresolved-slot assignments: {len(valid_assignments):,}")
    print(f"  Unique bracket matchup/date rows : {len(matchups):,}")

    return matchups



def collect_possible_round16_matchups(slot_candidate_pool):
    """
    Builds only the possible Round-of-16 matchups required by the competition.

    This intentionally does not enumerate later knockout rounds and does not run
    Monte Carlo. Each unresolved placeholder is expanded into its runtime
    candidate list, producing the smallest matchup table needed for one batched
    predict_proba call.
    """
    rows = []

    for match in BRACKET_R16:
        home_candidates = (
            slot_candidate_pool[match["home_team"]]
            if is_unresolved_slot_name(match["home_team"])
            else [match["home_team"]]
        )
        away_candidates = (
            slot_candidate_pool[match["away_team"]]
            if is_unresolved_slot_name(match["away_team"])
            else [match["away_team"]]
        )

        for home_team in home_candidates:
            for away_team in away_candidates:
                if home_team == away_team:
                    continue

                rows.append({
                    "match_id": match["match_id"],
                    "round": "Round of 16",
                    "date": match["date"],
                    "home_team": home_team,
                    "away_team": away_team,
                })

    matchups = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)

    print(f"\n  Unique Round-of-16 matchup/date rows: {len(matchups):,}")
    return matchups

def build_round16_candidate_feature_export(matchups, fixture_feats, model_feature_cols):
    """
    Build a row-aligned Round-of-16 candidate feature export.

    This table is meant for post-hoc calibration and diagnostics. It is kept
    separate from the probability CSV so the probability/submission artifacts
    remain clean and competition-compatible.

    Row alignment:
    - Row i in this feature export corresponds to row i in the candidate
      probability table produced from the same `matchups` dataframe.
    - The identity columns come from `matchups`.
    - The model/calibration features come from `fixture_feats`, which is the
      exact dataframe used to build the TabPFN prediction matrix.

    Parameters
    ----------
    matchups:
        DataFrame with match_id, round, date, home_team, away_team.

    fixture_feats:
        DataFrame of engineered feature rows generated by
        make_single_fixture_features_fast(...).

    model_feature_cols:
        Feature names actually consumed by the selected model artifact.
    """
    if len(matchups) != len(fixture_feats):
        raise ValueError(
            "Cannot build aligned candidate feature export because row counts differ: "
            f"matchups={len(matchups)}, fixture_feats={len(fixture_feats)}"
        )

    identity_cols = ["match_id", "round", "date", "home_team", "away_team"]

    feature_cols = [
        col
        for col in model_feature_cols
        if col in fixture_feats.columns and col not in identity_cols
    ]

    audit_cols = [
        col
        for col in [
            "elo_diff",
            "home_elo",
            "away_elo",
            "home_attack_rating",
            "home_defense_rating",
            "away_attack_rating",
            "away_defense_rating",
            "home_attack_bias",
            "home_defense_bias",
            "away_attack_bias",
            "away_defense_bias",
        ]
        if col in fixture_feats.columns and col not in feature_cols
    ]

    out = matchups[identity_cols].reset_index(drop=True).copy()
    feature_frame = fixture_feats[audit_cols + feature_cols].reset_index(drop=True)

    out = pd.concat([out, feature_frame], axis=1)

    return out

def precompute_round16_probability_cache(model_artifact, ratings, state_snapshot,
                                         slot_candidate_pool,
                                         characteristics=None, stat_cols=None,
                                         style_eras_by_team=None, style_encoder=None):
    """
    Computes all possible Round-of-16 matchup probabilities in one predict_proba call.

    The returned cache is enough to write the competition CSV once every slot is
    concrete. No bracket simulation and no later-round prediction is performed.

    Style keyword handling:
    - Historical rows get style codes through join_style_keywords(...).
    - Future Round-of-16 rows are synthesized here, so style objects must be
      passed through to make_single_fixture_features_fast(...).
    - If STYLE_FEATURES is empty, the extra arguments are harmless and ignored.
    """
    matchups = collect_possible_round16_matchups(slot_candidate_pool)

    fixture_frames = []
    cache_keys = []

    for row in matchups.itertuples(index=False):
        fixture = make_single_fixture_features_fast(
            ratings=ratings,
            state_snapshot=state_snapshot,
            home_team=row.home_team,
            away_team=row.away_team,
            match_date=row.date,
            characteristics=characteristics,
            stat_cols=stat_cols,
            style_eras_by_team=style_eras_by_team,
            style_encoder=style_encoder,
        )

        fixture_frames.append(fixture)
        cache_keys.append(make_probability_cache_key(
            model_artifact=model_artifact,
            home_team=row.home_team,
            away_team=row.away_team,
            match_date=row.date,
        ))

    fixture_feats = pd.concat(fixture_frames, ignore_index=True)

    candidate_feature_export = build_round16_candidate_feature_export(
        matchups=matchups,
        fixture_feats=fixture_feats,
        model_feature_cols=model_artifact["features"],
    )
    candidate_feature_export.to_csv(ROUND16_CANDIDATE_FEATURES_CSV, index=False)
    print(f"  Saved Round-of-16 candidate features → {ROUND16_CANDIDATE_FEATURES_CSV}")

    X = Xf(
        fixture_feats,
        model_artifact["features"],
        "round16 strict v3 augmented features",
    )

    clf = model_artifact["clf"]

    print("\n  Predicting all Round-of-16 matchup probabilities in one forward pass...")
    print(f"  Prediction matrix shape: {X.shape[0]:,} rows × {X.shape[1]:,} features")

    proba_matrix = clf.predict_proba(X)

    probability_classes = resolve_probability_classes(
        clf=clf,
        proba=proba_matrix,
        label="batched Round-of-16 prediction",
    )
    classes = list(probability_classes)

    prob_cache = {}
    probability_rows = []

    for idx, key in enumerate(cache_keys):
        row = matchups.iloc[idx]
        proba = proba_matrix[idx]

        p_home_win = get_class_probability(proba, classes, "home_win")
        p_draw = get_class_probability(proba, classes, "draw")
        p_away_win = get_class_probability(proba, classes, "away_win")

        total = p_home_win + p_draw + p_away_win
        if not np.isfinite(total) or total <= 0:
            raise ValueError(
                f"Invalid probability total for {row['home_team']} vs {row['away_team']}: {total}"
            )
        p_home_win = float(p_home_win / total)
        p_draw = float(p_draw / total)
        p_away_win = float(p_away_win / total)

        prob_cache[key] = {
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "p_home_win_90": p_home_win,
            "p_draw_90": p_draw,
            "p_away_win_90": p_away_win,
        }

        probability_rows.append({
            "match_id": row["match_id"],
            "round": row["round"],
            "date": row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "p_home_win": p_home_win,
            "p_draw": p_draw,
            "p_away_win": p_away_win,
        })

    probability_table = pd.DataFrame(probability_rows)

    print(f"  Cached Round-of-16 probabilities: {len(prob_cache):,}")
    return prob_cache, probability_table


def is_unresolved_slot_name(team_name):
    """Returns True when a team field is a bracket placeholder such as 2J or 3AEHIJ."""
    return team_name in BRACKET_SLOT_CANDIDATES


def build_round16_submission_assignment(slot_candidate_pool):
    """
    Builds the concrete slot assignment used for the competition upload CSV.

    Priority order:
    1. Explicit entries in ROUND16_SUBMISSION_SLOT_ASSIGNMENT.
    2. Singleton runtime candidate lists, when a slot has exactly one candidate.

    If any Round-of-16 slot still has multiple possible teams, the competition
    upload cannot be written safely because the platform requires one concrete
    row per actual match with full country names.
    """
    assignment = {}
    unresolved = {}

    for slot, candidates in slot_candidate_pool.items():
        if slot in ROUND16_SUBMISSION_SLOT_ASSIGNMENT:
            team = ROUND16_SUBMISSION_SLOT_ASSIGNMENT[slot]
            if team not in candidates:
                raise ValueError(
                    f"ROUND16_SUBMISSION_SLOT_ASSIGNMENT maps {slot!r} to {team!r}, "
                    f"but the runtime candidate pool is {candidates}."
                )
            assignment[slot] = team
        elif len(candidates) == 1:
            assignment[slot] = candidates[0]
        else:
            unresolved[slot] = candidates

    return assignment, unresolved


def build_round16_submission_rows(model_artifact, prob_cache, slot_assignment):
    """
    Builds the exact CSV schema required by the competition for Round of 16.

    Uses 90-minute outcome probabilities:
    - p_home_win
    - p_draw
    - p_away_win

    It intentionally does not use knockout advancement probabilities, because
    the competition upload asks for match-outcome probabilities and includes a
    draw column.
    """
    rows = []

    for match in BRACKET_R16:
        resolved = resolve_r16_match(match, slot_assignment)
        home_team = resolved["home_team"]
        away_team = resolved["away_team"]

        if is_unresolved_slot_name(home_team) or is_unresolved_slot_name(away_team):
            raise ValueError(
                f"Round-of-16 match {match['match_id']} is still unresolved: "
                f"{home_team} vs {away_team}. Fill ROUND16_SUBMISSION_SLOT_ASSIGNMENT "
                "before creating an upload CSV."
            )

        key = make_probability_cache_key(
            model_artifact=model_artifact,
            home_team=home_team,
            away_team=away_team,
            match_date=match["date"],
        )

        if key not in prob_cache:
            raise KeyError(
                f"Missing cached Round-of-16 probability for {home_team} vs {away_team} "
                f"on {match['date']}."
            )

        probs = prob_cache[key]
        p_home_win = float(probs["p_home_win_90"])
        p_draw = float(probs["p_draw_90"])
        p_away_win = float(probs["p_away_win_90"])

        # Round probabilities to a practical CSV precision, then repair the final
        # column so each row sums exactly to 1 at that precision.
        p_home_win = round(p_home_win, 6)
        p_draw = round(p_draw, 6)
        p_away_win = round(1.0 - p_home_win - p_draw, 6)

        rows.append({
            "date": str(pd.Timestamp(match["date"]).date()),
            "home_team": home_team,
            "away_team": away_team,
            "p_home_win": p_home_win,
            "p_draw": p_draw,
            "p_away_win": p_away_win,
        })

    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def validate_submission_frame(submission_df, expected_rows=8):
    """Validates the Round-of-16 upload shape before writing it."""
    if list(submission_df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(
            f"Submission columns must be {SUBMISSION_COLUMNS}, got {list(submission_df.columns)}"
        )

    if len(submission_df) != expected_rows:
        raise ValueError(
            f"Round-of-16 submission must have {expected_rows} rows, got {len(submission_df)}."
        )

    for col in ["home_team", "away_team"]:
        unresolved = submission_df[col].map(is_unresolved_slot_name)
        if unresolved.any():
            bad_values = submission_df.loc[unresolved, col].tolist()
            raise ValueError(f"Submission still contains unresolved slots in {col}: {bad_values}")

    prob_cols = ["p_home_win", "p_draw", "p_away_win"]
    probs = submission_df[prob_cols].to_numpy(dtype=float)

    if not np.isfinite(probs).all():
        raise ValueError("Submission probabilities contain NaN or Inf.")

    if ((probs < 0) | (probs > 1)).any():
        raise ValueError("Submission probabilities must be between 0 and 1.")

    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(f"Submission probability rows do not sum to 1: {row_sums}")


def write_round16_competition_outputs(model_artifact, prob_cache, matchup_probability_table, slot_candidate_pool):
    """
    Writes both a diagnostic candidate-probability file and, when possible, the
    upload-ready Round-of-16 competition CSV.
    """
    r32_candidates = matchup_probability_table[
        matchup_probability_table["round"].eq("Round of 16")
    ].copy()
    r32_candidates.to_csv(ROUND16_CANDIDATE_PROBABILITIES_CSV, index=False)
    print(f"  Saved → {ROUND16_CANDIDATE_PROBABILITIES_CSV}")

    slot_assignment, unresolved = build_round16_submission_assignment(slot_candidate_pool)

    if unresolved:
        print("\n  Round-of-16 upload CSV was not written yet.")
        print("  These slots still have multiple possible future teams:")
        for slot, candidates in unresolved.items():
            print(f"    {slot}: {', '.join(candidates)}")
        print("  Once official teams are known, fill ROUND16_SUBMISSION_SLOT_ASSIGNMENT and rerun.")
        return None

    submission_df = build_round16_submission_rows(
        model_artifact=model_artifact,
        prob_cache=prob_cache,
        slot_assignment=slot_assignment,
    )
    validate_submission_frame(submission_df, expected_rows=len(BRACKET_R16))
    submission_df.to_csv(ROUND16_SUBMISSION_CSV, index=False)

    print(f"  Saved upload-ready Round-of-16 CSV → {ROUND16_SUBMISSION_CSV}")
    return submission_df


def fit_evaluate_v3_full_2014(feats):
    """
    Fits only the selected v3 model:
    baseline features + attack/defense ratings + rolling historical residual bias.

    Rolling-residual v3 policy:
    - Do not fill residual-bias NaNs with 0.0.
    - Residual-bias columns are populated from prior out-of-sample residuals.
    - require_complete_features(...) keeps only rows where the full v3 feature
      set is genuinely available.
    """
    WC_START = pd.Timestamp("2026-06-11")

    played = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)].copy()
    pre_wc = played[played["date"] < WC_START].copy()
    test = played[played["date"] >= WC_START].copy()

    print("\n  Rolling-residual v3 missingness before filtering:")
    for col in BIAS_COLS:
        print(f"    pre_wc {col:>18s}: {pre_wc[col].isna().sum():>6} / {len(pre_wc)} missing")
    for col in BIAS_COLS:
        print(f"    test   {col:>18s}: {test[col].isna().sum():>6} / {len(test)} missing")

    train_v3 = require_complete_features(
        pre_wc,
        AUGMENTED_V3_FEATURES,
        "v3 train rolling residual augmented rows",
    )

    test_v3 = require_complete_features(
        test,
        AUGMENTED_V3_FEATURES,
        "v3 test rolling residual augmented rows",
    )

    print(
        f"\n  v3 rolling-residual train rows : {len(train_v3)} matches "
        f"from {train_v3['date'].min().date()} to {train_v3['date'].max().date()}"
    )
    print(f"  v3 rolling-residual test rows  : {len(test_v3)} WC matches\n")

    print(
        "  Final v3 fit uses TabPFN client thinking mode: "
        f"enabled={FINAL_V3_THINKING_MODE}, "
        f"effort={FINAL_V3_THINKING_EFFORT}, "
        f"metric={FINAL_V3_THINKING_METRIC}, "
        f"timeout_s={FINAL_V3_THINKING_TIMEOUT_S}"
    )

    clf3 = train_tabpfn(
        Xf(train_v3, AUGMENTED_V3_FEATURES, "v3 rolling-residual train"),
        train_v3["outcome"].values,
        use_thinking_mode=FINAL_V3_THINKING_MODE,
    )

    acc, ll, pred, proba, probability_classes = evaluate(
        clf3,
        Xf(test_v3, AUGMENTED_V3_FEATURES, "v3 rolling-residual test"),
        test_v3["outcome"].values,
        "v3 rolling residual + ratings + bias",
    )

    results = test_v3[["date", "home_team", "away_team", "outcome"]].copy()
    results["v3_pred"] = pred
    results["v3_correct"] = pred == test_v3["outcome"].values

    for idx, class_name in enumerate(probability_classes):
        results[f"v3_p_{class_name}"] = proba[:, idx]

    metrics = {"v3": (acc, ll)}
    model_artifacts = {
        "v3": {
            "label": "v3 rolling residuals + ratings + final thinking",
            "clf": clf3,
            "features": AUGMENTED_V3_FEATURES,
            "accuracy": acc,
            "log_loss": ll,
            "n_eval_rows": len(test_v3),
        }
    }

    return results, metrics, model_artifacts

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build martj42 engineered features, add first-layer ratings and rolling "
            "residual bias, then fit/predict the Round-of-16 model."
        )
    )
    parser.add_argument(
        "--refresh-results",
        action="store_true",
        help="Re-download martj42 results.csv before building features.",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help=(
            "Export engineered feature CSVs and stop before the final thinking-mode "
            "TabPFN fit/prediction."
        ),
    )
    parser.add_argument(
        "--disable-style-keywords",
        action="store_true",
        help=(
            "Disable time-aware style keyword covariates. This is the ablation "
            "switch for testing whether the style-triplet categorical features help."
        ),
    )
    parser.add_argument(
        "--style-eras-csv",
        type=str,
        default=None,
        help=(
            "Optional path to team_style_keyword_eras_round32_time_aware.csv. "
            "Relative paths are resolved from the repo/script directory."
        ),
    )
    args = parser.parse_args()

    if args.style_eras_csv is not None:
        STYLE_ERAS = Path(args.style_eras_csv)
        if not STYLE_ERAS.is_absolute():
            STYLE_ERAS = ROOT_DIR / STYLE_ERAS

    configure_style_features(
        use_style_keywords=not args.disable_style_keywords,
    )

    print("=" * 65)
    print("Loading and building features...")
    print("=" * 65)
    df = load_data(refresh=args.refresh_results)
    feats, state_snapshot = build_features(df)

    print("\n" + "=" * 65)
    print("Loading time-aware style keyword eras...")
    print("=" * 65)
    style_eras_by_team, style_encoder = load_style_keyword_eras()
    feats = join_style_keywords(
        feats=feats,
        style_eras_by_team=style_eras_by_team,
        style_encoder=style_encoder,
    )

    print("\n" + "=" * 65)
    print("Loading our team ratings...")
    print("=" * 65)
    rating_history = load_rating_history()

    # Join attack/defense ratings first. Residual-bias columns remain NaN here
    # and are populated row-wise by compute_rolling_residual_bias_features(...).
    feats = join_ratings(feats, rating_history)

    # Dict snapshot (latest checkpoint) for the fixture/knockout path only.
    ratings = latest_ratings_dict(rating_history)

    feats, latest_bias_df = compute_rolling_residual_bias_features(feats)

    # Enrich ratings dict with the latest rolling residual bias for future
    # Round-of-16 fixtures. This is memory only, not written to team_ratings.csv.
    for _, row in latest_bias_df.iterrows():
        team = row["team"]
        if team in ratings:
            ratings[team]["attack_residual_bias"] = row["attack_residual_bias"]
            ratings[team]["defense_residual_bias"] = row["defense_residual_bias"]

    print("\n" + "=" * 65)
    print("Exporting engineered feature tables...")
    print("=" * 65)
    slot_candidate_pool = build_slot_candidate_pool(ratings)
    export_engineered_feature_tables(
        feats=feats,
        ratings=ratings,
        state_snapshot=state_snapshot,
        slot_candidate_pool=slot_candidate_pool,
        style_eras_by_team=style_eras_by_team,
        style_encoder=style_encoder,
    )

    if args.features_only:
        print("\nFeature export complete. Stopping before final TabPFN fit because --features-only was passed.")
        raise SystemExit(0)

    print("\n" + "=" * 65)
    print("Fitting selected v3 model with rolling residual-bias features...")
    print("=" * 65)
    results, metrics, model_artifacts = fit_evaluate_v3_full_2014(feats)

    results.to_csv(OUTPUT_DIR / "v3_full2014_client_rolling_residual_comparison_results.csv", index=False)
    print("\nSaved v3 comparison rows to v3_full2014_client_rolling_residual_comparison_results.csv")

    print("\n" + "=" * 65)
    print("Predicting Round of 16 probabilities only...")
    print("=" * 65)
    prob_cache, r32_probability_table = precompute_round16_probability_cache(
        model_artifact=model_artifacts["v3"],
        ratings=ratings,
        state_snapshot=state_snapshot,
        slot_candidate_pool=slot_candidate_pool,
        characteristics=None,
        stat_cols=None,
        style_eras_by_team=style_eras_by_team,
        style_encoder=style_encoder,
    )

    write_round16_competition_outputs(
        model_artifact=model_artifacts["v3"],
        prob_cache=prob_cache,
        matchup_probability_table=r32_probability_table,
        slot_candidate_pool=slot_candidate_pool,
    )

    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    acc, ll = metrics["v3"]
    print(f"  v3: accuracy={acc:.1%}  log-loss={ll:.4f}  n_eval={model_artifacts['v3']['n_eval_rows']}")
    print(f"  Saved → {ROUND16_CANDIDATE_PROBABILITIES_CSV}")
    if os.path.exists(ROUND16_SUBMISSION_CSV):
        print(f"  Saved → {ROUND16_SUBMISSION_CSV}")
