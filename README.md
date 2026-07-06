# FIFA World Cup 2026 — Match Outcome Prediction Pipeline

A TabPFN-based prediction system for the FIFA World Cup 2026 competition,
built to outperform the Prior Labs baseline. The pipeline layers time-varying
Dixon–Coles team ratings and rolling out-of-sample residual-bias features on
top of a Prior-Labs-style engineered feature set, then applies anchor-gated
post-hoc calibration to the frozen TabPFN forecasts.

The entire pipeline is reproducible from a single open data source: the
[martj42 international results dataset](https://github.com/martj42/international_results).
No scraping is involved.

Current stage: **Round of 16** (`03_predict_round16.py`). Round-of-32
performance: multiclass log-loss **0.8408** vs a 1.0986 uniform baseline,
11/16 correct modal picks.

---

## Repository layout

| File | Role |
|---|---|
| `01_dixon_coles_ratings.py` | Time-decayed Dixon–Coles goal model (attack/defense per team, home advantage, rho low-score correction). Exports the latest-snapshot `team_ratings.csv` and `dixon_coles_params.csv`. Also imported as a library by scripts 02 and 04. |
| `02_dixon_coles_rating_history.py` | Refits Dixon–Coles at quarterly checkpoints, each using only matches strictly before the checkpoint, producing a leak-free rating **history** (`team_rating_history.csv`). Script 03 joins ratings as-of each match date via backward asof-merge. Also refreshes the latest snapshot, so script 01 needs no separate run. |
| `reference_models.py` | Library: `OrderedLogitReference`, a weighted proportional-odds reference model used by the calibration step. No I/O. |
| `03_predict_round16.py` | Main modeling script. Builds martj42 engineered features, joins time-varying DC ratings and rolling residual-bias features, fits the v3 TabPFN model, and writes the Round-of-16 competition submission. |
| `04_anchor_gated_calibration_r16.py` | Anchor-constrained, gated geometric-pool calibration of the frozen TabPFN forecasts against a structural reference model (Dixon–Coles or ordered logit). Writes the calibrated R16 submission. |

Prior rounds' scripts (e.g. the Round-of-32 versions of 03/04) are superseded
each round and can be kept in a `legacy/` folder for provenance.

### Retired: Sofascore statistics path

Earlier iterations included a Sofascore scraping script and a
team-characteristics builder feeding an experimental "v4" model variant. That
path was retired: v4 was removed after repeated instability, the active v3
model consumes none of its output, and the scraping approach was dropped on
data-sourcing grounds. Script 03 retains dormant v4 hooks
(`load_team_characteristics`, `CHAR_FEATURES`) that are never invoked in the
main flow; they can be revived later against a properly licensed stats source
(FBref within its published crawl policy, StatsBomb open data, or a paid API).

### Data directories (created automatically)

```
data/raw/       results.csv (martj42)
data/interim/   team_ratings.csv, team_rating_history.csv, dixon_coles_params.csv,
                engineered-feature exports,
                team_style_keyword_eras_round32_time_aware.csv (team-era data,
                round-agnostic despite the filename),
                expert_anchors.csv (optional, for script 04)
data/output/    predictions, OOF streams, submissions, calibration diagnostics
```

---

## Run order (Round of 16)

**Step 1 — refresh `results.csv`** so it includes all matches through the
Round of 32:

```powershell
New-Item -ItemType Directory -Force -Path data\raw | Out-Null
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/martj42/international_results/master/results.csv" -OutFile "data\raw\results.csv"

# Sanity check: the July 3 R32 fixtures should appear. If empty, upstream
# hasn't updated yet — stop rather than train on stale data.
Select-String -Path data\raw\results.csv -Pattern "2026-07-03" | Select-Object -First 5
```

**Step 2 — rebuild the Dixon–Coles rating history:**

```powershell
python 02_dixon_coles_rating_history.py
```

Quarterly checkpoints from 2014 through the day after the latest played
match. Skippable under extreme time pressure (the pipeline stays valid —
fixture rows just carry older ratings), but it is fast and refreshes the
single most tournament-sensitive input, so run it when possible.

**Step 3 — fit v3 and predict the Round of 16:**

```powershell
python 03_predict_round16.py
```

Outputs:
- `data/output/round_of_16_submission_rolling_residual_v3.csv` — upload-ready
  90-minute outcome probabilities (8 rows).
- `data/output/round_of_16_candidate_probabilities_rolling_residual_v3.csv`
  and `..._candidate_features_...csv` — calibration inputs.
- `data/output/rolling_v2_oof_predictions_for_calibration.csv` — OOF stream
  for fitting the calibration map.
- `data/interim/final_v3_training_engineered_features.csv` — reference/gate
  covariates for script 04.

**Step 4 — post-hoc calibration:**

```powershell
python 04_anchor_gated_calibration_r16.py
```

Writes `data/output/calibrated_submission_anchor_gated_r16.csv` (the upload
candidate) plus reliability, lambda-path, and row-level diagnostics.

---

## Modeling design in brief

- **v1 baseline**: martj42 ELO / form / head-to-head / rest / importance
  features (Prior Labs replica).
- **v2**: v1 + time-varying Dixon–Coles attack/defense ratings, joined as-of
  each match date from checkpointed refits (no future-strength leakage).
- **v3 (selected)**: v2 + rolling out-of-sample residual-bias features
  (annual chronological folds), plus optional time-aware style-triplet
  categoricals.
- **Draw pricing**: the model prices 90-minute draws above typical market
  levels. This is structurally deliberate — historical WC knockout rounds run
  roughly 30–35% level at 90', and the R32 slate (5/16 draws) rewarded it.
- **Calibration**: gated geometric pooling between the frozen TabPFN forecast
  and a structural reference (Dixon–Coles or ordered logit), with one-sided
  expert anchor constraints; the pooling weight is a logistic gate over
  |elo_diff|, KL divergence, and excess favorite uncertainty. Lambda is chosen
  on realized-outcome log-loss only.

## Key conventions & gotchas

- **Team names** follow the martj42 convention everywhere; `OUR_TO_MARTJ42`
  in script 03 maps legacy display names (e.g. "USA" → "United States").
- **Leakage discipline** (training/validation side only — future fixtures are
  safe by construction): every fitted component (DC checkpoints, residual
  folds, ordered-logit reference, calibration OOF) is trained strictly on
  data dated before what it scores, so the model learns time-consistent
  feature relationships and validation numbers transfer honestly to unseen
  fixtures.
- **martj42 scores** for knockout matches are end-of-play (extra-time)
  scores, so shootout matches appear as draws; sanity-check WC test rows
  after refreshing.
- **TabPFN thinking mode** is toggleable per fit but off by default
  (quota + Python 3.10+ requirement); rolling residual baseline fits always
  use regular client mode. The submitted configuration keeps all thinking
  modes off and style keywords on, matching the validated R32 setup.
- **01 vs 02 training windows** differ by design: the latest snapshot trains
  from 2018 onward, while the checkpoint history trains from 2010 so early
  checkpoints have enough data.

## Advancing to later rounds

Copy `03_predict_round16.py`, rename the round identifiers, replace
`BRACKET_R16` with the four concrete Quarterfinal fixtures (home team =
winner of the first match id in each `BRACKET_DERIVED_ROUNDS` pair), drop the
Quarterfinal layer from `BRACKET_DERIVED_ROUNDS`, set
`validate_submission_frame(expected_rows=4)`, and point the script-04 copy at
the new candidate probability/feature files. Then repeat Steps 1–4.

## Requirements

Python 3.9+ (thinking mode requires 3.10+), `numpy`, `pandas`, `scipy`,
`scikit-learn`, `tabpfn_client`, `matplotlib` (visualization only).
