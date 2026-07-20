"""
05_optimize_final_submission_game_theoretic.py
===============================================================================
Reproduce the competition-aware Spain-Argentina final submission.

The TabPFN pipeline's unbiased probability estimate is loaded directly from:

    data/output/final_round_submission_rolling_residual_v3.csv

The final competition action is different from the unbiased estimate because
only first place receives the prize. With one match remaining, the objective
was changed from minimizing expected match log loss to maximizing the chance
of overtaking the public leader.

Strategic hypothesis
--------------------
1. Player X was expected to use the same model-shaped Spain-Argentina row but
   sharpen it toward Spain.
2. Therefore the useful disagreement cells were draw and Argentina.
3. We calculated the probability required in each targeted cell to overcome
   the estimated remaining total-log-loss gap.
4. We covered an explicit grid of leader-sharpness assumptions, rounded each
   required probability upward, and added small declared robustness margins.
5. The remaining probability was assigned to Spain, subject to the 0.04 floor.

This is a deterministic decision layer. It is not presented as a better
estimate of the match probabilities than the underlying TabPFN model.
"""

from __future__ import annotations

from pathlib import Path
import csv
import math

import numpy as np


ROOT = Path(__file__).resolve().parent
CURRENT_MODEL_CSV = (
    ROOT / "data" / "output" / "final_round_submission_rolling_residual_v3.csv"
)
PRE_MATCH_MODEL_CSV = (
    ROOT
    / "data"
    / "output_untuned"
    / "final_round_submission_rolling_residual_v3.csv"
)
OUTPUT_CSV = ROOT / "optimized_submission_m104.csv"

# Public leaderboard immediately before the France-England third-place match.
N_SCORED_MATCHES = 30
OUR_MEAN_LOG_LOSS = 0.839
DOMINIK_MEAN_LOG_LOSS = 0.824
VISIBLE_TOTAL_GAP = (OUR_MEAN_LOG_LOSS - DOMINIK_MEAN_LOG_LOSS) * N_SCORED_MATCHES

# Our already-submitted France-England ticket; England won in regulation.
SUBMITTED_M103 = np.array([0.46, 0.30, 0.24], dtype=float)
REALIZED_M103_INDEX = 2

# Player X's M103 row is approximated by temperature-sharpening the preserved
# pre-match model snapshot. T=0.30 is the central estimate used for the gap
# update; the final-row robustness grid is wider below.
M103_LEADER_TEMPERATURE = 0.30

# Player X's final row is modeled as the same unbiased model output sharpened
# toward Spain. This grid covers a range from strongly to moderately sharp.
M104_LEADER_TEMPERATURES = (0.30, 0.35, 0.40, 0.45)

# Submission constraints and transparent robustness choices.
P_MIN = 0.04
ROUND_STEP = 0.01

# After rounding the worst-case threshold upward, add these declared margins.
# The draw receives one extra point because the unbiased model assigns it a
# slightly larger probability than Argentina, so it is the primary target cell.
DRAW_MARGIN = 0.04
ARGENTINA_MARGIN = 0.03


def validate_probability_row(row: np.ndarray, label: str) -> None:
    row = np.asarray(row, dtype=float)
    if row.shape != (3,):
        raise ValueError(f"{label}: expected shape (3,), got {row.shape}")
    if not np.all(np.isfinite(row)):
        raise ValueError(f"{label}: contains non-finite values")
    if np.any(row <= 0.0) or np.any(row >= 1.0):
        raise ValueError(f"{label}: probabilities must be strictly between 0 and 1")
    if not math.isclose(float(row.sum()), 1.0, abs_tol=1e-9):
        raise ValueError(f"{label}: probabilities sum to {row.sum():.12f}, not 1")


def load_row(path: Path, date: str, home: str, away: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required model output not found: {path}")

    found: list[np.ndarray] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["date"] == date and row["home_team"] == home and row["away_team"] == away:
                found.append(
                    np.array(
                        [
                            float(row["p_home_win"]),
                            float(row["p_draw"]),
                            float(row["p_away_win"]),
                        ],
                        dtype=float,
                    )
                )

    if len(found) != 1:
        raise ValueError(
            f"Expected one {date} {home}-{away} row in {path}; found {len(found)}"
        )
    validate_probability_row(found[0], f"{path.name}: {home}-{away}")
    return found[0]


def sharpen(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    q = np.asarray(probabilities, dtype=float) ** (1.0 / temperature)
    q /= q.sum()
    validate_probability_row(q, f"sharpened row T={temperature}")
    return q


def ceil_to(value: float, step: float) -> float:
    return float(np.ceil(value / step - 1e-12) * step)


def derive_ticket() -> tuple[np.ndarray, dict[str, float]]:
    # Genuine pre-match M103 probabilities. The current output row is not used
    # here because it was regenerated after the result and would leak outcome
    # information into the reconstruction of the pre-match decision.
    model_m103_pre_match = load_row(
        PRE_MATCH_MODEL_CSV,
        "2026-07-18",
        "France",
        "England",
    )

    # Exact unbiased final probability vector from the current TabPFN pipeline.
    model_m104 = load_row(
        CURRENT_MODEL_CSV,
        "2026-07-19",
        "Spain",
        "Argentina",
    )

    player_x_m103 = sharpen(model_m103_pre_match, M103_LEADER_TEMPERATURE)

    # Gain is Player X's log loss minus ours. Positive means we gained ground.
    m103_gain = math.log(
        SUBMITTED_M103[REALIZED_M103_INDEX]
        / player_x_m103[REALIZED_M103_INDEX]
    )
    remaining_gap = VISIBLE_TOTAL_GAP - m103_gain
    pass_ratio = math.exp(remaining_gap)

    # For outcome j, beating Player X requires:
    #     log(u_j / d_j) > remaining_gap
    # or equivalently:
    #     u_j > exp(remaining_gap) * d_j
    draw_requirements: list[float] = []
    argentina_requirements: list[float] = []
    for temperature in M104_LEADER_TEMPERATURES:
        player_x_m104 = sharpen(model_m104, temperature)
        draw_requirements.append(pass_ratio * player_x_m104[1])
        argentina_requirements.append(pass_ratio * player_x_m104[2])

    max_draw_requirement = max(draw_requirements)
    max_argentina_requirement = max(argentina_requirements)

    p_draw = ceil_to(max_draw_requirement, ROUND_STEP) + DRAW_MARGIN
    p_argentina = ceil_to(max_argentina_requirement, ROUND_STEP) + ARGENTINA_MARGIN
    p_spain = 1.0 - p_draw - p_argentina

    # The final strategy deliberately uses the competition floor on Spain.
    # Fail loudly if later edits make the chosen robustness assumptions
    # incompatible with that documented ticket.
    if p_spain < P_MIN - 1e-12:
        raise ValueError(
            "Covered draw/Argentina requirements exceed the simplex after margins: "
            f"Spain remainder={p_spain:.6f}"
        )

    ticket = np.array([p_spain, p_draw, p_argentina], dtype=float)
    validate_probability_row(ticket, "strategic final ticket")

    diagnostics = {
        "remaining_gap": remaining_gap,
        "pass_ratio": pass_ratio,
        "max_draw_requirement": max_draw_requirement,
        "max_argentina_requirement": max_argentina_requirement,
        "unbiased_spain": float(model_m104[0]),
        "unbiased_draw": float(model_m104[1]),
        "unbiased_argentina": float(model_m104[2]),
    }
    return ticket, diagnostics


def export_submission(ticket: np.ndarray) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "date",
                "home_team",
                "away_team",
                "p_home_win",
                "p_draw",
                "p_away_win",
            ]
        )
        writer.writerow(
            [
                "2026-07-19",
                "Spain",
                "Argentina",
                f"{ticket[0]:.2f}",
                f"{ticket[1]:.2f}",
                f"{ticket[2]:.2f}",
            ]
        )


if __name__ == "__main__":
    ticket, diagnostics = derive_ticket()

    print("Unbiased TabPFN final row")
    print(
        "  Spain / Draw / Argentina: "
        f"[{diagnostics['unbiased_spain']:.6f} "
        f"{diagnostics['unbiased_draw']:.6f} "
        f"{diagnostics['unbiased_argentina']:.6f}]"
    )
    print(f"Estimated remaining gap: {diagnostics['remaining_gap']:.6f}")
    print(f"Required probability ratio: {diagnostics['pass_ratio']:.6f}")
    print(
        "Worst covered thresholds: "
        f"draw>{diagnostics['max_draw_requirement']:.6f}, "
        f"Argentina>{diagnostics['max_argentina_requirement']:.6f}"
    )
    print(f"Strategic ticket: {ticket}")

    export_submission(ticket)
    print(f"Submission written -> {OUTPUT_CSV}")
