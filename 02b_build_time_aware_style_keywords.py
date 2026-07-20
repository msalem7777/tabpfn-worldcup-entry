#!/usr/bin/env python3
"""
Build a time-aware style-keyword augmentation for World Cup Round-of-32 teams.

This script intentionally avoids the original leakage pattern where a single 2026
team description is copied backward across every historical match. Instead, each
team has dated style eras, and a match row receives only the era whose date range
contains the match date.

Outputs:
  1) results_with_style_keywords_time_aware.csv
  2) team_style_keyword_eras_round32_time_aware.csv

Smoke tests are included at the bottom and run by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


INPUT_RESULTS_CSV = Path("data/raw/results.csv")
OUTPUT_RESULTS_CSV = Path("data/raw/results_with_style_keywords_time_aware.csv")
OUTPUT_ERAS_CSV = Path("data/interim/team_style_keyword_eras_round32_time_aware.csv")

# Keep the same small single-token vocabulary used in the previous controlled-vocab pass.
# The fixed order is also used to canonicalize each triplet so category levels collapse
# when two eras have the same unordered set of style descriptors.
VOCAB_ORDER = [
    "compact",
    "counter",
    "creative",
    "direct",
    "disciplined",
    "efficient",
    "flexible",
    "physical",
    "possession",
    "press",
    "structured",
    "tempo",
    "transition",
    "vertical",
    "wide",
    "workrate",
]
VOCAB_RANK = {keyword: rank for rank, keyword in enumerate(VOCAB_ORDER)}
MAX_VOCAB_SIZE = 20


@dataclass(frozen=True)
class StyleEra:
    """A dated style-label era for one national team."""

    team: str
    start: str
    end: str
    keyword_1: str
    keyword_2: str
    keyword_3: str
    era_label: str
    style_summary: str
    confidence: str = "medium"
    source_urls: str = ""
    evidence_basis: str = (
        "curated_asof_era_label; replace source_urls with URLs published inside "
        "this interval if strict scraped-source provenance is required"
    )

    def as_record(self) -> dict[str, str | int]:
        keywords = canonicalize_keywords([self.keyword_1, self.keyword_2, self.keyword_3])
        return {
            "team": self.team,
            "style_period_start": self.start,
            "style_period_end": self.end,
            "keyword_1": keywords[0],
            "keyword_2": keywords[1],
            "keyword_3": keywords[2],
            "keyword_triplet": "|".join(keywords),
            "era_label": self.era_label,
            "style_summary": self.style_summary,
            "confidence": self.confidence,
            "source_urls": self.source_urls,
            "evidence_basis": self.evidence_basis,
            "n_keywords": 3,
        }


def canonicalize_keywords(keywords: Iterable[str]) -> list[str]:
    """Validate and order a three-keyword set using the fixed controlled vocabulary."""
    clean_keywords = [str(keyword).strip().lower() for keyword in keywords]
    unknown_keywords = sorted(set(clean_keywords) - set(VOCAB_ORDER))
    if unknown_keywords:
        raise ValueError(f"Unknown style keyword(s): {unknown_keywords}")
    if len(clean_keywords) != 3:
        raise ValueError(f"Each era must have exactly 3 keywords; got {clean_keywords}")
    if len(set(clean_keywords)) != 3:
        raise ValueError(f"Each era must have 3 distinct keywords; got {clean_keywords}")
    return sorted(clean_keywords, key=lambda keyword: VOCAB_RANK[keyword])


def build_style_eras() -> list[StyleEra]:
    """Return dated style eras for the 2026 Round-of-32 teams.

    The first covered date is intentionally 2016-03-26 because that was the start
    date of the engineered feature file from the previous pass. Rows before that
    date are left blank rather than being assigned modern style tags.
    """

    return [
        # South Africa
        StyleEra("South Africa", "2016-03-26", "2017-10-31", "compact", "disciplined", "physical", "Mashaba/Baxter transition", "Conservative defensive shape with direct outlets and physical duels."),
        StyleEra("South Africa", "2017-11-01", "2021-05-04", "compact", "structured", "counter", "Baxter/Molefi Ntseki era", "Cautious block-oriented team with structured possession and counter-attacking phases."),
        StyleEra("South Africa", "2021-05-05", "2026-07-05", "compact", "structured", "efficient", "Hugo Broos era", "Compact and cohesive team that prioritizes defensive organization and efficient attacking moments."),

        # Canada
        StyleEra("Canada", "2016-03-26", "2018-01-07", "compact", "direct", "physical", "pre-Herdman era", "Lower-block, pragmatic side relying on direct play and physical contests."),
        StyleEra("Canada", "2018-01-08", "2023-08-27", "press", "vertical", "workrate", "John Herdman era", "High-energy team built around pressing, forward running, and intense work rate."),
        StyleEra("Canada", "2023-08-28", "2024-05-12", "direct", "vertical", "workrate", "interim Biello era", "Transitional side still leaning on direct vertical attacks and athletic output."),
        StyleEra("Canada", "2024-05-13", "2026-07-05", "press", "vertical", "workrate", "Jesse Marsch era", "Aggressive pressing, fast forward play, disruption, and sustained physical output."),

        # Brazil
        StyleEra("Brazil", "2016-03-26", "2022-12-31", "possession", "press", "structured", "Tite era", "Organized possession side with coordinated pressing and strong rest-defense structure."),
        StyleEra("Brazil", "2023-01-01", "2023-12-31", "creative", "flexible", "possession", "Ramon/Diniz transition", "Experimental possession-heavy period emphasizing rotations and individual creativity."),
        StyleEra("Brazil", "2024-01-01", "2025-12-31", "creative", "structured", "transition", "Dorival transition", "More pragmatic setup balancing creative attackers with transitional threat."),
        StyleEra("Brazil", "2026-01-01", "2026-07-05", "press", "creative", "flexible", "Ancelotti era", "Flexible attacking side mixing pressure, individual creation, and situational pragmatism."),

        # Japan
        StyleEra("Japan", "2016-03-26", "2018-07-31", "compact", "disciplined", "transition", "Halilhodzic/Nishino era", "Disciplined compact structure with fast transitions and collective defensive work."),
        StyleEra("Japan", "2018-08-01", "2026-07-05", "press", "structured", "transition", "Hajime Moriyasu era", "Collective pressing, coordinated spacing, and quick transitions from a structured base."),

        # Germany
        StyleEra("Germany", "2016-03-26", "2021-07-31", "creative", "possession", "structured", "late Joachim Low era", "Possession-oriented side relying on technical control, rotations, and structured buildup."),
        StyleEra("Germany", "2021-08-01", "2023-09-21", "possession", "press", "vertical", "Hansi Flick era", "High-line pressing and possession with more vertical attacking intent."),
        StyleEra("Germany", "2023-09-22", "2026-07-05", "possession", "press", "structured", "Julian Nagelsmann era", "Pressing intensity, possession control, rotations, and organized attacking structure."),

        # Paraguay
        StyleEra("Paraguay", "2016-03-26", "2018-09-02", "compact", "counter", "physical", "Arce/Morosini transition", "Physical and compact side most dangerous through counter-attacking moments."),
        StyleEra("Paraguay", "2018-09-03", "2021-10-13", "compact", "disciplined", "structured", "Eduardo Berizzo era", "Disciplined defensive base with a more structured possession plan."),
        StyleEra("Paraguay", "2021-10-14", "2023-09-19", "compact", "counter", "physical", "Guillermo Barros Schelotto era", "Compact and combative team relying on physicality and direct counters."),
        StyleEra("Paraguay", "2023-09-20", "2026-07-05", "compact", "counter", "disciplined", "Gustavo Alfaro era", "Deeply organized, disciplined team comfortable absorbing pressure and countering."),

        # Netherlands
        StyleEra("Netherlands", "2016-03-26", "2017-12-31", "possession", "structured", "wide", "Danny Blind transition", "Possession-oriented side seeking width but often lacking stability."),
        StyleEra("Netherlands", "2018-01-01", "2020-08-18", "direct", "transition", "wide", "first Ronald Koeman era", "Wide attacking outlets and direct transitional play from a rebuilt structure."),
        StyleEra("Netherlands", "2020-08-19", "2021-08-03", "possession", "structured", "wide", "Frank de Boer era", "Structured possession side using width and positional circulation."),
        StyleEra("Netherlands", "2021-08-04", "2022-12-31", "compact", "disciplined", "transition", "Louis van Gaal era", "Disciplined tournament side with compact spacing and transition attacks."),
        StyleEra("Netherlands", "2023-01-01", "2026-07-05", "direct", "possession", "wide", "second Ronald Koeman era", "Possession base with direct wide threats and aggressive wing play."),

        # Morocco
        StyleEra("Morocco", "2016-03-26", "2019-07-31", "compact", "counter", "disciplined", "Herve Renard era", "Disciplined compact team with strong defensive organization and counter-attacks."),
        StyleEra("Morocco", "2019-08-01", "2022-08-31", "direct", "disciplined", "structured", "Vahid Halilhodzic era", "Structured and disciplined side with direct attacking phases."),
        StyleEra("Morocco", "2022-09-01", "2026-07-05", "compact", "disciplined", "transition", "Regragui/Ouahbi era", "Compact defensive identity with fast transitions and high collective discipline."),

        # Ivory Coast
        StyleEra("Ivory Coast", "2016-03-26", "2020-03-03", "direct", "physical", "transition", "Dussuyer/Kamara era", "Physical team relying on direct attacks and transitions."),
        StyleEra("Ivory Coast", "2020-03-04", "2022-05-19", "physical", "structured", "transition", "Patrice Beaumelle era", "More structured version of a physical transition-oriented side."),
        StyleEra("Ivory Coast", "2022-05-20", "2024-01-23", "creative", "physical", "transition", "Jean-Louis Gasset era", "Physical squad with creative attackers and transition danger."),
        StyleEra("Ivory Coast", "2024-01-24", "2026-07-05", "creative", "physical", "transition", "Emerse Fae era", "Athletic and creative team that can break quickly through powerful runners."),

        # Norway
        StyleEra("Norway", "2016-03-26", "2020-12-06", "compact", "direct", "disciplined", "Lagerback era", "Compact and disciplined side with direct attacking routes."),
        StyleEra("Norway", "2020-12-07", "2026-07-05", "possession", "structured", "vertical", "Stale Solbakken era", "Structured build-up with vertical attacks toward elite forwards."),

        # France
        StyleEra("France", "2016-03-26", "2026-07-05", "efficient", "structured", "transition", "Didier Deschamps era", "Tournament-efficient team with strong structure, defensive balance, and lethal transitions."),

        # Sweden
        StyleEra("Sweden", "2016-03-26", "2021-07-31", "compact", "direct", "disciplined", "early Janne Andersson era", "Compact and disciplined side using direct attacks and set-piece strength."),
        StyleEra("Sweden", "2021-08-01", "2024-06-30", "compact", "direct", "physical", "late Janne Andersson transition", "Physical and direct team with compact defensive phases."),
        StyleEra("Sweden", "2024-07-01", "2026-07-05", "possession", "press", "structured", "Graham Potter era", "More possession- and press-oriented structure with positional control."),

        # Mexico
        StyleEra("Mexico", "2016-03-26", "2018-07-31", "flexible", "press", "transition", "Juan Carlos Osorio era", "Flexible game plans, pressing phases, and aggressive transition attacks."),
        StyleEra("Mexico", "2018-08-01", "2018-12-31", "direct", "transition", "wide", "post-Osorio interim", "Interim period leaning on direct wide attacks and transition moments."),
        StyleEra("Mexico", "2019-01-01", "2022-12-31", "possession", "press", "structured", "Gerardo Martino era", "Structured possession side with pressing and combination play."),
        StyleEra("Mexico", "2023-01-01", "2024-07-21", "direct", "transition", "wide", "Cocca/Lozano transition", "Wider and more direct team during an unstable coaching transition."),
        StyleEra("Mexico", "2024-07-22", "2026-07-05", "counter", "disciplined", "structured", "Javier Aguirre era", "Pragmatic and disciplined side with structured defending and counter opportunities."),

        # Ecuador
        StyleEra("Ecuador", "2016-03-26", "2020-08-31", "direct", "physical", "transition", "Quinteros/Gomez transition", "Athletic side emphasizing direct play, duels, and transitions."),
        StyleEra("Ecuador", "2020-09-01", "2022-12-31", "compact", "physical", "transition", "Gustavo Alfaro era", "Compact, athletic team with transition speed and physical defensive play."),
        StyleEra("Ecuador", "2023-01-01", "2026-07-05", "press", "structured", "transition", "Sanchez/Beccacece era", "Structured and intense side with pressing phases and transition threat."),

        # England
        StyleEra("England", "2016-03-26", "2024-07-16", "disciplined", "possession", "structured", "Gareth Southgate era", "Controlled tournament side built around structure, caution, and possession security."),
        StyleEra("England", "2024-07-17", "2024-12-31", "creative", "possession", "press", "interim transition", "More experimental possession side with creative midfield use and pressing attempts."),
        StyleEra("England", "2025-01-01", "2026-07-05", "creative", "structured", "wide", "Thomas Tuchel era", "Defined structure with central creators and direct wide attacking threats."),

        # DR Congo
        StyleEra("DR Congo", "2016-03-26", "2021-05-30", "direct", "physical", "transition", "pre-Cuper era", "Physical and direct side with transition danger."),
        StyleEra("DR Congo", "2021-05-31", "2022-08-10", "compact", "counter", "disciplined", "Hector Cuper era", "Compact and defensive team designed around counters and discipline."),
        StyleEra("DR Congo", "2022-08-11", "2026-07-05", "compact", "physical", "transition", "Sebastien Desabre era", "Compact, athletic side with physical duels and fast attacking transitions."),

        # Belgium
        StyleEra("Belgium", "2016-03-26", "2022-12-31", "creative", "possession", "press", "Roberto Martinez era", "Possession-heavy and creative golden-generation side with pressing phases."),
        StyleEra("Belgium", "2023-01-01", "2025-12-31", "flexible", "press", "vertical", "Domenico Tedesco era", "Flexible, more vertical team with pressing and quicker forward attacks."),
        StyleEra("Belgium", "2026-01-01", "2026-07-05", "creative", "possession", "wide", "Rudi Garcia era", "Creative possession side using wide attackers and experienced playmakers."),

        # Senegal
        StyleEra("Senegal", "2016-03-26", "2024-10-31", "compact", "physical", "transition", "Aliou Cisse era", "Compact, physical, and transition-oriented team with strong defensive identity."),
        StyleEra("Senegal", "2024-11-01", "2026-07-05", "physical", "structured", "transition", "Pape Thiaw era", "Structured but still physical side with transition outlets."),

        # United States
        StyleEra("United States", "2016-03-26", "2018-12-01", "direct", "physical", "transition", "Arena/Sarachan transition", "Direct, athletic side during a rebuilding transition."),
        StyleEra("United States", "2018-12-02", "2022-12-31", "possession", "press", "structured", "early Berhalter era", "Structured possession model with pressing and positional buildup."),
        StyleEra("United States", "2023-01-01", "2024-09-09", "possession", "press", "wide", "late Berhalter era", "Possession and pressing side relying heavily on wide attackers."),
        StyleEra("United States", "2024-09-10", "2026-07-05", "press", "vertical", "workrate", "Pochettino era", "High-energy pressing side with vertical attacks and strong running volume."),

        # Bosnia and Herzegovina
        StyleEra("Bosnia and Herzegovina", "2016-03-26", "2019-12-31", "creative", "direct", "possession", "Bazdarevic/Prosinecki era", "Technical attacking side with creative midfielders and direct forward routes."),
        StyleEra("Bosnia and Herzegovina", "2020-01-01", "2023-12-31", "compact", "direct", "physical", "Bajevic/Petev/Hadzibegic era", "Inconsistent but generally compact and direct team relying on physical contests."),
        StyleEra("Bosnia and Herzegovina", "2024-01-01", "2026-07-05", "compact", "counter", "structured", "Barbarez era", "More structured defensive side looking for counter-attacking chances."),

        # Spain
        StyleEra("Spain", "2016-03-26", "2018-07-08", "possession", "press", "wide", "Lopetegui/Hierro era", "Possession-first side using high pressing and wide positional occupation."),
        StyleEra("Spain", "2018-07-09", "2022-12-31", "possession", "press", "wide", "Luis Enrique era", "High-possession, high-pressing side with wide occupation and positional rotations."),
        StyleEra("Spain", "2023-01-01", "2026-07-05", "possession", "press", "vertical", "Luis de la Fuente era", "Possession base with more vertical attacks, pressing, and direct wide progression."),

        # Austria
        StyleEra("Austria", "2016-03-26", "2017-12-31", "compact", "direct", "disciplined", "Marcel Koller era", "Disciplined and compact team with direct attacking phases."),
        StyleEra("Austria", "2018-01-01", "2022-05-31", "compact", "direct", "structured", "Franco Foda era", "Structured and pragmatic team with directness and compact defending."),
        StyleEra("Austria", "2022-06-01", "2026-07-05", "press", "vertical", "workrate", "Ralf Rangnick era", "High-intensity pressing team with vertical attacks and heavy work rate."),

        # Portugal
        StyleEra("Portugal", "2016-03-26", "2022-12-31", "counter", "efficient", "structured", "Fernando Santos era", "Pragmatic tournament side emphasizing structure, efficiency, and counters."),
        StyleEra("Portugal", "2023-01-01", "2026-07-05", "creative", "possession", "wide", "Roberto Martinez era", "More attacking possession side built around creative players and wide overloads."),

        # Croatia
        StyleEra("Croatia", "2016-03-26", "2017-10-06", "creative", "disciplined", "possession", "Ante Cacic era", "Technical possession side with creative midfield control."),
        StyleEra("Croatia", "2017-10-07", "2026-07-05", "creative", "possession", "structured", "Zlatko Dalic era", "Structured technical team built around midfield possession and creative control."),

        # Switzerland
        StyleEra("Switzerland", "2016-03-26", "2021-07-31", "compact", "disciplined", "transition", "Vladimir Petkovic era", "Compact and disciplined tournament team with transition threat."),
        StyleEra("Switzerland", "2021-08-01", "2026-07-05", "compact", "flexible", "transition", "Murat Yakin era", "Flexible tournament side, usually compact and comfortable attacking in transition."),

        # Algeria
        StyleEra("Algeria", "2016-03-26", "2018-07-31", "direct", "physical", "transition", "pre-Belmadi transition", "Physical side relying on direct attacking and transition quality."),
        StyleEra("Algeria", "2018-08-01", "2022-12-31", "creative", "press", "transition", "early Djamel Belmadi era", "Energetic side with pressing, transition speed, and creative attackers."),
        StyleEra("Algeria", "2023-01-01", "2024-02-28", "creative", "possession", "transition", "late Belmadi era", "More possession-oriented side still relying on creative transition attacks."),
        StyleEra("Algeria", "2024-02-29", "2026-07-05", "creative", "possession", "structured", "Vladimir Petkovic era", "More structured possession team with creative attacking midfielders."),

        # Australia
        StyleEra("Australia", "2016-03-26", "2017-11-21", "possession", "press", "vertical", "Ange Postecoglou era", "Aggressive possession side with pressing and vertical progression."),
        StyleEra("Australia", "2017-11-22", "2018-07-31", "compact", "counter", "structured", "Bert van Marwijk era", "More compact and pragmatic tournament setup focused on structure and counters."),
        StyleEra("Australia", "2018-08-01", "2024-09-19", "direct", "disciplined", "physical", "Graham Arnold era", "Disciplined and physical team with direct attacking phases."),
        StyleEra("Australia", "2024-09-20", "2026-07-05", "direct", "disciplined", "structured", "Tony Popovic era", "Structured and disciplined team with direct attacking routes."),

        # Egypt
        StyleEra("Egypt", "2016-03-26", "2018-06-30", "compact", "counter", "disciplined", "Hector Cuper era", "Compact defensive side designed around counters and disciplined spacing."),
        StyleEra("Egypt", "2018-07-01", "2021-09-07", "direct", "structured", "transition", "Aguirre/El Badry era", "Structured side using direct attacks and transition moments."),
        StyleEra("Egypt", "2021-09-08", "2022-06-30", "compact", "counter", "disciplined", "Carlos Queiroz era", "Very compact and disciplined side looking for counter-attacking chances."),
        StyleEra("Egypt", "2022-07-01", "2024-02-04", "creative", "possession", "structured", "Rui Vitoria era", "More possession-oriented side using creative attackers inside a structured model."),
        StyleEra("Egypt", "2024-02-05", "2026-07-05", "direct", "physical", "transition", "Hossam Hassan era", "More direct and physical team relying on quick transitions."),

        # Argentina
        StyleEra("Argentina", "2016-03-26", "2018-07-31", "creative", "possession", "press", "Bauza/Sampaoli era", "Possession and pressing ideas built around elite creative attackers but with instability."),
        StyleEra("Argentina", "2018-08-01", "2026-07-05", "creative", "possession", "structured", "Lionel Scaloni era", "Structured possession side with elite creators, balance, and adaptable control."),

        # Cape Verde
        StyleEra("Cape Verde", "2016-03-26", "2020-01-14", "compact", "direct", "physical", "pre-Bubista era", "Compact and physical side relying on direct attacking moments."),
        StyleEra("Cape Verde", "2020-01-15", "2026-07-05", "compact", "disciplined", "transition", "Bubista era", "Compact and disciplined team with dangerous transitions."),

        # Colombia
        StyleEra("Colombia", "2016-03-26", "2018-08-31", "creative", "possession", "transition", "Jose Pekerman era", "Creative possession side with technical midfielders and transition threat."),
        StyleEra("Colombia", "2018-09-01", "2022-06-30", "compact", "direct", "physical", "Queiroz/Rueda era", "More compact and direct team relying on physicality and individual forwards."),
        StyleEra("Colombia", "2022-07-01", "2026-07-05", "creative", "possession", "transition", "Nestor Lorenzo era", "Creative possession team with strong transition attacking and midfield control."),

        # Ghana
        StyleEra("Ghana", "2016-03-26", "2017-12-31", "direct", "physical", "transition", "Avram Grant/Appiah transition", "Physical transition side with direct attacking outlets."),
        StyleEra("Ghana", "2018-01-01", "2022-02-28", "direct", "physical", "transition", "Appiah/Rajevac era", "Direct and physical team leaning on transitional attacks."),
        StyleEra("Ghana", "2022-03-01", "2024-01-31", "compact", "counter", "disciplined", "Otto Addo/Hughton era", "Compact and cautious side relying on counter-attacks."),
        StyleEra("Ghana", "2024-02-01", "2026-07-05", "efficient", "physical", "transition", "modern Ghana era", "Physical and transition-oriented side seeking efficient finishing moments."),
    ]


def build_era_lookup(eras_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group era rows by team and convert dates to pandas timestamps."""
    lookup: dict[str, pd.DataFrame] = {}
    for team, team_eras in eras_df.groupby("team", sort=False):
        team_eras = team_eras.copy()
        team_eras["style_period_start_dt"] = pd.to_datetime(team_eras["style_period_start"])
        team_eras["style_period_end_dt"] = pd.to_datetime(team_eras["style_period_end"])
        lookup[team] = team_eras.sort_values("style_period_start_dt")
    return lookup


def match_era(team: str, match_date: pd.Timestamp, era_lookup: dict[str, pd.DataFrame]) -> pd.Series | None:
    """Return the style era row active for a team on a specific match date."""
    team_eras = era_lookup.get(team)
    if team_eras is None:
        return None
    mask = (team_eras["style_period_start_dt"] <= match_date) & (match_date <= team_eras["style_period_end_dt"])
    matches = team_eras.loc[mask]
    if matches.empty:
        return None
    return matches.iloc[-1]


def side_style_fields(prefix: str, era_row: pd.Series | None, is_round32_team: bool) -> dict[str, object]:
    """Build row-level style fields for either the home or away side."""
    fields: dict[str, object] = {f"{prefix}_is_2026_round32": int(is_round32_team)}
    if era_row is None:
        fields.update(
            {
                f"{prefix}_style_keyword_1": pd.NA,
                f"{prefix}_style_keyword_2": pd.NA,
                f"{prefix}_style_keyword_3": pd.NA,
                f"{prefix}_style_keyword_triplet": pd.NA,
                f"{prefix}_style_summary": pd.NA,
                f"{prefix}_style_confidence": pd.NA,
                f"{prefix}_style_source_urls": pd.NA,
                f"{prefix}_style_period_start": pd.NA,
                f"{prefix}_style_period_end": pd.NA,
                f"{prefix}_style_era_label": pd.NA,
                f"{prefix}_style_evidence_basis": pd.NA,
                f"{prefix}_style_timing_status": "no_era_coverage" if is_round32_team else "not_round32_team",
            }
        )
        return fields

    fields.update(
        {
            f"{prefix}_style_keyword_1": era_row["keyword_1"],
            f"{prefix}_style_keyword_2": era_row["keyword_2"],
            f"{prefix}_style_keyword_3": era_row["keyword_3"],
            f"{prefix}_style_keyword_triplet": era_row["keyword_triplet"],
            f"{prefix}_style_summary": era_row["style_summary"],
            f"{prefix}_style_confidence": era_row["confidence"],
            f"{prefix}_style_source_urls": era_row["source_urls"],
            f"{prefix}_style_period_start": era_row["style_period_start"],
            f"{prefix}_style_period_end": era_row["style_period_end"],
            f"{prefix}_style_era_label": era_row["era_label"],
            f"{prefix}_style_evidence_basis": era_row["evidence_basis"],
            f"{prefix}_style_timing_status": "as_of_match_date",
        }
    )
    return fields


def augment_results(results_df: pd.DataFrame, eras_df: pd.DataFrame) -> pd.DataFrame:
    """Attach time-aware style fields to each home and away side."""
    round32_teams = set(eras_df["team"].unique())
    era_lookup = build_era_lookup(eras_df)

    output_records: list[dict[str, object]] = []
    dated_results = results_df.copy()
    dated_results["match_date_dt"] = pd.to_datetime(dated_results["date"])

    for row in dated_results.to_dict(orient="records"):
        home_team = row["home_team"]
        away_team = row["away_team"]
        match_date = row["match_date_dt"]

        row.pop("match_date_dt", None)
        row.update(side_style_fields("home", match_era(home_team, match_date, era_lookup), home_team in round32_teams))
        row.update(side_style_fields("away", match_era(away_team, match_date, era_lookup), away_team in round32_teams))
        output_records.append(row)

    return pd.DataFrame(output_records)


def run_smoke_tests(original_df: pd.DataFrame, augmented_df: pd.DataFrame, eras_df: pd.DataFrame) -> None:
    """Minimal tests that catch the original future-leakage failure mode."""
    assert len(augmented_df) == len(original_df), "Row count changed during augmentation."

    vocab_used = set(eras_df["keyword_1"]) | set(eras_df["keyword_2"]) | set(eras_df["keyword_3"])
    assert len(vocab_used) <= MAX_VOCAB_SIZE, f"Vocabulary too large: {len(vocab_used)}"
    assert vocab_used <= set(VOCAB_ORDER), f"Unexpected keyword outside controlled vocabulary: {vocab_used - set(VOCAB_ORDER)}"

    england_1872 = augmented_df.loc[
        (augmented_df["date"] == "1872-11-30")
        & (augmented_df["home_team"] == "Scotland")
        & (augmented_df["away_team"] == "England")
    ].iloc[0]
    assert pd.isna(england_1872["away_style_keyword_triplet"]), "Pre-coverage England row should not receive 2026 tags."
    assert england_1872["away_style_timing_status"] == "no_era_coverage", "Pre-coverage England timing status is wrong."

    england_2020_rows = augmented_df.loc[
        (augmented_df["date"] >= "2020-01-01")
        & (augmented_df["date"] <= "2020-12-31")
        & ((augmented_df["home_team"] == "England") | (augmented_df["away_team"] == "England"))
    ]
    if not england_2020_rows.empty:
        row = england_2020_rows.iloc[0]
        triplet = row["home_style_keyword_triplet"] if row["home_team"] == "England" else row["away_style_keyword_triplet"]
        assert triplet == "disciplined|possession|structured", "England 2020 should use Southgate-era tags."

    england_2026_rows = augmented_df.loc[
        (augmented_df["date"] >= "2026-01-01")
        & ((augmented_df["home_team"] == "England") | (augmented_df["away_team"] == "England"))
    ]
    if not england_2026_rows.empty:
        row = england_2026_rows.iloc[0]
        triplet = row["home_style_keyword_triplet"] if row["home_team"] == "England" else row["away_style_keyword_triplet"]
        assert triplet == "creative|structured|wide", "England 2026 should use Tuchel-era tags."

    statuses = set(augmented_df["home_style_timing_status"].dropna()) | set(augmented_df["away_style_timing_status"].dropna())
    expected_statuses = {"as_of_match_date", "no_era_coverage", "not_round32_team"}
    assert statuses <= expected_statuses, f"Unexpected timing status values: {statuses - expected_statuses}"


def main() -> None:
    results_df = pd.read_csv(INPUT_RESULTS_CSV)
    eras_df = pd.DataFrame([era.as_record() for era in build_style_eras()])
    augmented_df = augment_results(results_df, eras_df)

    run_smoke_tests(results_df, augmented_df, eras_df)

    augmented_df.to_csv(OUTPUT_RESULTS_CSV, index=False)
    eras_df.to_csv(OUTPUT_ERAS_CSV, index=False)

    vocab_used = sorted(
        set(eras_df["keyword_1"]) | set(eras_df["keyword_2"]) | set(eras_df["keyword_3"]),
        key=lambda keyword: VOCAB_RANK[keyword],
    )
    print(f"Wrote: {OUTPUT_RESULTS_CSV}")
    print(f"Wrote: {OUTPUT_ERAS_CSV}")
    print(f"Rows: {len(augmented_df):,}")
    print(f"Teams: {eras_df['team'].nunique()}")
    print(f"Era rows: {len(eras_df)}")
    print(f"Vocabulary size: {len(vocab_used)}")
    print("Vocabulary: " + ", ".join(vocab_used))


if __name__ == "__main__":
    main()
