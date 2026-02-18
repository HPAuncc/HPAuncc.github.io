---
layout: default
title: "Home-Field Advantage in the NFL (2000-2023)"
nav_exclude: true
---
# Home-Field Advantage in the NFL (2000-2023)

**Course:** DTSC-2301 Data Science Foundations (UNC Charlotte)  
**Tools:** Python, pandas, matplotlib, seaborn  
**Dataset:** nflverse game-level data (`games.csv`)

---

## Why this project matters
"Home-field advantage" is one of the most repeated ideas in football analysis. It shows up in broadcasts, betting markets, and fan narratives, often as an assumed truth rather than a measured effect. This project asks a more careful question: not just whether home-field advantage exists, but whether its strength has changed over time.

That distinction matters. A league can still have a home advantage overall while experiencing meaningful long-run shifts in how strong that advantage is.

---

## 1) Problem definition

### Research question
To what extent does home-field advantage influence NFL game outcomes, and how has that effect changed from 2000-2023?

### Scope and intent
This is exploratory analysis, not prediction. The goal is to describe trend behavior, quantify effect size, and interpret patterns with appropriate caution.

### Who might care
- Fans and media interpreting "home-field advantage" claims
- Analysts building descriptive league trends
- Students evaluating how to connect cleaning decisions to interpretation quality

---

## 2) Data description and sample design

### Data source
- nflverse `games.csv`
- https://github.com/nflverse/nfldata/raw/master/data/games.csv

### Unit of analysis
Each row is one NFL game.

### Main fields used
- `season`
- `game_type`
- `home_score`
- `away_score`
- `game_id`

### Sample size and filtering path
- Raw file: **7,276** games (1999-2025)
- Keep 2000-2023 only: **6,447** games
- Keep regular season only: **6,175** games
- Seasons covered: **24**
- Avg games/season: **257.3** (range 248-272)
- Ties in sample: **14** (0.23%)

### Why this sample definition
I targeted 2000-2023 to balance depth and comparability in the modern era. The regular season provides the most stable structure for year-over-year comparison.

---

## 3) Cleaning and preprocessing decisions

This section focuses on reasoning, not just steps.

### Decision A: Restrict to 2000-2023
**Why:** captures long-run movement without mixing too much cross-era structural change.

**Tradeoff:** excludes older history and any post-2023 updates.

### Decision B: Focus analysis on regular-season games
**Why:** regular season has consistent volume and context each year, making trend comparisons cleaner.

**Important caveat:** playoff rows in this dataset are labeled `WC`, `DIV`, `CON`, and `SB` (not `POST`), so the current postseason export logic yields an empty postseason file. Rather than forcing a weak comparison, final claims stay tied to the validated regular-season sample.

### Decision C: Coerce score fields to numeric and drop invalid rows
- `home_score` and `away_score` converted with `errors="coerce"`
- rows with missing scores removed

**Why:** prevents hidden type/data issues from contaminating winner logic.

**Observed effect:** 0 regular-season rows were dropped, but this validation still matters for reproducibility.

### Decision D: Treat ties explicitly
- `home_win = 1` for home win
- `home_win = 0` for home loss
- `home_win = NaN` for ties
- separate `home_tie` indicator

**Why:** forcing ties into 0/1 would bias win-rate interpretation.

### Decision E: Verify uniqueness with `game_id`
`game_id.nunique() == len(df_reg)` (6,175 = 6,175), confirming one row per unique game.

---

## 4) Visual analysis and why these visuals were chosen

### Visual 1: Home win rate by season
![Home win rate by season](/assets/images/home-field-advantage/home_win_trend.png)

**Why this chart:** It directly answers the trend question by showing year-by-year home win percentage.

**What it shows:**
- Home teams are usually above 50%, indicating a persistent home edge.
- The series is volatile year-to-year, which is expected for single-season rates.
- 2020 is visibly weak relative to surrounding years.

### Visual 2: Home win rate with 5-year rolling average
![Home win rate with 5-year rolling average](/assets/images/home-field-advantage/rolling_avg.png)

**Why this chart:** It separates short-term fluctuation from medium-term direction.

**Why 5-year (not 2- or 3-year):**
I evaluated smoothness by mean absolute year-to-year change in the smoothed series:
- 2-year: 0.0147
- 3-year: 0.0112
- 5-year: 0.0051
- 7-year: 0.0050

A 2- or 3-year window still overreacted to short-run noise. A 7-year window barely improved smoothness over 5-year but reduced responsiveness. The 5-year window gave the best balance between readability and signal retention.

### Visual 3: Home margin distribution
![Distribution of home margin](/assets/images/home-field-advantage/margin_distribution.png)

**Why this chart:** Win rate alone is binary; margin distribution adds effect-size context.

**What it shows:**
- Home teams do not only win slightly more often; they also hold a positive scoring margin on average.
- The distribution supports a moderate, not extreme, advantage profile.

---

## 5) Findings: what changed, when, and by how much

### Finding 1: Home-field advantage is real
Across 6,175 regular-season games, home teams won **56.26%** of games.

### Finding 2: The advantage appears weaker than in the early 2000s
- Early period (2000-2008) mean home win rate: **57.03%**
- Later period (2015-2023) mean home win rate: **54.84%**
- Change: **-2.19 percentage points**

### Finding 3: 2020 is the clearest disruption
- Lowest single-season rate in sample: **2020 at 49.80%**
- Highest single-season rate: **2003 at 61.33%**

### Finding 4: Post-2020 rebound exists, but does not erase the longer-run softening
The rate rebounds after 2020, but the broader pattern is still consistent with a narrower home edge than in the early 2000s.

### Margin context
- Mean home score: **23.17**
- Mean away score: **20.94**
- Mean home margin: **+2.23** points

---

## 6) Interpretation and narrative

The strongest defensible conclusion is not "home-field advantage disappeared." It is that the effect remains present but appears less dominant than it was in earlier years.

This pattern is consistent with multiple possible drivers (travel optimization, communication systems, league parity shifts, environmental changes including the 2020 attendance shock), but this project does not isolate mechanisms. The analysis supports directional interpretation, not causal attribution.

---

## 7) What would be misleading to conclude

It would be misleading to claim:
- home-field advantage is gone,
- crowd noise alone explains everything,
- this trend is a direct causal estimate,
- or that all teams are equally affected.

Descriptive trend evidence is useful, but only when paired with scope discipline.

---

## 8) Limitations, assumptions, and reflection

### Major limitations
- No controls for team strength, quarterback quality, injuries, coaching stability
- No direct attendance/crowd variable
- Neutral/international game context not separately modeled
- No causal framework (regression, matching, or quasi-experimental design)

### Explicit assumptions
- `game_id` uniquely identifies games
- score fields are valid after coercion checks
- seasonal aggregation is appropriate for league-level trend analysis

### Reflection
The most important lesson from this project is that small percentage shifts can still be meaningful at league scale, but overclaiming mechanism is easy when context is omitted. Better analysis means documenting decisions, showing uncertainty, and resisting single-cause narratives.

---

## 9) Next steps that would improve the analysis

If this project were extended, the highest-value improvements would be:
- add attendance/capacity data and test crowd-linked hypotheses directly,
- separate neutral-site/international games,
- add controls for team quality and quarterback continuity,
- compare pre-2010, 2010s, and post-2020 subperiods more formally.

---

## 10) References and transparency

### Data source
- nflverse `games.csv`: https://github.com/nflverse/nfldata/raw/master/data/games.csv

### Tool/process transparency
- Analysis performed in Python with pandas, matplotlib, and seaborn.
- Cleaning and smoothing decisions are explicitly documented to make reasoning auditable.

---

## View full code
GitHub Repository:  
[DTSC-2301-Project-1](https://github.com/HPAuncc/DTSC-2301/Project-1)
