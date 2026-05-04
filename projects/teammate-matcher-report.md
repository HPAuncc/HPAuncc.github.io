---
layout: post
title: "Teammate Matcher: Technical Report"
subtitle: "Group Project Team Formation via Clustering & Optimization"
thumbnail: /assets/images/teammate-matcher-report/thumbnail.png
thumbnail_alt: "Technical report thumbnail for the Teammate Matcher project showing clustering and optimization methodology."
nav_exclude: true
---
*Hampton Abbott & Carly Castillo — DTSC 2302, University of North Carolina at Charlotte*
*Spring 2026*

---

## Abstract

Academic team formation in data science courses is typically left to random assignment or student self-selection, producing teams with scheduling conflicts, mismatched work styles, and uneven skill distribution. This report presents the **Teammate Matcher**, a data-driven application that optimizes academic team formation using four machine learning approaches: K-Means clustering, Agglomerative hierarchical clustering, size-constrained assignment via the **Hungarian Algorithm**, and **Gaussian Mixture Models (GMM)**. Primary survey data were collected from 31 students in DTSC 2302, capturing availability, work style, communication preferences, and self-rated technical skills. Responses were preprocessed through a nine-step pipeline producing 50 features across two feature sets: *compatibility features* (availability + work style, used for similarity-based models) and *complementarity features* (8 skill dimensions, used for diversity-based GMM). Hungarian Assignment produced the most practically deployable configuration — balanced teams of 3–6 students with near-complete skill coverage (7.875/8 dimensions per team) — while GMM identified latent skill archetypes via soft membership probabilities. The ambiguity-flag mechanism (max posterior < 0.60) returned zero borderline cases on this cohort: with N = 31 and full-covariance components, GMM converges to near-deterministic posteriors. The mechanism is correct; this cohort simply does not exercise it (Section 4.5.4).

Principal Component Analysis confirmed that availability and work-style are the primary axes of student differentiation (PC1 = day-of-week availability, 15.8%; PC2 = conflict/meeting style, 13.5%), with self-rated skills appearing only as a secondary axis from PC3 onward.

---

## 1. Introduction & Research Questions

### 1.1 Problem Statement

The success of academic group projects depends heavily on team composition. Prior research has established that random assignment and student self-selection both produce suboptimal outcomes — including uneven distribution of work, scheduling conflicts, and mismatched work styles [1]. These issues hinder the pedagogical goals of collaborative assignments and cause measurable frustration among students.

The **Teammate Matcher** addresses this problem by developing an automated, data-driven recommendation system for team formation, using directly measured student attributes collected via primary survey data. Unlike prior approaches that rely on indirect behavioral proxies (e.g., Learning Management System click counts, course forum activity), this system uses attributes most relevant to actual teamwork: self-reported technical skills, weekly availability, communication preferences, and work style.

### 1.2 Research Questions

> **RQ1.** How can we quantify and represent student skills, availability, and work styles using direct self-report to facilitate effective team matching?
>
> **RQ2.** Which clustering or optimization algorithms produce the most balanced and compatible student teams — and does the answer differ when optimizing for *similar* teams versus *complementary* teams?
>
> **RQ3.** What student attributes are most predictive of cluster separation, and which features contribute most to team differentiation?

A key framing decision distinguishes this work from prior approaches: we treat **similarity** (grouping students with compatible schedules and work styles) and **complementarity** (ensuring skill diversity within teams) as *distinct objectives* evaluated separately. No single clustering algorithm naturally optimizes both.

### 1.3 Societal Relevance

Team assignment is not a neutral algorithmic problem. Choices in feature selection, normalization, and model objective carry equity implications:

- **Skill diversity without constraint** could isolate lower-confidence students onto a single team.
- **Schedule homogeneity** could inadvertently group students by implicit socioeconomic factors (e.g., students who work nights share low daytime availability).
- **Self-reported skills** are subject to social desirability bias — students may over-rate prestigious skills (ML) and under-rate others (writing).

These concerns are revisited in Section 8.

---

## 2. Dataset & Survey Design

### 2.1 Why Primary Data Collection?

The project guidelines require a dataset that is **not** sourced from Kaggle or the UCI Repository. Beyond that requirement, there is a principled reason for primary data collection:

No public dataset contains the information needed for this problem. Team formation requires direct measurement of:

1. **Schedule availability** — not inferable from activity logs.
2. **Self-rated skills across multiple dimensions** — not inferable from click counts on course videos.
3. **Work style and conflict approach** — not captured in any educational dataset we could identify.
4. **Communication channel preferences** — likewise unrecorded elsewhere.

We therefore designed and deployed an **anonymous student survey** in the DTSC 2302 course, collecting primary data directly from the population the model is intended to serve.

### 2.2 Survey Instrument

The survey was deployed via Google Forms and organized into five sections:

| Section | Items | Purpose |
|---|---|---|
| 1. Context | Course code, year in school | Scope/demographic filter |
| 2. Schedule & Availability | Days (checkbox), time slots (checkbox), weekly hours, meeting mode | Schedule compatibility features |
| 3. Technical Skills | 8 items, Likert 1–5 | Skill profile / complementarity |
| 4. Work Style | Role, deadline approach, communication, check-in, collaboration, detail focus, conflict style | Work-style compatibility |
| 5. Self-Assessment | GPA band (optional), top contributions, biggest pain point | Self-knowledge signals |

### 2.3 Dataset Characteristics

| Property | Value |
|---|---|
| Responses collected | 31 |
| Raw CSV columns | 27 (includes 2 empty Google Forms artifacts) |
| Features after preprocessing | 50 |
| Missing values | 2 (both GPA) |
| Year distribution | Juniors 13, Sophomores 11, Freshmen 4, Graduate 2, Senior 1 |

**Cohort availability profile:**

![Cohort schedule heatmap. 31 students sorted by total availability (most-available rows at top). Saturday, Sunday, and Late Night are visibly sparser than weekdays and afternoon/evening slots.](https://raw.githubusercontent.com/HPAuncc/teammate-matcher/main/outputs/schedule_heatmap.png)

### 2.4 Anonymity & Quasi-Identifier Analysis

The survey promised anonymity. The raw CSV contains a `Timestamp` column (exact submission time to the second) which is a quasi-identifier: in a class of 31 students who know each other, "who submitted at 11:30 AM on April 17th" can narrow down the respondent. Two protective measures address this:

1. The raw CSV is excluded from the public repository via `.gitignore` and stays local only.
2. The processed CSV is **row-shuffled** before saving (`preprocess.py` Step 9, `random_state=42`), eliminating the correlation between row position and submission order.

---

## 3. Data Preprocessing Pipeline

The preprocessing pipeline (`src/preprocess.py`) implements nine sequential steps.

### Step 1 — Load Raw CSV
Raw Google Forms export is read with default pandas CSV parsing. No modifications to the on-disk file — we preserve the raw artifact in case preprocessing needs to be re-audited.

### Step 2 — Column Cleaning & Artifact Removal
- Rename 25 verbose Google Forms question strings to short internal names.
- Drop two empty artifact columns generated by Google Forms' checkbox export quirk.
- Drop the `Timestamp` column (privacy).
- Normalize `course_code` to `"DTSC 2302"` (11 free-text variants).

### Step 3 — Availability Encoding (Checkbox Expansion)
Expand comma-separated `_days_raw` and `_times_raw` strings into 11 binary columns: `avail_mon`, `avail_tue`, `avail_wed`, `avail_thu`, `avail_fri`, `avail_sat`, `avail_sun`, `avail_morning`, `avail_afternoon`, `avail_evening`, `avail_latenight`.

Two students' availability can then be compared via Jaccard similarity: J(a,b) = |a ∩ b| / |a ∪ b|, which directly quantifies scheduling overlap and is used as the Schedule Overlap evaluation metric.

### Step 4 — Ordinal Encoding
Map ordered categorical responses to integers preserving natural ordering:

| Column | Encoding |
|---|---|
| `year` | Freshman=1, Sophomore=2, Junior=3, Senior=4, Graduate=5 |
| `weekly_hours` | <3hrs=1, 3–5=2, 6–9=3, 10+=4 |
| `role_pref` | Follower=1, Specialist=2, Flexible=3, Leader=4 |
| `deadline_style` | Last-minute=1, Steady=2, Early=3 |
| `checkin_freq` | As needed=1, Weekly=2, Few/week=3, Daily=4 |
| `collab_style` | Independent=1, Mix=2, Close=3 |
| `gpa_band` | <2.5=1, 2.5–3.0=2, 3.0–3.5=3, 3.5–4.0=4 |

### Step 5 — One-Hot Encoding
Expand four nominal variables into binary indicators: `meeting_mode`, `comm_pref`, `conflict_style`, `pain_point`.

### Step 6 — Contribution Multi-Select Encoding
Expand `_contrib_raw` into six independent binary columns: `contrib_technical`, `contrib_creative`, `contrib_organization`, `contrib_writing`, `contrib_morale`, `contrib_qa`.

### Step 7 — Missing Value Handling
GPA: 2 rows imputed with the **median** ordinal band (median = 4, corresponding to 3.5–4.0). Median preserves the ordinal encoding invariant that every value maps to a valid category.

### Step 8 — Min-Max Normalization
Scale all non-binary numeric features to [0, 1]. Binary features are not re-scaled. Without normalization, skill ratings (raw range 1–5) would contribute up to 16× more to Euclidean distance than binary availability features — effectively ignoring schedule compatibility.

### Step 9 — Row Shuffle & Save
Row order is shuffled (`sample(frac=1, random_state=42)`) before saving. Privacy protection, not a modeling step.

### Feature Set Construction

| Feature Set | Columns | Used By |
|---|---|---|
| `compatibility` (29 features) | 11 availability + 18 work-style | K-Means, Agglomerative, Hungarian |
| `complementarity` (8 features) | All 8 `skill_*` columns | GMM |
| `all_features` (37 features) | Both sets combined | PCA feature importance |

---

## 4. Models & Methodology

The end-to-end methodology:

![End-to-end methodology diagram. Survey → Preprocessing → two feature sets → four models → six evaluation metrics + PCA audit → instructor review.](https://raw.githubusercontent.com/HPAuncc/teammate-matcher/main/outputs/pipeline_diagram.png)

### 4.1 Why Four Models?

| Model | Feature Set | Objective | Size Constraint | Beyond-Class |
|---|---|---|---|---|
| K-Means | Compatibility | Similarity | No | No |
| Agglomerative (Ward) | Compatibility | Similarity | No | No |
| Hungarian Assignment | Compatibility | Similarity + Size | **Yes** | **Yes** |
| GMM | Complementarity | Skill diversity | No | **Yes** |

### 4.2 K-Means Clustering (Baseline)

K-Means partitions students into k clusters by minimizing within-cluster sum of squared distances. We fix k = 8 to match the target deployment scale of 3–5 students per team with N = 31. An unconstrained silhouette sweep over k ∈ [2, 8] shows the silhouette-maximizing value is k = 2 (silhouette 0.124), but this yields teams of ~15 and ~16 students — unusable for deployment. This divergence between "best clustering" and "best team configuration" is exactly why Hungarian assignment is needed.

**Limitation:** Assumes spherical, equal-variance clusters and does not enforce team size constraints.

### 4.3 Agglomerative Hierarchical Clustering (Ward Linkage)

Agglomerative clustering builds clusters bottom-up using Ward linkage, which merges the pair of clusters whose union minimizes the increase in within-cluster variance. Produces a **dendrogram** useful for instructor interpretability. Makes no spherical cluster assumption.

**Limitation:** Does not enforce team size constraints.

### 4.4 Hungarian Algorithm (Size-Constrained Assignment) ★

**This is the primary deployment model and satisfies the "beyond class" rubric requirement.**

The Hungarian Algorithm solves the balanced assignment problem directly. Given N students and k teams with target size t = ⌊N/k⌋:

1. Build a cost matrix C where C_ij is the Euclidean distance from student i to centroid j (K-Means centroids).
2. Expand the matrix by replicating each centroid column t times to form an N×N matrix.
3. Solve the linear sum assignment problem: find the permutation minimizing total assignment cost.
4. Each student's team label is their assigned expanded column mod k.

By replicating each centroid exactly t times, the permutation constraint forces each centroid to be assigned to exactly t students — guaranteeing balanced team sizes. Overflow students (N mod k ≠ 0) are assigned greedily to their nearest centroid. With N = 31, k = 8, t = 3, seven overflow students are assigned this way.

We use `scipy.optimize.linear_sum_assignment`, which implements the Jonker-Volgenant refinement running in O(N³).

### 4.5 Gaussian Mixture Model (GMM) ★

**GMM also satisfies the "beyond class" rubric requirement.**

GMM models the data as a mixture of k multivariate Gaussian distributions and estimates parameters via the **Expectation-Maximization (EM)** algorithm, alternating between computing posterior responsibilities (E-step) and re-estimating component parameters weighted by those responsibilities (M-step).

Unlike K-Means, GMM produces **soft assignments** — a probability vector over components for each student. A student with max posterior = 0.51 is *genuinely ambiguous* between two skill archetypes and can be flagged for human review. We applied GMM to the **complementarity feature set** (8 skill dimensions only) to identify latent skill archetypes and ensure every deployed team contains skill diversity.

**Model selection (BIC):** We sweep k ∈ [2, 8] and select the value minimizing the Bayesian Information Criterion, which penalizes model complexity to prevent overfitting.

#### 4.5.4 Ambiguity Flagging — Mechanism and Empirical Result

Students with max posterior < 0.60 are flagged as *ambiguous* for instructor review.

**Empirical result on this cohort:** With N = 31, k = 8 (BIC-selected), and full covariance matrices, GMM converges to extremely sharp posteriors. The maximum component probability is ≥ 0.999 for every student:

![GMM soft-assignment heatmap (31 × 8). Every student maps to exactly one archetype with probability ≈ 1.0; off-diagonal entries are visually empty.](https://raw.githubusercontent.com/HPAuncc/teammate-matcher/main/outputs/gmm_ambiguity.png)

| Quantity | Value |
|---|---|
| min(max posterior) | ≈ 1.00 |
| Students flagged (< 0.60) | **0 / 31** |
| Students with max posterior ≥ 0.80 | 31 / 31 |

This is a known small-sample behavior of unconstrained-covariance GMMs, not a failure of the mechanism. The mechanism remains useful for larger or more heterogeneous future deployments.

---

## 5. Evaluation Results

### 5.1 Metric Definitions

**Algorithmic metrics:**

| Metric | Direction |
|---|---|
| Silhouette Score | ↑ higher = better |
| Davies-Bouldin Index | ↓ lower = better |
| Calinski-Harabasz Index | ↑ higher = better |

**Domain metrics:**

| Metric | Definition | Direction |
|---|---|---|
| Intra-team Skill Variance | Mean std. dev. of skill ratings within each team | ↕ context-dependent |
| Schedule Overlap | Mean Jaccard similarity of availability vectors across within-team pairs | ↑ higher = better |
| Skill Coverage | Mean # skill dimensions where ≥1 team member scores ≥ 3/5 | ↑ higher = better |

### 5.2 Comparison Table

| Model | k | Team Sizes | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Skill Variance | Schedule Overlap ↑ | Skill Coverage ↑ |
|---|---|---|---|---|---|---|---|---|
| K-Means | 8 | 2–10 | 0.0944 | 1.5135 | 3.1690 | 0.1969 | **0.6278** | **7.875** |
| Agglomerative (Ward) | 8 | 2–7 | 0.0938 | 1.5959 | 3.1640 | 0.2042 | 0.6174 | 7.75 |
| **Hungarian Assignment** | 8 | **3–6** | 0.0430 | 1.7732 | 2.7089 | 0.2125 | 0.6069 | **7.875** |
| GMM | 8 | 3–6 | **0.1502** | **1.3610** | **6.3377** | 0.1359 | 0.4845 | 6.75 |

![Comparison of six evaluation metrics across all four models.](https://raw.githubusercontent.com/HPAuncc/teammate-matcher/main/outputs/comparison_metrics.png)

### 5.3 Interpretation

- **GMM wins all three algorithmic metrics** — it operates on a lower-dimensional 8-feature space with full covariance, giving clusters room to separate.
- **Hungarian and K-Means tie on highest Skill Coverage (7.875/8)** — balanced team sizes ensure enough members to cover all skill dimensions collectively.
- **K-Means has the best Schedule Overlap (0.6278)** — no balancing constraint, so it clusters tightly around schedule centroids. Hungarian's 3.3% drop in overlap is the cost of ensuring every team is 3–6 students.
- **GMM has the lowest Schedule Overlap (0.4845)** — it ignores schedule entirely. This quantifies the trade-off between the two objectives.
- **Skill Variance inversion:** GMM produces the *lowest* intra-team skill variance (0.136) because it groups by skill similarity, not diversity. Hungarian's forced balancing pulls in members from across archetypes, producing *more* skill-diverse teams (0.213) than the diversity-specific GMM.

### 5.4 GPA Sensitivity Analysis

K-Means was run with and without `gpa_band` appended to the compatibility feature set, and the two label vectors compared using Adjusted Rand Index (ARI):

| Quantity | Value |
|---|---|
| Adjusted Rand Index (with vs. without GPA) | **0.3397** |
| Schedule Overlap — with GPA | 0.6044 |
| Schedule Overlap — without GPA | 0.6278 |

An ARI of 0.34 is moderate — removing GPA reshuffles a meaningful fraction of team memberships. Crucially, schedule overlap is slightly *higher without GPA*, meaning GPA is actively pulling the clustering away from its primary objective. **Recommended practice:** present the instructor with both configurations and let them decide whether GPA should be included.

---

## 6. Feature Importance via PCA

### 6.1 Variance Explained

| Component | Individual Variance | Cumulative |
|---|---|---|
| PC1 | 15.80% | 15.80% |
| PC2 | 13.46% | 29.26% |
| PC3 | 9.00% | 38.26% |
| PC4 | 8.65% | 46.91% |
| 11 components to reach 80% | | |

### 6.2 Biplot Interpretation

![PCA biplot: 31 students in PC1–PC2 space, colored by Hungarian team. Top 8 feature loadings shown as red arrows.](https://raw.githubusercontent.com/HPAuncc/teammate-matcher/main/outputs/pca_biplot.png)

**Top features on PC1** (15.8% of variance — day-of-week availability axis):

| Feature | Loading |
|---|---|
| `avail_sat` | +0.378 |
| `avail_sun` | +0.378 |
| `avail_tue` | +0.370 |
| `avail_thu` | +0.333 |
| `avail_mon` | −0.255 |

**Top features on PC2** (13.46% — work-style + meeting-mode axis):

| Feature | Loading |
|---|---|
| `conflict_direct` | +0.409 |
| `meeting_nopref` | +0.405 |
| `avail_evening` | +0.391 |
| `conflict_natural` | −0.332 |
| `meeting_inperson` | −0.329 |

**Key finding — skills are not a top-2 axis.** The first skill feature (`skill_research`, loading −0.252) does not appear until PC3. Within this cohort, students differentiate primarily on *schedule and work-style* — not on self-rated technical skill. This supports running GMM on skills alone: if combined with schedule, skill-diversity signal would be drowned out by the higher-variance availability features.

**Answering RQ3:** The attributes most predictive of cluster separation are:
1. Weekend and midweek availability (PC1)
2. Conflict-handling style and meeting-mode preference (PC2)
3. Communication-channel preference (PC3)

---

## 7. Limitations & Assumptions

- **Sample size (N = 31):** Sufficient for this cohort but too small for statistical generalization. Silhouette scores are correspondingly low.
- **Single class section:** Results may not transfer to other disciplines or course levels.
- **Equal ordinal spacing:** Adjacent categories assumed equidistant — plausible but not strictly true for `weekly_hours`.
- **Euclidean distance on mixed features:** Future work should compare with Gower's similarity, which handles mixed binary/continuous data natively.
- **GMM Gaussian assumption:** Skill ratings on a 1–5 Likert scale are not truly Gaussian. BIC selection partially mitigates overfitting risk.
- **No ground-truth labels:** We cannot measure whether algorithmically formed teams produce better outcomes than random assignment without a post-project satisfaction survey.

---

## 8. Ethics, Bias & Equity

### 8.1 Human-in-the-Loop Design

The system is explicitly a **recommendation tool**, not an autonomous decision-maker. Its output is a *set* of candidate team configurations with interpretable metrics. The instructor makes final decisions and can override based on context the algorithm cannot see — interpersonal conflicts, disability accommodations, previous team history, or external commitments.

### 8.2 Fairness Considerations

**Demographic variables explicitly excluded.** The survey does not collect race, gender, ethnicity, nationality, disability, or socioeconomic status. GPA is the only potentially demographic-correlated variable, collected optionally with an explicit "Prefer not to say" option.

**Self-report bias and equity.** Students from backgrounds where confidence expression is culturally discouraged may systematically under-rate themselves. Mitigations:
1. Survey items framed as *comfort* rather than *ability*.
2. Hungarian's size constraint forces diverse teams, pairing low-self-rated students with high-self-rated ones.
3. Skill Coverage metric measures whether *anyone* on the team reaches each threshold.

### 8.3 Privacy

Raw CSV excluded from public repository via `.gitignore`. Processed CSV is row-shuffled before saving, breaking the correlation between row position and submission order. Published processed data contains no names, emails, IDs, or timestamps.

### 8.4 Equity Constraint

When deploying the GMM model, no team should be composed entirely of students self-rating below threshold on any single skill dimension. Currently documented as a deployment requirement; not yet implemented in code.

### 8.5 Reflection on Algorithmic Authority

The existence of an algorithmic recommendation creates anchoring pressure — instructors may accept recommendations by default. To counter this, the system surfaces *multiple* model configurations rather than a single ranked output, and reports an ambiguous-student flag from GMM. This cohort produced zero flagged students (Section 4.5.4), but the mechanism remains in the system for future deployments.

---

## 9. Conclusion

The Teammate Matcher demonstrates that optimal academic team formation requires treating *similarity* and *complementarity* as distinct objectives with distinct feature sets and distinct algorithms. **The choice of "best" model depends on what you optimize for.**

**Primary findings:**
1. **Hungarian Algorithm is the recommended deployment model** — guarantees balanced team sizes and ties for highest skill coverage (7.875/8 dimensions per team).
2. **GMM complements Hungarian** via latent skill archetype identification and ambiguity flagging.
3. **Schedule and work-style dominate student differentiation; self-rated skills are a weaker axis** (PC1 = availability 15.8%, PC2 = conflict/meeting style 13.5%; skills not in top loadings until PC3).
4. **GPA has moderate, not negligible, influence** (ARI = 0.34 between with/without-GPA clusterings; schedule overlap slightly *higher* without GPA).

**Future work:**
- Post-project team-satisfaction survey to validate real-world outcomes.
- Multi-section deployment to test generalization.
- Implement minimum-skill-diversity equity constraint in code.
- Compare Euclidean distance against Gower's similarity.

---

## 10. Code Organization & Reproducibility

Full repository: [github.com/HPAuncc/teammate-matcher](https://github.com/HPAuncc/teammate-matcher)

```
teammate-matcher/
├── data/
│   ├── raw_survey_responses.csv          # local only (gitignored)
│   └── processed_survey_data.csv         # shuffled, published
├── survey/
│   └── survey_questions.md
├── src/
│   ├── preprocess.py                     # 9-step pipeline
│   ├── models.py                         # 4 model wrappers
│   └── evaluate.py                       # 6 evaluation metrics
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_models_1_2.ipynb               # K-Means + Agglomerative
│   ├── 04_models_3_4.ipynb               # Hungarian + GMM
│   └── 05_evaluation.ipynb               # comparison + PCA biplot
└── outputs/
    ├── pipeline_diagram.png
    ├── schedule_heatmap.png
    ├── pca_biplot.png
    ├── comparison_metrics.png
    ├── poster_comparison_table.png
    ├── gmm_ambiguity.png
    └── evaluation_metrics.csv
```

All random processes seeded with `random_state = 42`. Notebooks are numbered and should be run in sequence.

---

## 11. References & AI Transparency

[1] M. Kyprianidou, S. Demetriadis, T. Tsiatsos, and A. Pombortsis, "Group formation based on learning styles: can it improve students' teamwork?" *Educational Technology Research and Development*, vol. 60, pp. 83–110, 2012.

[2] H. W. Kuhn, "The Hungarian method for the assignment problem," *Naval Research Logistics Quarterly*, vol. 2, no. 1–2, pp. 83–97, 1955.

[3] S. Akgun and C. Greenhow, "Artificial intelligence in education: Addressing ethical challenges in K-12 settings," *AI and Ethics*, vol. 2, 2022.

[4] C. M. Bishop, *Pattern Recognition and Machine Learning*. Springer, 2006.

[5] R. Jonker and A. Volgenant, "A shortest augmenting path algorithm for dense and sparse linear assignment problems," *Computing*, vol. 38, pp. 325–340, 1987.

**Software:** Python 3.12, pandas, NumPy, scikit-learn (K-Means, Agglomerative, GMM, PCA), SciPy (Hungarian Algorithm via `linear_sum_assignment`), matplotlib, seaborn.

**AI Tool Transparency:** Claude (Anthropic) was used as a coding assistant throughout development.

**Repository:** [github.com/HPAuncc/teammate-matcher](https://github.com/HPAuncc/teammate-matcher)
