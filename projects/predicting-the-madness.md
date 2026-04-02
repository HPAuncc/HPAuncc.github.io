---
layout: post
title: "Predicting the Madness"
subtitle: "Can a Machine Actually Predict March Madness?"
thumbnail: /assets/images/march-madness/thumbnail.png
thumbnail_alt: "An infographic illustration depicting NCAA March Madness basketball analytics and predictive data modeling with various charts and graphs."
nav_exclude: true
---
Every March, 68 college basketball teams enter a tournament and millions of people fill out brackets convinced they've figured it out. Most of them haven't. Even the most informed fans rarely get more than 60% of games right.

So I decided to ask a different question: **what does the data actually say about who wins — and can a machine learning model do better than conventional wisdom?**

---

## The Data

I pulled historical NCAA tournament and regular season data going back to 2003 — over 120,000 games worth of box scores, seeds, and outcomes from the Kaggle *March Machine Learning Mania* competition. The dataset includes every Division I regular season game and every tournament game since 2003, along with third-party rankings like KenPom.

---

## What the Data Actually Shows

Before touching any model, the numbers tell an interesting story.

**Seeds matter — but not as much as you'd think.**

1-seeds win about 85% of their first-round games. But by the Sweet 16, the gap narrows fast. A 5-seed and a 3-seed have played nearly the same number of tournament games historically. The chart below (from the notebook EDA) shows win rate by seed across all tournament rounds since 2003 — notice how quickly the advantage flattens out past seed 4.

![Tournament Win Rate by Seed](https://raw.githubusercontent.com/HPAuncc/March-Madness-Model/main/visualizations/seed_win_rates.png)

**Games are close.** Nearly 29% of all tournament games since 2003 were decided by 5 points or fewer. The average winning margin is only 12 points. That means a lot of outcomes come down to a few possessions — which no model can reliably predict.

---

## How I Built the Model

Rather than feeding raw stats into a model, I engineered features that measure *relative* team strength — how much better or worse is Team A compared to Team B on each dimension? This includes:

- **Net efficiency** — the gap between points scored and allowed per possession
- **Effective field goal percentage** — accounts for the extra value of 3-pointers
- **Turnover rate** — teams that take care of the ball consistently go further
- **Offensive rebound rate** — second-chance points matter in close games
- **Strength of schedule** — beating bad teams all year isn't the same as competing in a tough conference
- **Seed difference** — still a strong signal, since it reflects expert judgment from the selection committee

I tested four models: **Logistic Regression**, **K-Nearest Neighbors**, **Random Forest**, and **XGBoost**, then compared them on accuracy, log loss, and ROC-AUC.

---

## How Well Does It Work?

Short answer: better than chance, but not perfect.

| Model | Accuracy (2025 Tournament) | ROC-AUC |
|---|---|---|
| Logistic Regression | 76.1% | 0.869 |
| K-Nearest Neighbors | 85.1% | 0.881 |
| Random Forest | 73.9% | 0.883 |
| XGBoost | 75.4% | 0.848 |

On the 2025 tournament — data the model had never seen — all four models beat the 50% coin-flip baseline by a wide margin. The **Random Forest** was selected as the best performer based on log loss, which measures not just whether the model picked the right winner but how *confident* it was.

![Model Comparison](https://raw.githubusercontent.com/HPAuncc/March-Madness-Model/main/visualizations/model_comparison.png)

The model is good at picking expected winners. It's not good at picking 12-over-5 upsets — and neither is anyone else.

---

## What Actually Wins in March?

According to the feature importance analysis, the factors that matter most are:

1. **Seed difference** — by far the strongest predictor. The selection committee's judgment about team quality is hard to beat.
2. **Strength of schedule** — teams that earned their record against tough competition hold up better in March
3. **Net efficiency** — the gap between points scored and allowed per possession; the best efficiency-based predictor
4. **Offensive efficiency** — scoring efficiently matters more than scoring a lot

What matters less than people think: free throw rate and defensive rebounding in isolation. It's the combination of seed, schedule strength, and efficient play that separates tournament teams.

![Feature Importance - Random Forest](https://raw.githubusercontent.com/HPAuncc/March-Madness-Model/main/visualizations/feature_importance_rf.png)

---

## Why This Matters

This isn't just about basketball brackets. The same framework — using relative performance metrics to predict head-to-head outcomes — applies broadly in sports analytics, business competition modeling, and any domain where two entities compete on measurable dimensions. The result shows that efficiency-based metrics consistently outperform simple win-loss records as predictors, which has real implications for how teams are evaluated and seeded.

For March Madness specifically: the model confirms that seed is meaningful but not deterministic. Roughly 1 in 3 games is genuinely a toss-up based on the numbers. That's not a flaw in the model — that's March Madness.

---

## 2026 Predictions — In Progress

The model has the full 2026 bracket loaded with this season's regular season stats and is generating win probabilities for every matchup. The best teams by net efficiency are the clear favorites — but if history is any guide, at least one double-digit seed is going to break everyone's bracket.

The 2026 NCAA Championship game tips off on **April 7**. Once the tournament concludes, this post will be updated with a full breakdown of how the model performed on a live bracket it never saw — which games it got right, which upsets it missed, and whether a data-driven approach actually beats conventional wisdom in real time.

*Check back after April 7 for the final results.*

---

## References & AI Transparency

**Data source:** Kaggle — [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)

**Libraries used:** pandas, numpy, scikit-learn, XGBoost, matplotlib, seaborn

**AI tools used:** Claude (Anthropic) was used as a coding assistant to help structure and debug the notebook. All analysis, interpretation, and writing reflect my own understanding and decisions.

**Full code:** [github.com/HPAuncc/March-Madness-Model](https://github.com/HPAuncc/March-Madness-Model)
