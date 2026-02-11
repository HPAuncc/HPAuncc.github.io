---
layout: default
title: "Home-Field Advantage in the NFL (2000–2023)"
nav_exclude: true
---


# Home-Field Advantage in the NFL (2000–2023)

**Course:** DTSC-2301 — Data Science Foundations  
**Tools:** Python, pandas, matplotlib, seaborn  
**Dataset:** nflverse (1999–present game-level data)  

---

## Research Question

> To what extent does home-field advantage influence NFL game outcomes, and how has its impact changed from 2000–2023?

Home-field advantage is widely accepted in sports culture. Teams are assumed to perform better at home due to crowd support, reduced travel, and familiarity with playing conditions. However, modern changes in league structure, travel logistics, and the COVID-19 season raise an important question:

Has home-field advantage weakened over time?

---

## Dataset Overview

The analysis uses publicly available NFL game-level data from nflverse:

https://github.com/nflverse/nfldata/raw/master/data/games.csv

Each row represents one NFL game and includes:

- Season and week  
- Home and away teams  
- Final scores  
- Game type (REG, POST, PRE)  
- Stadium and weather conditions  
- Overtime indicator  

The dataset was filtered to:

- Seasons 2000–2023  
- Regular season games only  

Derived variables created:

- **Home Margin** = home_score − away_score  
- **Home Win Indicator** (1 = win, 0 = loss)  

---

## Key Findings

### 1️⃣ Home-Field Advantage Exists

Across 2000–2023, home teams win a clear majority of games. The advantage is statistically meaningful and persistent.

---

### 2️⃣ Home Advantage Has Declined Slightly

The trendline shows a gradual narrowing of home win percentage since the early 2000s.

![Home Win Rate Trend](/assets/images/home-field-advantage/home_win_trend.png)


The decline becomes most noticeable:

- Post-2015  
- Most sharply during the 2020 season  

---

### 3️⃣ The 2020 Season Is a Structural Outlier

The COVID-19 season shows a pronounced dip in home win rate.

This suggests crowd presence may meaningfully contribute to home advantage.

---

### 4️⃣ Rolling Trend Confirms Long-Term Softening

A five-year rolling average smooths volatility and confirms that home-field advantage, while present, is less dominant than in the early 2000s.

![Rolling Average Trend](/assets/images/home-field-advantage/rolling_avg.png)

---

### 5️⃣ Margin Distribution Shows Moderate Edge

The scoring margin distribution confirms that home teams not only win more often, but win by modest scoring margins.

![Margin Distribution](/assets/images/home-field-advantage/margin_distribution.png)

---

## What Would Be Misleading to Conclude

It would be incorrect to conclude that:

- Home-field advantage has disappeared  
- Crowd noise alone explains the effect  
- The trend proves causation  
- Modern NFL has no meaningful home bias  

The analysis is descriptive, not causal.

---

## What the Data Does Not Capture

Several contextual factors are not controlled for:

- **Travel Distance:** Cross-country travel and time zones  
- **Neutral Site Games:** International and alternate stadium games  
- **Attendance Levels:** Crowd size not directly measured  
- **Team Strength Controls:** No adjustments for roster or quarterback quality  

This limits causal interpretation.

---

## Ethical Reflection

Sports analytics influences:

- Betting markets  
- Media narratives  
- Performance evaluation  

A simplified conclusion such as “home advantage is disappearing” could influence betting behavior or be misapplied in predictive models.

Responsible interpretation requires transparency about uncertainty and limitations.

---

## Final Takeaway

Home-field advantage remains real in the NFL, but its magnitude has modestly declined over the past two decades, with the sharpest disruption occurring during the 2020 season.

The data suggests that structural and environmental factors influence competitive balance, but further analysis would be required to identify causal drivers.

---

## View Full Code

GitHub Repository:  
[DTSC-2301-Project-1](https://github.com/YOUR_USERNAME/DTSC-2301-Project-1)
