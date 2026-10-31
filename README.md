# DATA3406 Assignment 2: Personal informatics with Fitbit step data

## Introduction

**Driving question (Q1):** Do the people in this dataset achieve at least 7,000 steps on most days by avoiding long periods of inactivity?

**Why it matters:** about 7,000 steps a day is linked to a substantially lower risk of death, and if breaking up long inactive periods is how people reach that goal, "get up more often" becomes simple, personal advice a tracker can give.

## Team

| Name | Roles | Personal Planning and Progress |
|---|---|---|
| Ritishka Gupta | Individual group: Manager, Tracker, Analyst and Author | [Personal Planning and Progress ‐ Ritishka Gupta](https://github.com/rgup0018/assignment-2-data3406/wiki/Personal-Planning-and-Progress-%E2%80%90-Ritishka-Gupta) · [TPP10 progress](https://github.com/rgup0018/assignment-2-data3406/wiki/TPP10_Ritishka_Gupta_Personal_Planning_and_Progress) |

## Ethical analysis

[Ethical Analysis (wiki)](https://github.com/rgup0018/assignment-2-data3406/wiki/Ethical-Analysis): stakeholder analysis (the 33 participants; health promoters and app designers), characterisation of the raw data, data ownership, access and privacy, and how well the data can answer the driving question.

## Use of Issues and wiki for group processes

- [Team Contributions (wiki)](https://github.com/rgup0018/assignment-2-data3406/wiki/Team-Contributions): each contribution, with links to Issues, commits and wiki pages.
- [Meeting Minutes (wiki)](https://github.com/rgup0018/assignment-2-data3406/wiki/Meeting-Minutes): time, attendance, scrum report, discussion, decisions and sign-off for each meeting, linked to Issues.
- [GitHub Issues](https://github.com/rgup0018/assignment-2-data3406/issues?q=is%3Aissue): one Issue per task (#1–#11), assigned, with acceptance criteria, and closed with a comment linking the commit once done.

## Group reflection on group processes and product notebook

[Group Reflection (wiki)](https://github.com/rgup0018/assignment-2-data3406/wiki/Group-Reflection): the Big-5 teamwork reflection (planning, back-up behaviour, adaptation, mutual performance monitoring, closed-loop communication), a critique of the product notebook, and the think-aloud evaluation.

## Product notebook code

[Product Notebook - Q1 Steps and Inactivity.ipynb](Product%20Notebook%20-%20Q1%20Steps%20and%20Inactivity.ipynb)

It imports the pipeline module [`q1_steps.py`](q1_steps.py). Run all cells with the three CSVs in [`data/`](data); it takes about 30 seconds with Python 3, pandas, NumPy and Matplotlib. The process notebooks it builds on are [Checkpoint 1](Checkpoint%201-Ritishka%20Gupta-Step%20data%20exploration.ipynb) and [Checkpoint 2](Checkpoint%202-Ritishka%20Gupta-Step%20data%20analysis.ipynb).

## Product notebook results and answer to the driving question

**Data pipeline:** raw data (three CSVs, quality checks) → cleaned data (minute file reshaped to 1.3 million person-minutes; the three files checked against each other; daily steps taken from the minute data because the daily file stops part-way through 12 May) → adherence (valid days and people) → descriptive analytics → Part A, Part B and robustness → answer.

**Adherence (Tang et al., 2018):** I classified every person-day as valid (complete, with at least 10 hours containing a step), low-wear or incomplete, and compared four rules for which days count. The share of goal days rises from 54% (every day in the daily file) to 65% (valid days), and the number of people meeting "most days" from 18 to 22. Four people with fewer than 10 valid days get no verdict. All four had median daily steps below 7,000 on the days they wore the tracker, so excluding them probably makes the group look slightly more active than it is.

**Answer:**
- **Part A, 7,000 steps on most days: yes, for most people.** 22 of the 29 people with enough data reach 7,000 steps on more than half of their valid days, 14 of them clearly (95% Wilson interval above 50%). 14 of 29 meet the stricter 5 days in 7.
- **Part B, by avoiding long inactive periods: yes, by *interrupting* them rather than being inactive less.** Comparing each person's goal days with their own other days, 18 of 21 people spend less time in long inactive periods (≥ 60 min) on goal days, by a median of 91 minutes (sign test P = 0.0015). Their total inactive time hardly changes (median −54 minutes). Instead, short inactive spells increase for 20 of 21 people: extra walking (×1.9 more walking minutes, but only ×1.2 faster) breaks long periods into short ones.
- The result holds for long-period thresholds from 30 to 90 minutes and for other valid-day rules, but not under a strict "0 steps = inactive" rule. It is an association in one month of data from 33 volunteers.

## Repository contents

| Path | What it is |
|---|---|
| `Product Notebook - Q1 Steps and Inactivity.ipynb` | Final product notebook |
| `q1_steps.py` | Reusable pipeline: loading, cleaning, adherence, inactivity measures and statistics |
| `Checkpoint 1-…ipynb`, `Checkpoint 2-…ipynb` | Individual process notebooks |
| `data/` | The three supplied files from FitBit Fitness Tracker Data (Kaggle, CC0: Public Domain) |
