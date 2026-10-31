"""Reusable data pipeline for the DATA3406 Assignment 2 product notebook.

Driving problem (Q1): do the people in the Fitbit dataset achieve at least 7,000 steps
on most days by avoiding long periods of inactivity?

Author: Ritishka Gupta. The per-person versions of `person_minutes`, `day_table`,
`estimate_waking_start` and `inactive_runs` were first written in my Checkpoint 2
notebook; here they are rewritten to process all 33 people at once. GenAI use is
documented in the last section of the product notebook.
"""

from dataclasses import dataclass
from math import comb, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Definitions:
    """Every analysis choice in one place (rationale in the notebook's Definitions table).

    Use `dataclasses.replace(DEFS, long_bout_min=30)` to rerun the pipeline with
    a different choice, as the sensitivity checks do.
    """

    step_goal: int = 7000            # daily step goal from the driving problem
    min_valid_hours: int = 10        # valid day: >= 10 hours containing at least one step
    waking_hours: int = 16           # length of each person's estimated waking window
    inactive_max_steps: int = 19     # inactive minute: 0-19 steps (no purposeful walking)
    long_bout_min: int = 60          # long inactive period: >= 60 consecutive inactive minutes
    most_days: float = 0.5           # "most days": more than half of valid days
    strict_days: float = 5 / 7       # stricter reading: at least 5 days in every 7
    min_valid_days: int = 10         # people need >= 10 valid days for a verdict
    min_days_per_group: int = 3      # within-person comparisons need >= 3 goal and 3 other days


DEFS = Definitions()


# ---------------------------------------------------------------- loading and cleaning
def load_raw(data_dir):
    """Read the three supplied CSV files and parse their date columns."""
    data_dir = Path(data_dir)
    daily = pd.read_csv(data_dir / "dailySteps_merged.csv")
    hourly = pd.read_csv(data_dir / "hourlySteps_merged.csv")
    minute_wide = pd.read_csv(data_dir / "minuteStepsWide_merged.csv")
    daily["ActivityDay"] = pd.to_datetime(daily["ActivityDay"], format="%m/%d/%Y")
    hour_format = "%m/%d/%Y %I:%M:%S %p"
    hourly["ActivityHour"] = pd.to_datetime(hourly["ActivityHour"], format=hour_format)
    minute_wide["ActivityHour"] = pd.to_datetime(
        minute_wide["ActivityHour"], format=hour_format
    )
    return daily, hourly, minute_wide


def quality_report(frames):
    """Rows, people, missing values, duplicate rows, negative steps and date range per file."""
    rows = {}
    for name, (frame, time_col) in frames.items():
        steps = frame.filter(like="Step")
        rows[name] = {
            "rows": len(frame),
            "people": frame["Id"].nunique(),
            "missing values": int(frame.isna().sum().sum()),
            "duplicate rows": int(frame.duplicated().sum()),
            "duplicate Id-time keys": int(frame.duplicated(["Id", time_col]).sum()),
            "negative step values": int((steps < 0).sum().sum()),
            "first": frame[time_col].min(),
            "last": frame[time_col].max(),
        }
    return pd.DataFrame(rows).T


def to_long(minute_wide):
    """Reshape the wide minute file (one row per hour, Steps00-Steps59) to one row per minute."""
    long = minute_wide.melt(
        id_vars=["Id", "ActivityHour"], var_name="col", value_name="Steps"
    )
    long["Timestamp"] = long["ActivityHour"] + pd.to_timedelta(
        long["col"].str[-2:].astype(int), unit="min"
    )
    long["Day"] = long["Timestamp"].dt.normalize()
    long["Hour"] = long["Timestamp"].dt.hour
    return (
        long.drop(columns=["col", "ActivityHour"])
        .sort_values(["Id", "Timestamp"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------- day-level table
def day_table(minutes, daily, defs=DEFS):
    """One row per person-day: minute-based steps, wear (adherence) flags and goal status."""
    keys = ["Id", "Day"]
    days = minutes.groupby(keys).agg(Minutes=("Steps", "size"), Steps=("Steps", "sum"))
    days["ValidHours"] = (
        minutes[minutes["Steps"] > 0].groupby(keys)["Hour"].nunique()
    )
    days["ValidHours"] = days["ValidHours"].fillna(0).astype(int)
    days["Complete"] = days["Minutes"] == 24 * 60
    days["Valid"] = days["Complete"] & (days["ValidHours"] >= defs.min_valid_hours)
    days["Goal"] = days["Steps"] >= defs.step_goal
    file_steps = daily.rename(columns={"ActivityDay": "Day"}).set_index(keys)["StepTotal"]
    days["DailyFileSteps"] = file_steps.reindex(days.index)

    # Purposeful walking: minutes above the inactive threshold, across the whole day
    walking = minutes[minutes["Steps"] > defs.inactive_max_steps]
    days["ActiveMinutes"] = walking.groupby(keys).size().reindex(days.index, fill_value=0)
    days["ActiveSteps"] = (
        walking.groupby(keys)["Steps"].sum().reindex(days.index, fill_value=0)
    )
    return days


def estimate_waking_start(minutes, days, defs=DEFS):
    """Start hour of each person's `waking_hours`-long window in which they most often move.

    For each hour of the day, take the share of the person's valid days with at least
    one step in that hour, then choose the start hour whose window has the largest
    total share. The window does not wrap past midnight. People with no valid days
    get 07:00.
    """
    valid_keys = days.index[days["Valid"]]
    on_valid = minutes.set_index(["Id", "Day"]).index.isin(valid_keys)
    hour_has_steps = (
        minutes[on_valid].groupby(["Id", "Day", "Hour"])["Steps"].sum().gt(0)
    )
    share = (
        hour_has_steps.groupby(["Id", "Hour"]).mean().unstack(fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )
    last_start = 24 - defs.waking_hours
    windows = pd.DataFrame(
        {s: share.loc[:, s : s + defs.waking_hours - 1].sum(axis=1)
         for s in range(last_start + 1)}
    )
    starts = windows.idxmax(axis=1).astype(int)
    all_ids = days.index.get_level_values("Id").unique()
    return starts.reindex(all_ids, fill_value=7).rename("WakeStart")


def inactive_runs(minutes, wake_start, defs=DEFS):
    """Every run of consecutive inactive minutes inside each day's waking window.

    Only time between the first and last step of the day inside the window counts:
    before the first step or after the last one, the person is probably asleep or not
    wearing the device, so those zeros are not evidence of inactivity.
    """
    start = minutes["Id"].map(wake_start)
    window = minutes[
        (minutes["Hour"] >= start) & (minutes["Hour"] < start + defs.waking_hours)
    ]
    keys = ["Id", "Day"]
    stepping = window[window["Steps"] > 0].groupby(keys)["Timestamp"].agg(["min", "max"])
    bounds = stepping.reindex(pd.MultiIndex.from_frame(window[keys]))
    bounds.index = window.index
    awake = window[window["Timestamp"].between(bounds["min"], bounds["max"])]

    inactive = awake["Steps"].le(defs.inactive_max_steps)
    same_day = awake["Id"].eq(awake["Id"].shift()) & awake["Day"].eq(awake["Day"].shift())
    run_id = (inactive.ne(inactive.shift()) | ~same_day).cumsum()
    runs = (
        awake[inactive]
        .groupby(run_id[inactive])
        .agg(Id=("Id", "first"), Day=("Day", "first"),
             Start=("Timestamp", "first"), Minutes=("Steps", "size"))
        .reset_index(drop=True)
    )
    runs["End"] = runs["Start"] + pd.to_timedelta(runs["Minutes"], unit="min")
    awake_minutes = awake.groupby(keys).size().rename("AwakeMinutes")
    first_step = stepping["min"]
    first_step_hour = ((first_step - first_step.index.get_level_values("Day"))
                       / pd.Timedelta(hours=1)).rename("FirstStepHour")
    return runs, awake_minutes, first_step_hour


def analyse(minutes, daily, defs=DEFS):
    """Whole pipeline: day table + waking windows + inactivity measures for every person."""
    days = day_table(minutes, daily, defs)
    wake = estimate_waking_start(minutes, days, defs)
    runs, awake_minutes, first_step_hour = inactive_runs(minutes, wake, defs)
    long_bouts = runs[runs["Minutes"] >= defs.long_bout_min]
    by_day = ["Id", "Day"]

    days["AwakeMinutes"] = awake_minutes.reindex(days.index, fill_value=0)
    days["FirstStepHour"] = first_step_hour.reindex(days.index)
    days["InactiveMinutes"] = (
        runs.groupby(by_day)["Minutes"].sum().reindex(days.index, fill_value=0)
    )
    days["LongestInactive"] = (
        runs.groupby(by_day)["Minutes"].max().reindex(days.index, fill_value=0)
    )
    days["LongBouts"] = long_bouts.groupby(by_day).size().reindex(days.index, fill_value=0)
    days["LongBoutMinutes"] = (
        long_bouts.groupby(by_day)["Minutes"].sum().reindex(days.index, fill_value=0)
    )
    return {"days": days, "runs": runs, "bouts": long_bouts, "wake": wake, "defs": defs}


# ---------------------------------------------------------------- statistics
def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n (works well for small n)."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    p = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def sign_test(n_positive, n_total):
    """Two-sided exact sign test: P-value for n_positive of n_total under a 50:50 null."""
    if n_total == 0:
        return np.nan
    tail = min(n_positive, n_total - n_positive)
    p = sum(comb(n_total, i) for i in range(tail + 1)) / 2**n_total
    return min(1.0, 2 * p)


def rank_correlation(x, y):
    """Spearman rank correlation; NaN with < 3 values or when a measure never changes."""
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return x.rank().corr(y.rank())


# ---------------------------------------------------------------- person-level summaries
def person_summary(result):
    """One row per person: adherence, share of goal days with a 95% interval, and verdict."""
    defs = result["defs"]
    days = result["days"]
    valid = days[days["Valid"]]
    table = pd.DataFrame(
        {
            "days with minute data": days.groupby("Id").size(),
            "valid days": valid.groupby("Id").size(),
            "goal days": valid.groupby("Id")["Goal"].sum(),
            "median steps": valid.groupby("Id")["Steps"].median(),
            "median long-inactive min": valid.groupby("Id")["LongBoutMinutes"].median(),
        }
    ).fillna({"valid days": 0, "goal days": 0})
    table[["valid days", "goal days"]] = table[["valid days", "goal days"]].astype(int)
    n, k = table["valid days"], table["goal days"]
    table["% goal days"] = 100 * k / n.where(n > 0)
    low, high = wilson_interval(k, n)
    table["95% CI low"], table["95% CI high"] = 100 * low, 100 * high
    table["wake start"] = result["wake"].reindex(table.index)
    table["verdict"] = np.select(
        [
            n < defs.min_valid_days,
            low > defs.most_days,
            table["% goal days"] > 100 * defs.most_days,
            high < defs.most_days,
        ],
        [
            "too few valid days",
            "yes (clearly)",
            "yes (uncertain)",
            "no (clearly)",
        ],
        default="no (uncertain)",
    )
    return table.rename_axis("Id")


def within_person(result):
    """For each person with enough goal and other days, compare the two kinds of day.

    Comparing a person with themselves removes between-person differences (job,
    fitness, how they wear the device), so the comparison is about *their* good and
    bad days rather than about active and inactive people.
    """
    defs = result["defs"]
    valid = result["days"][result["days"]["Valid"]].copy()
    valid["Kind"] = np.where(valid["Goal"], "goal", "other")
    counts = valid.groupby(["Id", "Kind"]).size().unstack(fill_value=0)
    eligible = counts.index[
        (counts.get("goal", 0) >= defs.min_days_per_group)
        & (counts.get("other", 0) >= defs.min_days_per_group)
    ]
    valid = valid[valid.index.get_level_values("Id").isin(eligible)]

    with np.errstate(divide="ignore"):
        valid["lnActiveSteps"] = np.log(valid["ActiveSteps"])
        valid["lnActiveMinutes"] = np.log(valid["ActiveMinutes"])
        valid["lnCadence"] = valid["lnActiveSteps"] - valid["lnActiveMinutes"]
    medians = valid.groupby(["Id", "Kind"])[
        ["Steps", "LongBoutMinutes", "LongBouts", "LongestInactive", "FirstStepHour"]
    ].median().unstack()
    logs = valid.groupby(["Id", "Kind"])[
        ["lnActiveSteps", "lnActiveMinutes", "lnCadence"]
    ].mean().unstack()

    out = pd.DataFrame(index=eligible)
    out["goal days"] = counts.loc[eligible, "goal"]
    out["other days"] = counts.loc[eligible, "other"]
    for col in medians.columns.levels[0]:
        out[f"{col} (goal)"] = medians[(col, "goal")]
        out[f"{col} (other)"] = medians[(col, "other")]
    out["long-inactive min difference"] = (
        out["LongBoutMinutes (goal)"] - out["LongBoutMinutes (other)"]
    )
    out["rank corr"] = valid.groupby("Id").apply(
        lambda g: rank_correlation(g["LongBoutMinutes"], g["Steps"])
    )
    # Exact identity per day: ln(steps while walking) = ln(walking minutes) + ln(cadence)
    for part in ["lnActiveSteps", "lnActiveMinutes", "lnCadence"]:
        out[f"gap {part}"] = logs[(part, "goal")] - logs[(part, "other")]
    out["share of gap from more walking minutes"] = (
        out["gap lnActiveMinutes"] / out["gap lnActiveSteps"]
    )
    return out.rename_axis("Id")
