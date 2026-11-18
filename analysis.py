
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Set a fixed seed so results can be reproduced
np.random.seed(42)
rng = np.random.default_rng(42)

# Read the application data
raw = pd.read_csv(
    "job-board-applications.csv",
    dtype=str,
    keep_default_na=False
)


# --------------------------------------------------
# 1. DATA CLEANING
# --------------------------------------------------

cleaning_log = []

# Remove duplicate records
df = raw.drop_duplicates().copy()

duplicates = len(raw) - len(df)
cleaning_log.append(
    f"Removed {duplicates} duplicate rows "
    f"({len(raw)} -> {len(df)})"
)

# Make work mode values consistent
changed_modes = (
    df["work_mode"] !=
    df["work_mode"].str.lower().str.replace("-", "_")
).sum()

df["work_mode"] = (
    df["work_mode"]
    .str.lower()
    .str.replace("-", "_")
)

cleaning_log.append(
    f"Standardised work mode in {changed_modes} rows"
)

# Replace missing application sources
missing_sources = (df["source"] == "").sum()
df["source"] = df["source"].replace("", "unknown")

cleaning_log.append(
    f"Replaced {missing_sources} blank sources with 'unknown'"
)

# Convert columns into appropriate data types
df["applied_on"] = pd.to_datetime(df["applied_on"])
df["stipend"] = pd.to_numeric(df["stipend_inr_month"])
df["steps_done"] = df["steps_done"].astype(int)

# Convert yes/no columns into 1 and 0
for column in ["shortlisted", "interviewed", "hired"]:
    df[column] = (df[column] == "yes").astype(int)

# Calculate reply information
df["reply_days"] = pd.to_numeric(
    df["days_to_first_reply"].replace("", np.nan)
)

df["replied"] = df["reply_days"].notna().astype(int)

# Create paid/unpaid and completion categories
df["paid"] = np.where(df["stipend"] > 0, "Paid", "Unpaid")

df["finished"] = np.where(
    df["steps_done"] == 4,
    "Finished all 4",
    "Did not finish"
)

# Check whether the data is consistent
assert (df["hired"] <= df["interviewed"]).all()
assert (df["interviewed"] <= df["shortlisted"]).all()
assert df["application_id"].is_unique
assert df["applied_on"].between(
    "2026-07-01", "2026-09-30"
).all()

cleaning_log.append(
    "Data checks passed: hiring stages, unique IDs and dates"
)

print("\n".join(cleaning_log))
print()


# --------------------------------------------------
# 2. BOOTSTRAP CONFIDENCE INTERVAL FUNCTION
# --------------------------------------------------

def boot_ci(group1, group2, statistic, n=5000):
    """
    Calculate a 95% confidence interval using bootstrap sampling.
    The two groups are sampled independently.
    """

    results = []

    for _ in range(n):
        sample1 = rng.choice(group1, size=len(group1))
        sample2 = rng.choice(group2, size=len(group2))

        results.append(statistic(sample1, sample2))

    return np.percentile(results, [2.5, 97.5])


# --------------------------------------------------
# 3. Q1 - HIRES PER 100 APPLICATIONS BY SOURCE
# --------------------------------------------------

def calculate_hire_rate(group):
    return pd.Series({
        "applications": len(group),
        "hires": group["hired"].sum(),
        "hires_per_100": group["hired"].mean() * 100
    })


# Exclude unknown sources from the source ranking
known_sources = df[df["source"] != "unknown"]

source_results = (
    known_sources.groupby("source")
    .apply(calculate_hire_rate)
    .sort_values("hires_per_100", ascending=False)
)

print("Q1: Overall hiring rate by source")
print(source_results.round(2))
print()


# Hiring rate by role and source
role_source = (
    known_sources
    .groupby(["role_category", "source"])["hired"]
    .agg(["size", "sum", "mean"])
    .assign(per100=lambda x: x["mean"] * 100)
    .reset_index()
)

rate_table = role_source.pivot(
    index="role_category",
    columns="source",
    values="per100"
).round(1)

count_table = role_source.pivot(
    index="role_category",
    columns="source",
    values="size"
)

print("Hiring rate by role and source")
print(rate_table)

print("\nApplication counts")
print(count_table)
print()


# Find the best-performing source for each role
best_sources = role_source.loc[
    role_source.groupby("role_category")["per100"].idxmax(),
    ["role_category", "source", "per100", "size", "sum"]
]

print("Best source for each role")
print(best_sources.to_string(index=False))
print()


# Compare referrals with other known sources
referrals = df[df["source"] == "referral"]["hired"].values

other_sources = df[
    ~df["source"].isin(["referral", "unknown"])
]["hired"].values

low, high = boot_ci(
    referrals,
    other_sources,
    lambda x, y: 100 * (x.mean() - y.mean())
)

print(
    f"Referral hiring rate: {referrals.mean()*100:.1f} per 100"
)

print(
    f"Other sources hiring rate: {other_sources.mean()*100:.1f} per 100"
)

print(
    f"Difference confidence interval: [{low:.1f}, {high:.1f}]"
)

# Confidence interval for referral hiring rate
referral_ci = np.percentile(
    [
        rng.choice(referrals, size=len(referrals)).mean() * 100
        for _ in range(5000)
    ],
    [2.5, 97.5]
)

print(
    f"Referral 95% CI: "
    f"[{referral_ci[0]:.1f}, {referral_ci[1]:.1f}]"
)
print()


# Compare hiring performance across months
print("Hiring rate by month")

for month, group in known_sources.groupby(
    known_sources["applied_on"].dt.month
):
    monthly_results = (
        group.groupby("source")["hired"]
        .mean()
        .mul(100)
        .round(1)
        .to_dict()
    )

    print(f"Month {month}: {monthly_results}")

print()


# Check completion rates by source
completion_rates = (
    df.groupby("source")
    .apply(lambda x: (x["steps_done"] == 4).mean() * 100)
    .round(1)
)

print("All four steps completed by source")
print(completion_rates.to_dict())
print()


# --------------------------------------------------
# 4. Q2 - COMPLETING ALL FOUR STEPS VS SHORTLISTING
# --------------------------------------------------

q2_results = (
    df.groupby("finished")["shortlisted"]
    .agg(["size", "sum", "mean"])
)

q2_results["percentage"] = q2_results["mean"] * 100

print("Q2: Shortlisting results")
print(q2_results.round(3))
print()


# Compare completed and incomplete applications
completed = df[df["steps_done"] == 4]["shortlisted"].values

incomplete = df[df["steps_done"] < 4]["shortlisted"].values

difference = 100 * (completed.mean() - incomplete.mean())

low, high = boot_ci(
    completed,
    incomplete,
    lambda x, y: 100 * (x.mean() - y.mean())
)

print(f"Difference: {difference:.1f} percentage points")
print(f"95% CI: [{low:.1f}, {high:.1f}]")
print(f"Shortlisting ratio: {completed.mean()/incomplete.mean():.1f}x")
print()


# Shortlisting rate for each number of completed steps
step_results = (
    df.groupby("steps_done")["shortlisted"]
    .mean()
    .mul(100)
    .round(1)
)

print("Shortlisting rate by completed steps")
print(step_results.to_dict())
print()


# Compare completion results across job roles
role_results = (
    df.groupby(["role_category", "finished"])["shortlisted"]
    .mean()
    .unstack() * 100
)

role_results["gap_pts"] = (
    role_results["Finished all 4"] -
    role_results["Did not finish"]
)

print("Shortlisting by role")
print(role_results.round(1))
print()


# Compare paid and unpaid applications
pay_results = (
    df.groupby(["paid", "finished"])["shortlisted"]
    .mean()
    .unstack() * 100
)

print("Shortlisting by paid status")
print(pay_results.round(1))
print()


# Check completion rate by payment status
paid_completion = (
    df.groupby("paid")
    .apply(lambda x: (x["steps_done"] == 4).mean() * 100)
    .round(1)
)

print("Completion rate by payment status")
print(paid_completion.to_dict())
print()


# --------------------------------------------------
# 5. Q3 - FIRST REPLY TIME: PAID VS UNPAID
# --------------------------------------------------

def reply_summary(group):
    return pd.Series({
        "applications": len(group),
        "no_reply_pct": (1 - group["replied"].mean()) * 100,
        "median_days": group["reply_days"].median(),
        "mean_days": group["reply_days"].mean(),
        "p75_days": group["reply_days"].quantile(0.75),
        "p90_days": group["reply_days"].quantile(0.90)
    })


reply_results = df.groupby("paid").apply(reply_summary)

print("Q3: Reply time comparison")
print(reply_results.round(2))
print()


# Separate paid and unpaid applications
paid_group = df[df["paid"] == "Paid"]
unpaid_group = df[df["paid"] == "Unpaid"]


# Compare no-reply percentages
low, high = boot_ci(
    paid_group["replied"].values,
    unpaid_group["replied"].values,
    lambda x, y: 100 * ((1 - x.mean()) - (1 - y.mean()))
)

no_reply_gap = (
    100 * (unpaid_group["replied"].mean() -
           paid_group["replied"].mean())
)

print(f"No-reply gap (unpaid - paid): {no_reply_gap:.1f} points")
print(f"95% CI: [{-high:.1f}, {-low:.1f}]")
print()


# Compare median reply times
paid_days = paid_group["reply_days"].dropna().values
unpaid_days = unpaid_group["reply_days"].dropna().values

low, high = boot_ci(
    paid_days,
    unpaid_days,
    lambda x, y: np.median(y) - np.median(x)
)

median_gap = np.median(unpaid_days) - np.median(paid_days)

print(f"Median reply gap (unpaid - paid): {median_gap:.1f} days")
print(f"95% CI: [{low:.1f}, {high:.1f}]")
print()


# Reply times by role and payment status
print("Reply time by role")

role_reply = (
    df.groupby(["role_category", "paid"])
    .apply(lambda x: pd.Series({
        "median_days": x["reply_days"].median(),
        "no_reply_pct": (1 - x["replied"].mean()) * 100,
        "applications": len(x)
    }))
    .unstack()
)

print(role_reply.round(1))
print()


# Reply times by work mode
print("Reply time by work mode")

mode_reply = (
    df.groupby(["work_mode", "paid"])
    .apply(lambda x: pd.Series({
        "median_days": x["reply_days"].median(),
        "no_reply_pct": (1 - x["replied"].mean()) * 100
    }))
    .unstack()
)

print(mode_reply.round(1))
print()


# Check whether application month affects no-reply rates
print("No-reply percentage by month and payment status")

monthly_reply = (
    df.groupby([df["applied_on"].dt.month, "paid"])
    .apply(lambda x: (1 - x["replied"].mean()) * 100)
    .unstack()
)

print(monthly_reply.round(1))
print()


# Group applications into stipend ranges
df["tier"] = pd.cut(
    df["stipend"],
    bins=[-1, 0, 10000, 20000, 40000],
    labels=["0", "1-10k", "10-20k", "20-40k"]
)

tier_results = (
    df.groupby("tier", observed=True)
    .apply(lambda x: pd.Series({
        "applications": len(x),
        "median_days": x["reply_days"].median(),
        "no_reply_pct": (1 - x["replied"].mean()) * 100
    }))
)

print("Reply time by stipend range")
print(tier_results.round(1))
print()


# Check paid application share for each role
paid_share = (
    df.groupby("role_category")
    .apply(lambda x: (x["paid"] == "Paid").mean() * 100)
    .round(1)
)

print("Paid application share by role")
print(paid_share.to_dict())
print()


# --------------------------------------------------
# 6. CHART - SHORTLISTING RATE BY COMPLETED STEPS
# --------------------------------------------------

chart_data = (
    df.groupby("steps_done")["shortlisted"]
    .agg(["size", "sum", "mean"])
)

# Calculate confidence intervals for each step group
confidence_intervals = {}

for step in chart_data.index:

    values = df.loc[
        df["steps_done"] == step,
        "shortlisted"
    ].values

    sample_size = len(values)

    bootstrap_rates = [
        rng.choice(values, size=sample_size).mean() * 100
        for _ in range(3000)
    ]

    confidence_intervals[step] = np.percentile(
        bootstrap_rates,
        [2.5, 97.5]
    )


# Prepare chart values
percentages = chart_data["mean"] * 100

fig, ax = plt.subplots(figsize=(8, 4.8))

bar_colors = ["#9ca3af"] * 4 + ["#c2410c"]

ax.bar(
    chart_data.index,
    percentages,
    color=bar_colors,
    width=0.62,
    yerr=[
        [
            percentages[k] - confidence_intervals[k][0]
            for k in chart_data.index
        ],
        [
            confidence_intervals[k][1] - percentages[k]
            for k in chart_data.index
        ]
    ],
    capsize=4,
    ecolor="#374151"
)


# Add percentage and sample size labels
for step in chart_data.index:

    upper_limit = confidence_intervals[step][1]

    ax.text(
        step,
        upper_limit + 1.5,
        f"{percentages[step]:.0f}%\n(n={chart_data.loc[step, 'size']})",
        ha="center",
        fontsize=9.5
    )


ax.set_xticks(chart_data.index)

ax.set_xticklabels([
    "0 steps", "1", "2", "3", "All 4"
])

ax.set_ylabel("% of applications shortlisted")
ax.set_xlabel("Application steps finished")

ax.set_ylim(0, 62)

ax.spines[["top", "right"]].set_visible(False)

ax.set_title(
    "Finishing all 4 steps: 48% shortlisted vs 9% for everyone else",
    loc="left",
    fontsize=12,
    fontweight="bold"
)

ax.text(
    0,
    -0.2,
    "Jul-Sep 2026, 1,200 applications after removing 6 duplicates. "
    "Bars show 95% bootstrap intervals.",
    transform=ax.transAxes,
    fontsize=8.5,
    color="#4b5563"
)

fig.tight_layout()

# Save the chart
fig.savefig(
    "chart_q2_shortlist_by_steps.png",
    dpi=160
)

plt.close(fig)


# --------------------------------------------------
# 7. SAVE FINAL OUTPUT FILES
# --------------------------------------------------

source_results.round(2).to_csv("out_q1_by_source.csv")

rate_table.to_csv("out_q1_by_role_source.csv")

df.drop(columns=["tier"]).to_csv(
    "applications_clean.csv",
    index=False
)

print("Analysis completed successfully!")
print("Files saved:")
print("- chart_q2_shortlist_by_steps.png")
print("- out_q1_by_source.csv")
print("- out_q1_by_role_source.csv")
print("- applications_clean.csv")
