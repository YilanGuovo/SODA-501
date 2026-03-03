# Big Data Experiments as Data Pipelines
# This script provides a complete framework for reproducible large-scale experimentation
# Authors should modify and extend the code to meet their specific needs
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No if/else conditional statements.
# - Steps are explicit so students can follow and modify each piece.

###############################################
# SECTION 1: INITIAL SETUP AND CONFIGURATION
###############################################

# Install packages manually (kept as comments for teaching).
# In a real project, install once in a virtual environment.
#
# pip install numpy pandas scipy statsmodels matplotlib
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.special as sc
import statsmodels.api as sm
import statsmodels.formula.api as smf

os.chdir(r"D:\Yilan\1\Courses\SODA\501\soda_501\05_experiments\demo")
os.getcwd()

np.random.seed(123)

###############################################
# SECTION 2: PROJECT DIRECTORY SETUP
###############################################

# Create necessary directories for our project
os.makedirs("data/raw", exist_ok=True)         # For raw, unprocessed data (e.g., event logs)
os.makedirs("data/processed", exist_ok=True)   # For cleaned, processed data (analysis-ready)
os.makedirs("outputs/figures", exist_ok=True)  # For plots and visualizations
os.makedirs("outputs/tables", exist_ok=True)   # For analysis results and summaries

###############################################
# SECTION 3: LOGGING SETUP
###############################################

# Minimal logging (file + console) without functions
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("analysis_log.txt", mode="w"),
        logging.StreamHandler()
    ]
)

###############################################
# SECTION 4: BIG DATA EXPERIMENT PIPELINE (SEQUENTIAL)
###############################################

###############################################
# STEP 0: GLOBAL SETTINGS
###############################################

# Reproducibility seed
np.random.seed(123)

# "Big data" knobs (adjust upward if you want more scale)
n_users = 100000     # number of users in the experiment
n_days  = 14         # number of post-assignment days to log

logging.info("Starting big data experiment pipeline")
logging.info(f"n_users = {n_users} | n_days = {n_days}")

###############################################
# STEP 1: GENERATE SYNTHETIC USERS (UNIT TABLE)
###############################################

logging.info("Generating synthetic user table")

user_id = np.arange(1, n_users + 1)

platform = np.random.choice(
    ["ios", "android", "web"],
    size=n_users,
    replace=True,
    p=[0.35, 0.35, 0.30]
)

cluster_id = np.random.randint(1, 501, size=n_users)

baseline_activity = np.random.gamma(shape=2.0, scale=2.0, size=n_users)

signup_cohort = np.random.choice(
    ["cohort_A", "cohort_B", "cohort_C"],
    size=n_users,
    replace=True,
    p=[0.40, 0.35, 0.25]
)


users = pd.DataFrame({
    "user_id": user_id,
    "platform": platform,
    "cluster_id": cluster_id,
    "baseline_activity": baseline_activity,
    "signup_cohort": signup_cohort
})

# Pre-treatment metric (placebo outcome) correlated with baseline_activity
users["pre_metric"] = users["baseline_activity"] + np.random.normal(0, 0.5, size=n_users)

# Save raw user table
users.to_csv("data/raw/users.csv", index=False)
logging.info("Saved: data/raw/users.csv")

###############################################
# STEP 2: BLOCKING + RANDOM ASSIGNMENT (SAVE ASSIGNMENT!)
###############################################

logging.info("Creating blocked assignment table (and saving it)")

# Blocking: deciles of baseline activity
users["block"] = pd.qcut(users["baseline_activity"], 10, labels=False) + 1  # 1..10

# Randomize within blocks (50/50)
# (groupby + transform returns aligned vector; no functions defined)
users["treat"] = (
    users.groupby("block")["user_id"]
    .transform(lambda s: (np.random.rand(len(s)) < 0.5).astype(int))
)

# Simulate noncompliance: treated units receive with probability p < 1
p_receive = 0.80
users["received"] = (users["treat"] * (np.random.rand(len(users)) < p_receive).astype(int)).astype(int)
logging.info(f"Noncompliance p_receive = {p_receive}")

assignment = users[[
    "user_id", "treat", "received", "block", "platform", "cluster_id",
    "signup_cohort", "baseline_activity", "pre_metric"
]].copy()

assignment["assignment_date"] = np.datetime64("2026-04-16")

# SAVE assignment table (essential reproducibility artifact)
assignment.to_csv("data/raw/assignment_table.csv", index=False)
logging.info("Saved: data/raw/assignment_table.csv")

###############################################
# STEP 3: GENERATE RAW EVENT LOGS (USER-DAY BIG TABLE)
###############################################

logging.info("Generating synthetic event logs (user-day table)")

# Cross join users x days using a cartesian product via merge on a dummy key
dt_assign = assignment.copy()
dt_assign["dummy"] = 1

dt_days = pd.DataFrame({"day": np.arange(1, n_days + 1)})
dt_days["dummy"] = 1

logs = dt_assign.merge(dt_days, on="dummy", how="outer").drop(columns=["dummy"])

# Date variable
logs["date"] = logs["assignment_date"] + pd.to_timedelta(logs["day"] - 1, unit="D")

# Day-of-week (Mon=1 ... Sun=7) to match the R logic
logs["dow"] = logs["date"].dt.dayofweek + 1

# Logging instrumentation dropout
logs["logged_ok"] = (np.random.rand(len(logs)) < 0.98).astype(int)

# Base click rate (Poisson intensity)
logs["base_rate"] = np.exp(
    -1.2
    + 0.15 * np.log1p(logs["baseline_activity"])
    + 0.05 * (logs["platform"] == "ios").astype(float)
    + 0.03 * (logs["platform"] == "android").astype(float)
    + 0.02 * (logs["dow"].isin([6, 7])).astype(float)
    + 0.01 * logs["day"]
)

# Treatment effect operates through received (~5% lift)
logs["click_rate"] = logs["base_rate"] * np.exp(0.05 * logs["received"])

# Click counts (Poisson)
logs["clicks"] = np.random.poisson(lam=logs["click_rate"].to_numpy())

# Purchase probability (logistic)
# logistic(x) = 1 / (1 + exp(-x))
lin = (
    -5.0
    + 0.08 * logs["clicks"]
    + 0.10 * np.log1p(logs["baseline_activity"])
    + 0.15 * logs["received"]
    + 0.02 * (logs["dow"].isin([6, 7])).astype(float)
)
logs["purchase_prob"] = sc.expit(lin.to_numpy())

# Purchase (Bernoulli)
logs["purchase"] = (np.random.rand(len(logs)) < logs["purchase_prob"].to_numpy()).astype(int)

# Active day indicator
logs["active"] = ((logs["clicks"] > 0) | (logs["purchase"] > 0)).astype(int)

# Apply logging dropout by setting outcomes to missing
# (No if/else: use .where)
logs["clicks"] = logs["clicks"].where(logs["logged_ok"] == 1, np.nan)
logs["purchase"] = logs["purchase"].where(logs["logged_ok"] == 1, np.nan)
logs["active"] = logs["active"].where(logs["logged_ok"] == 1, np.nan)

# Save raw logs
logs.to_csv("data/raw/event_logs.csv", index=False)
logging.info("Saved: data/raw/event_logs.csv")

###############################################
# STEP 4: BUILD AN ANALYSIS-READY DATASET (USER-LEVEL)
###############################################

logging.info("Building analysis-ready dataset (user-level aggregation)")

user = (
    logs.groupby([
        "user_id", "treat", "received", "block", "platform", "cluster_id",
        "signup_cohort", "baseline_activity", "pre_metric"
    ], as_index=False)
    .agg(
        post_clicks=("clicks", "sum"),
        post_purchases=("purchase", "sum"),
        days_observed=("active", lambda x: x.notna().sum()),
        missing_share=("active", lambda x: x.isna().mean()),
    )
)

# converted: any purchase > 0
user["converted"] = (user["post_purchases"] > 0).astype(int)

user.to_csv("data/processed/analysis_dataset.csv", index=False)
logging.info("Saved: data/processed/analysis_dataset.csv")

###############################################
# STEP 5: RANDOMIZATION CHECKS / BALANCE CHECKS
###############################################

logging.info("Running randomization checks / balance checks")

balance_table = (
    user.groupby("treat", as_index=False)
    .agg(
        n=("user_id", "size"),
        mean_baseline_activity=("baseline_activity", "mean"),
        mean_pre_metric=("pre_metric", "mean"),
        mean_missing_share=("missing_share", "mean"),
    )
)
balance_table.to_csv("outputs/tables/balance_means.csv", index=False)

# Standardized mean differences (SMD)
c = user[user["treat"] == 0]
t = user[user["treat"] == 1]

sd_pool_baseline = np.sqrt((c["baseline_activity"].var() + t["baseline_activity"].var()) / 2)
smd_baseline_activity = (t["baseline_activity"].mean() - c["baseline_activity"].mean()) / sd_pool_baseline

sd_pool_pre = np.sqrt((c["pre_metric"].var() + t["pre_metric"].var()) / 2)
smd_pre_metric = (t["pre_metric"].mean() - c["pre_metric"].mean()) / sd_pool_pre

smd_table = pd.DataFrame({
    "variable": ["baseline_activity", "pre_metric"],
    "smd": [smd_baseline_activity, smd_pre_metric]
})
smd_table.to_csv("outputs/tables/balance_smd.csv", index=False)

###############################################
# STEP 6: ESTIMATE EXPERIMENTAL EFFECTS (ATE)
###############################################

logging.info("Estimating treatment effects (ATE)")

ate_converted = user.loc[user["treat"] == 1, "converted"].mean() - user.loc[user["treat"] == 0, "converted"].mean()
ate_purchases = user.loc[user["treat"] == 1, "post_purchases"].mean() - user.loc[user["treat"] == 0, "post_purchases"].mean()
ate_clicks    = user.loc[user["treat"] == 1, "post_clicks"].mean() - user.loc[user["treat"] == 0, "post_clicks"].mean()

ate_simple = pd.DataFrame({
    "outcome": ["converted", "post_purchases", "post_clicks"],
    "ate_diff_in_means": [ate_converted, ate_purchases, ate_clicks]
})
ate_simple.to_csv("outputs/tables/ate_diff_in_means.csv", index=False)

# Regression adjustment with cluster-robust SE
# Note: statsmodels uses C(...) for categorical variables (do NOT define variable named C)
fit_conv = smf.ols(
    "converted ~ treat + baseline_activity + pre_metric + C(block)",
    data=user
).fit(cov_type="cluster", cov_kwds={"groups": user["cluster_id"]})

fit_pur = smf.ols(
    "post_purchases ~ treat + baseline_activity + pre_metric + C(block)",
    data=user
).fit(cov_type="cluster", cov_kwds={"groups": user["cluster_id"]})

fit_conv.summary2().tables[1].to_csv("outputs/tables/regression_converted.csv")
fit_pur.summary2().tables[1].to_csv("outputs/tables/regression_purchases.csv")

###############################################
# STEP 6B: ADD A RETENTION-STYLE OUTCOME AND ESTIMATE ITS ATE
###############################################

logging.info("Adding retention-style outcomes and estimating ATE")

# From the user-day logs, create days_active = number of days with active == 1 for each user (ignore missing days)
retention = (
    logs.groupby(["user_id"], as_index=False)
    .agg(
        days_active=("active", lambda x: (x == 1).sum())
    )
)

# Create retained_any = 1 if days_active >= 1 else 0
retention["retained_any"] = (retention["days_active"] >= 1).astype(int)

# Add both outcomes to the analysis-ready dataset
user = user.merge(retention, on="user_id", how="left")

# (a) difference in means
ate_days_active = user.loc[user["treat"] == 1, "days_active"].mean() - user.loc[user["treat"] == 0, "days_active"].mean()
ate_retained_any = user.loc[user["treat"] == 1, "retained_any"].mean() - user.loc[user["treat"] == 0, "retained_any"].mean()

# (b) regression adjustment with factor(block) and cluster-robust SEs clustered at cluster_id
fit_days_active = smf.ols(
    "days_active ~ treat + C(block)",
    data=user
).fit(cov_type="cluster", cov_kwds={"groups": user["cluster_id"]})

fit_retained_any = smf.ols(
    "retained_any ~ treat + C(block)",
    data=user
).fit(cov_type="cluster", cov_kwds={"groups": user["cluster_id"]})

ate_retention = pd.DataFrame({
    "outcome": ["days_active", "retained_any"],
    "ate_diff_in_means": [ate_days_active, ate_retained_any],
    "ate_regression": [fit_days_active.params.get("treat", np.nan), fit_retained_any.params.get("treat", np.nan)],
    "se_cluster": [fit_days_active.bse.get("treat", np.nan), fit_retained_any.bse.get("treat", np.nan)]
})

ate_retention.to_csv("outputs/tables/ate_retention.csv", index=False)
logging.info("Saved: outputs/tables/ate_retention.csv")

###############################################
# STEP 6C: SIMULATE NONCOMPLIANCE AND COMPARE ITT VS TOT (IV)
###############################################

logging.info("Estimating ITT vs TOT (IV) under noncompliance")

# ITT: regress outcome on treat (with C(block) and cluster-robust SEs)
fit_itt = smf.ols(
    "converted ~ treat + C(block)",
    data=user
).fit(cov_type="cluster", cov_kwds={"groups": user["cluster_id"]})

# TOT / LATE using IV: treat as an instrument for received
from statsmodels.sandbox.regression.gmm import IV2SLS

# Build design matrices explicitly
y = user["converted"].astype(float).to_numpy()

exog = pd.DataFrame({
    "const": 1.0,
    "received": user["received"].astype(float).to_numpy()
})

instr = pd.DataFrame({
    "const": 1.0,
    "treat": user["treat"].astype(float).to_numpy()
})

block_dummies = pd.get_dummies(user["block"], prefix="block", drop_first=True).astype(float)

exog = pd.concat([exog, block_dummies], axis=1).astype(float)
instr = pd.concat([instr, block_dummies], axis=1).astype(float)

iv_fit = IV2SLS(y, exog.to_numpy(dtype=float), instr.to_numpy(dtype=float)).fit()

iv_fit_cr = iv_fit.get_robustcov_results(cov_type="cluster", groups=user["cluster_id"].to_numpy())

itt_vs_tot_explain = (
    "With one-sided noncompliance, assignment (treat) moves outcomes only through the subset that actually receives. "
    "ITT averages the effect over everyone assigned, including those who do not receive, so it is attenuated toward zero. "
    "TOT/LATE rescales by the take-up (first stage), recovering the effect among compliers, so it is typically larger in magnitude."
)

itt_vs_tot = pd.DataFrame({
    "estimand": ["ITT", "TOT_IV"],
    "p_receive": [p_receive, p_receive],
    "estimate": [fit_itt.params.get("treat", np.nan), iv_fit_cr.params[1]],
    "se_cluster": [fit_itt.bse.get("treat", np.nan), iv_fit_cr.bse[1]],
    "explanation": [itt_vs_tot_explain, itt_vs_tot_explain]
})

itt_vs_tot.to_csv("outputs/tables/itt_vs_tot.csv", index=False)
logging.info("Saved: outputs/tables/itt_vs_tot.csv")

###############################################
# STEP 7: VISUALIZATIONS (BIG DATA EXPERIMENT DIAGNOSTICS)
###############################################

logging.info("Creating figures")

# Figure 1: baseline activity distribution by treatment
sample_user = user.sample(n=5000, random_state=123)

plt.figure()
plt.hist(sample_user.loc[sample_user["treat"] == 0, "baseline_activity"], bins=60, alpha=0.7, label="Control")
plt.hist(sample_user.loc[sample_user["treat"] == 1, "baseline_activity"], bins=60, alpha=0.7, label="Treatment")
plt.title("Baseline Activity Distribution by Treatment Arm (sample)")
plt.xlabel("Baseline activity")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/baseline_activity_by_treat.png", dpi=150)
plt.close()

# Day-level table from logs (ignore missing purchase days)
day = (
    logs.groupby(["date", "day", "treat"], as_index=False)
    .agg(
        conversion_rate=("purchase", "mean"),
        mean_clicks=("clicks", "mean"),
        missing_share=("purchase", lambda x: x.isna().mean())
    )
)

# Figure 2: daily conversion rate by treatment arm
plt.figure()
plt.plot(day.loc[day["treat"] == 0, "date"], day.loc[day["treat"] == 0, "conversion_rate"], label="Control")
plt.plot(day.loc[day["treat"] == 1, "date"], day.loc[day["treat"] == 1, "conversion_rate"], label="Treatment")
plt.title("Daily Conversion Rate by Treatment Arm")
plt.xlabel("Date")
plt.ylabel("Conversion rate")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/daily_conversion_rate.png", dpi=150)
plt.close()

# Figure 3: daily missingness by treatment arm
plt.figure()
plt.plot(day.loc[day["treat"] == 0, "date"], day.loc[day["treat"] == 0, "missing_share"], label="Control")
plt.plot(day.loc[day["treat"] == 1, "date"], day.loc[day["treat"] == 1, "missing_share"], label="Treatment")
plt.title("Daily Missingness in Logged Purchases (Instrumentation Check)")
plt.xlabel("Date")
plt.ylabel("Share missing")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/daily_missingness.png", dpi=150)
plt.close()

###############################################
# STEP 8: PLACEBO TESTS (A/A + PLACEBO OUTCOME)
###############################################

logging.info("Running placebo tests (A/A and placebo outcome)")

# Placebo outcome: pre_metric should not differ by treatment
placebo_pre = smf.ols("pre_metric ~ treat + C(block)", data=user).fit()
placebo_pre.summary2().tables[1].to_csv("outputs/tables/placebo_pre_metric.csv")

# A/A test: split CONTROL into two pseudo-arms many times and compute diff-in-means
np.random.seed(123)

control_ids = user.loc[user["treat"] == 0, "user_id"].to_numpy()

aa_rows = []

for b in range(1, 201):
    a_group = np.random.choice(control_ids, size=int(np.floor(len(control_ids) / 2)), replace=False)

    dt_sub = user.loc[user["treat"] == 0, ["user_id", "converted"]].copy()
    dt_sub["aa"] = dt_sub["user_id"].isin(a_group).astype(int)

    aa_effect = dt_sub.loc[dt_sub["aa"] == 1, "converted"].mean() - dt_sub.loc[dt_sub["aa"] == 0, "converted"].mean()

    aa_rows.append({"iter": b, "aa_effect": aa_effect})

aa_results = pd.DataFrame(aa_rows)
aa_results.to_csv("outputs/tables/aa_effects.csv", index=False)

plt.figure()
plt.hist(aa_results["aa_effect"], bins=40)
plt.title("A/A Placebo Distribution (Control Split Into Two Arms)")
plt.xlabel("Difference in means (converted)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/figures/aa_placebo_hist.png", dpi=150)
plt.close()

###############################################
# STEP 9: SAVE ENVIRONMENT INFO
###############################################

logging.info("Saving environment information")

# Save a lightweight environment snapshot
import sys
import platform

env_lines = [
    f"Python: {sys.version}",
    f"Executable: {sys.executable}",
    f"Platform: {platform.platform()}",
]
with open("outputs/python_environment.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(env_lines) + "\n")

logging.info("Pipeline complete")
# %%
