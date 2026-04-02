###############################################################################
# API Use + Forecasting Tutorial: Python
# Author: Jared Edgerton
# Date: date.today()
#
# This script demonstrates:
#   1) Loading and cleaning presidential vote data (1976–2020)
#   2) Pulling economic indicators from FRED (Q1/Q2 of election years)
#   3) Building a simple national vote-share model (OLS)
#   4) Loading state-level poll + census data and fitting a state model (OLS)
#   5) Producing a simple 2020 state-level visualization
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No conditional statements (no if/else).
# - You will see the same steps repeated so students can follow the logic and
#   edit one piece at a time.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# If you do not have these installed, run (in Terminal / Anaconda Prompt):
#   pip install pandas numpy matplotlib statsmodels fredapi pyreadr plotly lxml requests

# %%
import re
import os    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date
import statsmodels.formula.api as smf

# FRED API wrapper
from fredapi import Fred

# For reading .rds (RDS) files in Python (state-level poll/census data)
import pyreadr

# For a quick US states choropleth
import plotly.express as px

#packages for improved model
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error


os.chdir(r"D:\Yilan\1\Courses\SODA\501\soda_501\08_api\demo")
print(os.getcwd())

# -----------------------------------------------------------------------------
# Part 1: Presidential vote data (national-level)
# -----------------------------------------------------------------------------
# Read in the presidential election vote data
vote_data = pd.read_csv("1976-2020-president.csv")

# Keep only Democrat and Republican votes
vote_data = vote_data[
    vote_data["party_detailed"].isin(["DEMOCRAT", "REPUBLICAN"])
].copy()

# Summarize votes by year, candidate, party (mimics ddply summarize in R)
vote_data = (
    vote_data
    .groupby(["year", "candidate", "party_detailed"], as_index=False)
    .agg(
        candidatevotes=("candidatevotes", "sum"),
        totalvotes=("totalvotes", "sum")
    )
)

# Drop OTHER and blank candidate entries (mimics R filters)
vote_data = vote_data[
    (~vote_data["candidate"].isin(["OTHER", ""])) &
    (vote_data["candidate"].notna())
].copy()

# Compute vote percent
vote_data["vote_pct"] = vote_data["candidatevotes"] / vote_data["totalvotes"]

# Election years used in this dataset
election_years = np.sort(vote_data["year"].unique())


# -----------------------------------------------------------------------------
# Part 2: Pulling economic indicators from FRED (Q1/Q2 of election years)
# -----------------------------------------------------------------------------
# NOTE: Replace with your own key (students should get one from FRED).
fred_api_key = os.environ.get("FRED_API_KEY")

if not fred_api_key:
    raise ValueError("Missing FRED_API_KEY environment variable.")

fred = Fred(api_key=fred_api_key)

# Define observation window based on the election years in the vote data
obs_start = f"{int(election_years.min())}-01-01"
obs_end   = f"{int(election_years.max())}-06-30"

# --- Unemployment (UNRATE) ---
# FRED returns a time series with dates; we convert to quarterly and keep Q1/Q2
unrate = fred.get_series("UNRATE", observation_start=obs_start, observation_end=obs_end)
unrate = unrate.to_frame(name="unemployment_rate")
unrate.index = pd.to_datetime(unrate.index)
unrate = unrate.resample("QE").mean().reset_index().rename(columns={"index": "date"})
unrate["year"] = unrate["date"].dt.year
unrate["quarter"] = unrate["date"].dt.quarter
unemployment_data = unrate[
    (unrate["year"].isin(election_years)) &
    (unrate["quarter"] <= 2)
][["year", "quarter", "unemployment_rate"]].copy()

# --- GDP (GDP) ---
gdp = fred.get_series("GDP", observation_start=obs_start, observation_end=obs_end)
gdp = gdp.to_frame(name="gdp")
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.resample("QE").mean().reset_index().rename(columns={"index": "date"})
gdp["year"] = gdp["date"].dt.year
gdp["quarter"] = gdp["date"].dt.quarter
gdp_data = gdp[
    (gdp["year"].isin(election_years)) &
    (gdp["quarter"] <= 2)
][["year", "quarter", "gdp"]].copy()

# --- CPI (CPIAUCSL) ---
cpi = fred.get_series("CPIAUCSL", observation_start=obs_start, observation_end=obs_end)
cpi = cpi.to_frame(name="cpi")
cpi.index = pd.to_datetime(cpi.index)
cpi = cpi.resample("QE").mean().reset_index().rename(columns={"index": "date"})
cpi["year"] = cpi["date"].dt.year
cpi["quarter"] = cpi["date"].dt.quarter
cpi_data = cpi[
    (cpi["year"].isin(election_years)) &
    (cpi["quarter"] <= 2)
][["year", "quarter", "cpi"]].copy()

# (Optional, for teaching) inflation rate example (year-over-year using Q1 vs Q3 lag etc.)
# The original R code computed inflation_rate and then dropped it before widening.
# We replicate the same idea but do not use it in the final wide dataset.
inflation_data = cpi_data.sort_values(["year", "quarter"]).copy()
inflation_data["inflation_rate"] = (
    (inflation_data["cpi"] / inflation_data["cpi"].shift(2) - 1) * 100
)

# --- 10Y Treasury Rate (DGS10) ---
dgs10 = fred.get_series("DGS10", observation_start=obs_start, observation_end=obs_end)
dgs10 = dgs10.to_frame(name="dgs10")
dgs10.index = pd.to_datetime(dgs10.index)
dgs10 = dgs10.resample("QE").mean().reset_index().rename(columns={"index": "date"})
dgs10["year"] = dgs10["date"].dt.year
dgs10["quarter"] = dgs10["date"].dt.quarter
dgs10_data = dgs10[
    (dgs10["year"].isin(election_years)) &
    (dgs10["quarter"] <= 2)
][["year", "quarter", "dgs10"]].copy()

# Combine all economic data into one long table keyed by (year, quarter)
combined_long = (
    unemployment_data
    .merge(gdp_data, on=["year", "quarter"], how="outer")
    .merge(inflation_data[["year", "quarter", "cpi"]], on=["year", "quarter"], how="outer")
    .merge(dgs10_data, on=["year", "quarter"], how="outer") 
    .sort_values(["year", "quarter"])
)

# Pivot wider like R pivot_wider(names_from=quarter, values_from=c(...), names_sep="_Q")
combined_wide = combined_long.pivot_table(
    index="year",
    columns="quarter",
    values=["unemployment_rate", "gdp", "cpi", "dgs10"],
    aggfunc="first"
)

# Flatten column names to match the R naming style, e.g. unemployment_rate_Q1
combined_wide.columns = [f"{var}_Q{q}" for var, q in combined_wide.columns]
combined_wide = combined_wide.reset_index()


# -----------------------------------------------------------------------------
# Part 3: Merge vote data + economic data and build national forecast features
# -----------------------------------------------------------------------------
forecast_data = vote_data.merge(combined_wide, on="year", how="left").copy()

# Incumbent indicator (hard-coded, sequential assignments like the R mutate/ifelse chain)
forecast_data["incumbent"] = 0
forecast_data.loc[(forecast_data["candidate"] == "FORD, GERALD") & (forecast_data["year"] == 1976), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "CARTER, JIMMY") & (forecast_data["year"] == 1980), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "REAGAN, RONALD") & (forecast_data["year"] == 1984), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "BUSH, GEORGE H.W.") & (forecast_data["year"] == 1992), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "CLINTON, BILL") & (forecast_data["year"] == 1996), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "BUSH, GEORGE W.") & (forecast_data["year"] == 2004), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "OBAMA, BARACK H.") & (forecast_data["year"] == 2012), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "TRUMP, DONALD J.") & (forecast_data["year"] == 2020), "incumbent"] = 1

# Quarter-to-quarter changes (Q2 - Q1), matching the R code
forecast_data["gdp_change"] = forecast_data["gdp_Q2"] - forecast_data["gdp_Q1"]
forecast_data["cpi_change"] = forecast_data["cpi_Q2"] - forecast_data["cpi_Q1"]
forecast_data["unemploy_change"] = forecast_data["unemployment_rate_Q2"] - forecast_data["unemployment_rate_Q1"]
forecast_data["dgs10_change"] = forecast_data["dgs10_Q2"] - forecast_data["dgs10_Q1"]

# Split training (pre-2020) vs testing (2020)
forecast_data_training = forecast_data[forecast_data["year"] < 2020].copy()
forecast_data_testing  = forecast_data[forecast_data["year"] == 2020].copy()

# Fit the national OLS model
# R: vote_pct ~ incumbent * unemploy_change + party_detailed + poly(year, 2, raw = T)
# Python: use year + year^2 explicitly
train_ols = smf.ols(
    "vote_pct ~ incumbent * unemploy_change + C(party_detailed) + year + I(year**2)",
    data=forecast_data_training
).fit()

# ----------------------------
# Improved national model: Elastic Net
# ----------------------------
train_df = forecast_data_training.copy()
test_df  = forecast_data_testing.copy()

train_df["year2"] = train_df["year"]**2
test_df["year2"]  = test_df["year"]**2

train_df["inc_unemp"] = train_df["incumbent"] * train_df["unemploy_change"]
test_df["inc_unemp"]  = test_df["incumbent"] * test_df["unemploy_change"]

num_features = ["year", "year2", "incumbent", "unemploy_change", "inc_unemp", "dgs10_change"]
cat_features = ["party_detailed"]

X_train = train_df[num_features + cat_features]
y_train = train_df["vote_pct"]
X_test  = test_df[num_features + cat_features]
y_test  = test_df["vote_pct"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), cat_features),
        ("num", "passthrough", num_features)
    ]
)

model = ElasticNet(max_iter=20000)

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("model", model)
])

param_grid = {
    "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    "model__l1_ratio": [0.0, 0.5, 1.0]  # 0=Ridge, 1=Lasso
}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
test_pred_en = best_model.predict(X_test)

print("Best ElasticNet params:", grid.best_params_)
print("2020 ElasticNet predictions:", test_pred_en[:5])

# Generate predictions for training data
forecast_data_training["pred_vote"] = train_ols.predict(forecast_data_training)
print(forecast_data_training[["vote_pct", "pred_vote"]].head(20))

# Generate predictions for test data (2020)
test_pred = train_ols.predict(forecast_data_testing)
print("\n2020 test predictions (first few):")
print(test_pred.head())

# ----------------------------
# Out-of-sample evaluation (2020)
# ----------------------------
baseline_pred = test_pred.values if hasattr(test_pred, "values") else np.array(test_pred)
improved_pred = np.array(test_pred_en)

mae_baseline = mean_absolute_error(y_test, baseline_pred)
rmse_baseline = np.sqrt(mean_squared_error(y_test, baseline_pred))

mae_improved = mean_absolute_error(y_test, improved_pred)
rmse_improved = np.sqrt(mean_squared_error(y_test, improved_pred))

print("\nOut-of-sample (2020) comparison:")
print(f"Baseline OLS   MAE={mae_baseline:.4f}  RMSE={rmse_baseline:.4f}")
print(f"Improved ENet  MAE={mae_improved:.4f}  RMSE={rmse_improved:.4f}")


import matplotlib.pyplot as plt

plt.figure()
plt.scatter(y_test, baseline_pred)
plt.xlabel("Actual vote_pct")
plt.ylabel("Predicted vote_pct")
plt.title("Baseline OLS: Predicted vs Actual (2020)")
plt.savefig("figure/baseline_pred_vs_actual.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
plt.scatter(y_test, improved_pred)
plt.xlabel("Actual vote_pct")
plt.ylabel("Predicted vote_pct")
plt.title("Improved ElasticNet: Predicted vs Actual (2020)")
plt.savefig("figure/elasticnet_pred_vs_actual.png", dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------------------------------------------------------
# Part 4: State-level model (poll + census + economy)
# -----------------------------------------------------------------------------
# Load pre-existing poll and census data (RDS) and convert to pandas DataFrame
# NOTE: Update the path to wherever the RDS file lives on your system.
poll_census_path = "poll_census_data.rds"
poll_census_obj = pyreadr.read_r(poll_census_path)
poll_census_data = list(poll_census_obj.values())[0]

# Prepare economic data for merging with state-level data (distinct year-level fields)
forecast_econ = forecast_data[
    ["year",
     "unemployment_rate_Q1", "unemployment_rate_Q2",
     "gdp_Q1", "gdp_Q2",
     "cpi_Q1", "cpi_Q2",
     "dgs10_Q1", "dgs10_Q2",     
     "gdp_change", "cpi_change", "unemploy_change",
     "dgs10_change"]         
].drop_duplicates()

# Merge state-level poll/census data with economic data
state_data = poll_census_data.merge(forecast_econ, on="year", how="left")

# Fit the state-level OLS model (training: year < 2020)
# R: vote_pct ~ poll_avg + year + party_simplified + white + black + asian + hispanic
pred_results = smf.ols(
    "vote_pct ~ poll_avg + year + C(party_simplified) + white + black + asian + hispanic",
    data=state_data[state_data["year"] < 2020]
).fit()

# Out-of-sample predictions for 2020 and beyond
out_of_sample = pred_results.predict(state_data[state_data["year"] >= 2020])

# Prepare election outcomes table (actual + predicted)
elect_outcomes = state_data[state_data["year"] >= 2020][
    ["year", "state_po", "party_simplified", "candidate", "vote_pct"]
].copy()

elect_outcomes["vote_pred"] = out_of_sample.values


# -----------------------------------------------------------------------------
# Part 5: 2020 vote difference (Biden minus Trump) and a map
# -----------------------------------------------------------------------------
# Create a 2020-only dataset
elect_2020 = elect_outcomes[elect_outcomes["year"] == 2020].copy()

# Standardize candidate names into a simple label for pivoting
elect_2020["candidate_simple"] = elect_2020["candidate"].astype(str).str.lower()
elect_2020.loc[elect_2020["candidate_simple"].str.contains("biden"), "candidate_simple"] = "biden"
elect_2020.loc[elect_2020["candidate_simple"].str.contains("trump"), "candidate_simple"] = "trump"

# Pivot wide like R pivot_wider(... names_glue = "{candidate}_{.value}")
wide_2020 = elect_2020.pivot_table(
    index=["state_po", "year"],
    columns="candidate_simple",
    values=["vote_pct", "vote_pred"],
    aggfunc="first"
)

# Flatten column names to match the R naming style (candidate_value)
wide_2020.columns = [f"{cand}_{val}" for val, cand in wide_2020.columns]
wide_2020 = wide_2020.reset_index()

# Vote difference (Biden minus Trump), matching the R intent
vote_diff_2020 = wide_2020.copy()
vote_diff_2020["vote_diff"] = vote_diff_2020["biden_vote_pct"] - vote_diff_2020["trump_vote_pct"]
vote_diff_2020 = vote_diff_2020[["state_po", "vote_diff"]].drop_duplicates()

# (Optional) Remove AK and HI to mimic the R map example
vote_diff_2020 = vote_diff_2020[~vote_diff_2020["state_po"].isin(["AK", "HI"])].copy()

# Plot a simple choropleth map of the vote difference
fig = px.choropleth(
    vote_diff_2020,
    locations="state_po",
    locationmode="USA-states",
    color="vote_diff",
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    scope="usa",
    title="2020 Vote Share Difference (Biden − Trump)"
)
fig.show()

print("UNRATE sample:", unrate.head())




comparison_table = pd.DataFrame({
    "model": ["Baseline OLS", "Improved ElasticNet"],
    "MAE_2020": [mae_baseline, mae_improved],
    "RMSE_2020": [rmse_baseline, rmse_improved],
})
print("\nModel comparison table:")
print(comparison_table)


# -----------------------------------------------------------------------------
# Part 6: save files
# -----------------------------------------------------------------------------
forecast_data_training[["year","candidate","party_detailed","vote_pct","pred_vote"]].to_csv(
    "table/baseline_predictions.csv", index=False
)

with open("model_summary.txt", "w") as f:
    f.write(train_ols.summary().as_text())

fig.write_html("figure/vote_map_2020.html")

comparison_table.to_csv("table/model_comparison_2020.csv", index=False)


# %%
