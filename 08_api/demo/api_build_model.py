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

import re
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


# -----------------------------------------------------------------------------
# Part 1: Presidential vote data (national-level)
# -----------------------------------------------------------------------------
# Read in the presidential election vote data
vote_data = pd.read_csv("votes/1976-2020-president.csv")

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
fred_api_key = "YOUR_FRED_API_KEY_HERE"
fred = Fred(api_key=fred_api_key)

# Define observation window based on the election years in the vote data
obs_start = f"{int(election_years.min())}-01-01"
obs_end   = f"{int(election_years.max())}-06-30"

# --- Unemployment (UNRATE) ---
# FRED returns a time series with dates; we convert to quarterly and keep Q1/Q2
unrate = fred.get_series("UNRATE", observation_start=obs_start, observation_end=obs_end)
unrate = unrate.to_frame(name="unemployment_rate")
unrate.index = pd.to_datetime(unrate.index)
unrate = unrate.resample("Q").mean().reset_index().rename(columns={"index": "date"})
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
gdp = gdp.resample("Q").mean().reset_index().rename(columns={"index": "date"})
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
cpi = cpi.resample("Q").mean().reset_index().rename(columns={"index": "date"})
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

# Combine all economic data into one long table keyed by (year, quarter)
combined_long = (
    unemployment_data
    .merge(gdp_data, on=["year", "quarter"], how="outer")
    .merge(inflation_data[["year", "quarter", "cpi"]], on=["year", "quarter"], how="outer")
    .sort_values(["year", "quarter"])
)

# Pivot wider like R pivot_wider(names_from=quarter, values_from=c(...), names_sep="_Q")
combined_wide = combined_long.pivot_table(
    index="year",
    columns="quarter",
    values=["unemployment_rate", "gdp", "cpi"],
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

# Generate predictions for training data
forecast_data_training["pred_vote"] = train_ols.predict(forecast_data_training)
print(forecast_data_training[["vote_pct", "pred_vote"]].head(20))

# Generate predictions for test data (2020)
test_pred = train_ols.predict(forecast_data_testing)
print("\n2020 test predictions (first few):")
print(test_pred.head())


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
     "gdp_change", "cpi_change", "unemploy_change"]
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
    color_continuous_midpoint=0,
    scope="usa",
    title="2020 Vote Share Difference (Biden − Trump)"
)
fig.show()
