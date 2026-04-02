###############################################################################
# Election Forecasting + FRED Tutorial: R (API + Modeling + Mapping)
# Author: Jared Edgerton
# Date: Sys.Date()
#
# This script demonstrates:
#   1) Reading and cleaning U.S. presidential election vote data (1976–2020)
#   2) Pulling macroeconomic indicators from FRED (unemployment, GDP, CPI)
#   3) Building a simple election forecasting model with lm()
#   4) Extending the model to state-level predictions and mapping results
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No conditional statements (no if/else).
# - You will see steps written out explicitly so students can follow the workflow.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Install (if needed) and load the necessary libraries.
#
# If you do NOT have these packages installed, run:
# install.packages(c("plyr", "dplyr", "usdata", "fredr", "lubridate", "tidyr",
#                    "maps", "ggplot2"))

library(plyr)      # Data manipulation (older package, still used in this script)
library(dplyr)     # Data manipulation and transformation
library(usdata)    # US-specific data and utilities (not required for every step)
library(fredr)     # Accessing the FRED (Federal Reserve Economic Data) API
library(lubridate) # Working with dates (year(), quarter())
library(tidyr)     # Data tidying (pivot_wider)
library(maps)      # Creating maps
library(ggplot2)   # Plotting (including map_data + ggplot)

# -----------------------------------------------------------------------------
# Part 1: Read and Clean Presidential Election Vote Data (National)
# -----------------------------------------------------------------------------
# Goal:
# - Read the presidential vote data (1976–2020)
# - Keep only Democrat and Republican candidates
# - Aggregate votes by year/candidate/party
# - Compute each candidate's national vote share

# Read in the presidential election vote data
# NOTE: Make sure this file path exists relative to your working directory.
vote_data <- read.csv("1976-2020-president.csv")

# Keep only Democrat and Republican votes
vote_data <- vote_data %>%
  filter(party_detailed %in% c("DEMOCRAT", "REPUBLICAN"))

# Aggregate vote totals by year, candidate, and party
# - candidatevotes: total votes for the candidate
# - totalvotes: total votes cast (used to compute vote share)
vote_data <- vote_data %>%
  plyr::ddply(
    .(year, candidate, party_detailed),
    summarize,
    candidatevotes = sum(candidatevotes, na.rm = TRUE),
    totalvotes = sum(totalvotes, na.rm = TRUE)
  )

# Drop empty or "OTHER" candidate rows (data cleaning)
vote_data <- vote_data %>%
  filter(candidate != "OTHER", candidate != "")

# Compute national vote share for each candidate (vote_pct)
vote_data <- vote_data %>%
  mutate(vote_pct = candidatevotes / totalvotes)

# Look at the cleaned vote data
# print(head(vote_data, 10))

# -----------------------------------------------------------------------------
# Part 2: Pull Economic Data from FRED (Unemployment, GDP, CPI)
# -----------------------------------------------------------------------------
# Goal:
# - Pull quarterly data from FRED
# - Keep only Q1 and Q2 of each election year
# - Reshape the data into wide format: unemployment_rate_Q1, unemployment_rate_Q2, etc.

# Set up your FRED API key
# IMPORTANT:
# - You should NOT hard-code real API keys into scripts you share publicly.
# - Replace the string below with your own key.
fredr_set_key("YOUR_FRED_API_KEY_HERE")

# Identify election years in the vote dataset
election_years <- sort(unique(vote_data$year))

# Define the date range we will request from FRED
# (We request up through June 30 so we can include Q2 of the last election year.)
fred_start <- as.Date(paste0(min(election_years), "-01-01"))
fred_end   <- as.Date(paste0(max(election_years), "-06-30"))

# -----------------------------------------------------------------------------
# Step 1: Unemployment Rate (UNRATE) — first two quarters of election years
# -----------------------------------------------------------------------------
unemployment_raw <- fredr(
  series_id = "UNRATE",
  observation_start = fred_start,
  observation_end = fred_end,
  frequency = "q"
)

unemployment_data <- unemployment_raw %>%
  filter(year(date) %in% election_years, quarter(date) <= 2) %>%
  select(date, value) %>%
  rename(unemployment_rate = value)

# -----------------------------------------------------------------------------
# Step 2: GDP (GDP) — first two quarters of election years
# -----------------------------------------------------------------------------
gdp_raw <- fredr(
  series_id = "GDP",
  observation_start = fred_start,
  observation_end = fred_end,
  frequency = "q"
)

gdp_data <- gdp_raw %>%
  filter(year(date) %in% election_years, quarter(date) <= 2) %>%
  select(date, value) %>%
  rename(gdp = value)

# -----------------------------------------------------------------------------
# Step 3: CPI (CPIAUCSL) — first two quarters of election years
# -----------------------------------------------------------------------------
cpi_raw <- fredr(
  series_id = "CPIAUCSL",
  observation_start = fred_start,
  observation_end = fred_end,
  frequency = "q"
)

cpi_data <- cpi_raw %>%
  filter(year(date) %in% election_years, quarter(date) <= 2) %>%
  select(date, value) %>%
  rename(cpi = value)

# -----------------------------------------------------------------------------
# Step 4: (Optional) Inflation Rate Example
# -----------------------------------------------------------------------------
# This shows how you could compute year-over-year inflation for each quarter.
# NOTE: We do NOT use inflation_rate later in this script (we drop it), but this
# is a helpful example of lag() + grouped calculations.

inflation_data <- cpi_data %>%
  group_by(year = year(date)) %>%
  arrange(date) %>%
  mutate(inflation_rate = (cpi / lag(cpi, 2) - 1) * 100) %>%
  ungroup()

# -----------------------------------------------------------------------------
# Step 5: Combine economic series and reshape into wide format by year
# -----------------------------------------------------------------------------
combined_data <- unemployment_data %>%
  full_join(gdp_data, by = "date") %>%
  full_join(inflation_data, by = "date") %>%
  mutate(
    year = year(date),
    quarter = quarter(date)
  ) %>%
  select(year, quarter, unemployment_rate, gdp, cpi, inflation_rate) %>%
  arrange(year, quarter) %>%
  dplyr::select(-inflation_rate) %>%
  pivot_wider(
    id_cols = year,
    names_from = quarter,
    values_from = c(unemployment_rate, gdp, cpi),
    names_sep = "_Q"
  )

# Look at the combined economic data (wide format)
# print(head(combined_data, 10))

# -----------------------------------------------------------------------------
# Part 3: Merge Vote Data + Economic Data and Build a National Forecast Model
# -----------------------------------------------------------------------------
# Goal:
# - Merge vote share with economic indicators
# - Create simple features (incumbent, changes from Q1 to Q2)
# - Fit a linear regression model predicting vote share

forecast_data <- vote_data %>%
  left_join(combined_data, by = "year")

# -----------------------------------------------------------------------------
# Step 1: Create an incumbent indicator (hard-coded)
# -----------------------------------------------------------------------------
# We create a 0/1 variable:
# - 1 means the candidate is the incumbent in that election year
# - 0 otherwise
#
# NOTE:
# - We avoid if/else by using logical expressions converted to 0/1.

forecast_data <- forecast_data %>%
  mutate(
    incumbent = as.integer(
      (candidate == "FORD, GERALD"       & year == 1976) |
      (candidate == "CARTER, JIMMY"      & year == 1980) |
      (candidate == "REAGAN, RONALD"     & year == 1984) |
      (candidate == "BUSH, GEORGE H.W."  & year == 1992) |
      (candidate == "CLINTON, BILL"      & year == 1996) |
      (candidate == "BUSH, GEORGE W."    & year == 2004) |
      (candidate == "OBAMA, BARACK H."   & year == 2012) |
      (candidate == "TRUMP, DONALD J."   & year == 2020)
    )
  )

# -----------------------------------------------------------------------------
# Step 2: Create simple economic change variables (Q2 - Q1)
# -----------------------------------------------------------------------------
forecast_data <- forecast_data %>%
  mutate(
    gdp_change = gdp_Q2 - gdp_Q1,
    cpi_change = cpi_Q2 - cpi_Q1,
    unemploy_change = unemployment_rate_Q2 - unemployment_rate_Q1
  )

# -----------------------------------------------------------------------------
# Step 3: Split into training (pre-2020) and testing (2020) sets
# -----------------------------------------------------------------------------
forecast_data_training <- forecast_data %>%
  filter(year < 2020)

forecast_data_testing <- forecast_data %>%
  filter(year == 2020)

# -----------------------------------------------------------------------------
# Step 4: Fit a national forecasting model (OLS)
# -----------------------------------------------------------------------------
# This model predicts vote share using:
# - incumbent status interacted with unemployment change
# - party (Democrat vs Republican)
# - a quadratic time trend

train_ols <- lm(
  vote_pct ~ incumbent * unemploy_change + party_detailed + poly(year, 2, raw = TRUE),
  data = forecast_data_training
)

# -----------------------------------------------------------------------------
# Step 5: Generate predictions for training and testing data
# -----------------------------------------------------------------------------
forecast_data_training$pred_vote <- predict(train_ols, newdata = forecast_data_training)

# Print actual vs predicted (training)
print(forecast_data_training[, colnames(forecast_data_training) %in% c("year", "candidate", "party_detailed", "vote_pct", "pred_vote")])

# Predict for the 2020 test set
test_pred <- predict(train_ols, newdata = forecast_data_testing)
print(test_pred)

# -----------------------------------------------------------------------------
# Part 4: State-Level Model + Mapping (2020 Example)
# -----------------------------------------------------------------------------
# Goal:
# - Merge state-level polling/census data with our economic features
# - Fit a simple state-level regression model (trained on years < 2020)
# - Predict out-of-sample for 2020+
# - Create a map of the 2020 vote-share difference by state

# -----------------------------------------------------------------------------
# Step 1: Load state-level poll + census data
# -----------------------------------------------------------------------------
# NOTE:
# - Update this path to wherever your course data lives.
# - For teaching, it is usually easiest to keep the .rds in the same folder as
#   the script (or in a /data folder).
poll_census_data <- readRDS("poll_census_data.rds")

# -----------------------------------------------------------------------------
# Step 2: Keep one row per year for the economic predictors before merging
# -----------------------------------------------------------------------------
econ_by_year <- forecast_data %>%
  distinct(
    year,
    unemployment_rate_Q1, unemployment_rate_Q2,
    gdp_Q1, gdp_Q2,
    cpi_Q1, cpi_Q2,
    gdp_change, cpi_change, unemploy_change
  )

# Merge state-level data with economic predictors
state_data <- poll_census_data %>%
  left_join(econ_by_year, by = "year")

# -----------------------------------------------------------------------------
# Step 3: Fit a state-level prediction model (trained on years < 2020)
# -----------------------------------------------------------------------------
# This model predicts vote share using:
# - polling average
# - year (time trend)
# - party indicator
# - demographic covariates

pred_results <- lm(
  vote_pct ~ poll_avg + year + party_simplified + white + black + asian + hispanic,
  data = subset(state_data, year < 2020)
)

# -----------------------------------------------------------------------------
# Step 4: Out-of-sample predictions for 2020 and beyond
# -----------------------------------------------------------------------------
out_of_sample <- predict(
  pred_results,
  newdata = subset(state_data, year >= 2020)
)

# Put predictions next to the observed outcomes
elect_outcomes <- state_data %>%
  filter(year >= 2020) %>%
  dplyr::select(year, state_po, party_simplified, candidate, vote_pct) %>%
  bind_cols(data.frame(vote_pred = out_of_sample))

# -----------------------------------------------------------------------------
# Step 5: Compute the vote-share difference for 2020 (Dem - Rep)
# -----------------------------------------------------------------------------
# We reshape to wide form so each state has a Dem and Rep vote share,
# then compute a difference.

elect_outcome_diff_2020 <- elect_outcomes %>%
  filter(year == 2020) %>%
  pivot_wider(
    id_cols = c(state_po, year),
    names_from = candidate,
    values_from = c(vote_pct, vote_pred),
    names_glue = "{candidate}_{.value}"
  )

# NOTE:
# - The next line assumes the candidate names in your dataset produce columns
#   called biden_vote_pct and trump_vote_pct.
# - If your candidate names differ, inspect colnames(elect_outcome_diff_2020)
#   and update the column names below.

vote_diff_2020 <- elect_outcome_diff_2020 %>%
  mutate(vote_diff = biden_vote_pct - trump_vote_pct) %>%
  distinct(vote_diff, state_po)

# -----------------------------------------------------------------------------
# Step 6: Prepare map geometry and merge in vote differences
# -----------------------------------------------------------------------------
# Get US state map polygons (excluding AK/HI by default)
us_states <- map_data("state")

# Map state names to postal abbreviations (state_po)
state_mapping <- data.frame(
  region = tolower(c(state.name, "district of columbia")),
  state_po = c(state.abb, "DC"),
  stringsAsFactors = FALSE
)

us_states <- us_states %>%
  left_join(state_mapping, by = "region")

# Merge the map polygons with vote differences
map_data <- left_join(us_states, vote_diff_2020, by = "state_po", relationship = "many-to-many")

# Keep only states with vote_diff available
map_data <- map_data %>%
  filter(!is.na(vote_diff))

# -----------------------------------------------------------------------------
# Step 7: Plot the 2020 vote-share difference map
# -----------------------------------------------------------------------------
ggplot(map_data, aes(long, lat, group = group, fill = vote_diff)) +
  geom_polygon(color = "white", size = 0.2) +
  coord_map("albers", lat0 = 39, lat1 = 45) +
  scale_fill_gradient2(
    low = "red",
    mid = "purple",
    high = "blue",
    midpoint = 0,
    limits = c(-max(abs(map_data$vote_diff)), max(abs(map_data$vote_diff))),
    name = "Vote Share Difference\n",
    labels = c("More\nRepublican", "Even", "More\nDemocratic"),
    breaks = c(-max(abs(map_data$vote_diff)), 0, max(abs(map_data$vote_diff)))
  ) +
  theme_bw() +
  labs(x = "Longitude", y = "Latitude")

# -----------------------------------------------------------------------------
# Practice Tasks (Students)
# -----------------------------------------------------------------------------
# 1) Data inspection:
#    - Print summaries of vote_data, combined_data, and forecast_data.
#    - Make sure you understand what each column represents.
#
# 2) Model edits:
#    - Try adding gdp_change or cpi_change to the national model.
#    - Try removing the poly(year, 2) term.
#
# 3) State model edits:
#    - Add economic predictors (unemploy_change, gdp_change, cpi_change).
#    - Compare model fit when you include vs exclude demographics.
#
# 4) Mapping edits:
#    - Try mapping predicted differences (vote_pred) instead of observed vote_pct.
#
# 5) Extension:
#    - Build a new visualization for "close states" (vote_diff near 0).
###############################################################################
