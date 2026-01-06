###############################################################################
# Survey Exports + Cleaning + Codebooks Tutorial: R (Platform Data Example)
# Author: Jared Edgerton
# Date: Sys.Date()
#
# This script demonstrates:
#   1) Loading survey exports (CSV / Qualtrics-style structure)
#   2) Cleaning common survey fields (missingness, types, recodes)
#   3) Building a simple codebook (variable definitions + value meanings)
#   4) Adding labels (variable labels + value labels)
#   5) Running basic quality checks (speeding, attention checks, straightlining)
#   6) Saving a cleaned dataset for analysis
#
# Week context:
# - Surveys, Platforms, and Crowdsourcing
# - Coding lab: survey exports; cleaning; codebooks; labeling; quality checks.
# - Pre-class video: Data quality, missingness, and measurement in platform data.
# - Application papers: Berinsky et al. (2012) Political Analysis; Bisbee (2024) Political Analysis
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No conditional statements (no if/else).
# - Steps are written explicitly so students can follow how the code unfolds.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Install (if needed) and load the necessary libraries.

# install.packages(c("readr", "dplyr", "tidyr", "stringr", "ggplot2", "tibble",
#                    "janitor", "lubridate", "labelled", "haven"))
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(tibble)
library(janitor)
library(lubridate)
library(labelled)
library(haven)

# Create common project folders (safe to run repeatedly)
dir.create("data_raw", showWarnings = FALSE)
dir.create("data_processed", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)
dir.create("outputs", showWarnings = FALSE)
dir.create("src", showWarnings = FALSE)

# -----------------------------------------------------------------------------
# Part 1: Survey Export (Qualtrics-style) --- Loading data
# -----------------------------------------------------------------------------
# In real work, you'll start from a platform export:
#   - Qualtrics (CSV or .sav)
#   - Prolific / MTurk exports (CSV)
#   - Department pool exports (CSV)
#
# Qualtrics CSV note:
# - Qualtrics exports often include "extra header rows" (question text, etc.).
# - A common fix is: read_csv(..., skip = 2) to skip those rows.
#
# Examples (uncomment + edit file names once you have them):
#
# survey_raw <- read_csv("data_raw/survey_export.csv")
# survey_raw <- read_csv("data_raw/qualtrics_export.csv", skip = 2)
# survey_raw <- read_sav("data_raw/qualtrics_export.sav")
#
# For this tutorial, we create a small "toy export" that mimics typical fields.
# Replace this with your real export by overwriting survey_raw above.

survey_raw <- tibble(
  ResponseId = paste0("R_", sprintf("%03d", 1:12)),
  StartDate = c(
    "2026-02-05 10:00:05", "2026-02-05 10:01:10", "2026-02-05 10:02:02",
    "2026-02-05 10:03:41", "2026-02-05 10:05:00", "2026-02-05 10:05:45",
    "2026-02-05 10:06:10", "2026-02-05 10:07:33", "2026-02-05 10:08:09",
    "2026-02-05 10:09:18", "2026-02-05 10:10:07", "2026-02-05 10:11:55"
  ),
  EndDate = c(
    "2026-02-05 10:04:20", "2026-02-05 10:02:05", "2026-02-05 10:10:44",
    "2026-02-05 10:06:10", "2026-02-05 10:07:12", "2026-02-05 10:08:03",
    "2026-02-05 10:07:05", "2026-02-05 10:08:20", "2026-02-05 10:14:45",
    "2026-02-05 10:10:55", "2026-02-05 10:12:50", "2026-02-05 10:19:40"
  ),
  Duration_seconds = c(255, 55, 522, 149, 132, 138, 55, 47, 396, 97, 163, 465),
  IPAddress = c(
    "73.10.20.1", "73.10.20.2", "73.10.20.3", "73.10.20.4",
    "73.10.20.5", "73.10.20.6", "73.10.20.7", "73.10.20.8",
    "73.10.20.9", "73.10.20.10", "73.10.20.11", "73.10.20.12"
  ),
  Platform = c(
    "Prolific", "Prolific", "Prolific", "Prolific",
    "MTurk", "MTurk", "MTurk", "MTurk",
    "DepartmentPool", "DepartmentPool", "DepartmentPool", "DepartmentPool"
  ),
  Prolific_ID = c("5f1aA", "5f1aB", "5f1aC", "5f1aD", "", "", "", "", "", "", "", ""),
  MTurk_WorkerId = c("", "", "", "", "A11X", "A11Y", "A11Z", "A11W", "", "", "", ""),
  Consent = c("Yes", "Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"),
  Age = c("34", "29", "41", "", "22", "27", "18", "19", "45", "38", "33", "Prefer not to say"),
  Gender = c("Man", "Woman", "Woman", "Man", "Nonbinary", "Woman", "Man", "", "Woman", "Man", "Prefer not to say", "Woman"),
  PartyID = c("Democrat", "Independent", "Republican", "Democrat", "Independent", "Democrat", "", "Republican", "Democrat", "Independent", "Republican", "Independent"),
  AttentionCheck = c(
    "Strongly disagree", "Strongly disagree", "Strongly disagree", "Agree",
    "Strongly disagree", "Strongly disagree", "Agree", "Strongly disagree",
    "Strongly disagree", "Strongly disagree", "Strongly disagree", "Strongly disagree"
  ),
  Q1_policy = c("Agree", "Agree", "Strongly agree", "Agree", "Agree", "Agree", "Agree", "Agree", "Strongly agree", "Agree", "Agree", "Agree"),
  Q2_policy = c("Agree", "Agree", "Strongly agree", "Agree", "Agree", "Agree", "Agree", "Agree", "Strongly agree", "Agree", "Agree", "Agree"),
  Q3_policy = c("Agree", "Agree", "Strongly agree", "Agree", "Agree", "Agree", "Agree", "Agree", "Strongly agree", "Agree", "Agree", "Agree"),
  Q4_policy = c("Agree", "Agree", "Strongly agree", "Agree", "Agree", "Agree", "Agree", "Agree", "Strongly agree", "Agree", "Agree", "Agree")
)

# Inspect the raw export (students should do this every time)
print(survey_raw)
glimpse(survey_raw)

# -----------------------------------------------------------------------------
# Part 2: Cleaning --- names, types, and missing values
# -----------------------------------------------------------------------------
# Step 1: Standardize column names (consistent snake_case)
survey_clean <- survey_raw %>%
  clean_names()

# Inspect cleaned names
print(names(survey_clean))

# Step 2: Parse timestamps into POSIXct (platform metadata)
survey_clean <- survey_clean %>%
  mutate(
    start_date = ymd_hms(start_date, tz = "America/New_York"),
    end_date   = ymd_hms(end_date, tz = "America/New_York")
  )

# Step 3: Convert common "missing" strings into NA
survey_clean <- survey_clean %>%
  mutate(
    age = na_if(age, ""),
    age = na_if(age, "Prefer not to say"),
    gender = na_if(gender, ""),
    gender = na_if(gender, "Prefer not to say"),
    party_id = na_if(party_id, "")
  )

# Step 4: Convert age to numeric (after cleaning)
survey_clean <- survey_clean %>%
  mutate(
    age_num = as.numeric(age)
  )

# Step 5: Create duration in minutes (often easier to interpret)
survey_clean <- survey_clean %>%
  mutate(
    duration_min = duration_seconds / 60
  )

# Print a quick summary of key fields
survey_clean %>%
  select(response_id, platform, consent, duration_seconds, duration_min, age, age_num, gender, party_id) %>%
  print(n = 12)

# -----------------------------------------------------------------------------
# Part 3: Codebooks --- documenting variables and values
# -----------------------------------------------------------------------------
# A codebook describes:
#   - variable names
#   - what they measure / where they came from
#   - value meanings
#   - cleaning decisions
#
# Below is a simple codebook tibble you can expand.

codebook <- tibble(
  variable = c(
    "response_id", "start_date", "end_date", "duration_seconds",
    "ip_address", "platform", "prolific_id", "mturk_worker_id",
    "consent", "age_num", "gender", "party_id",
    "attention_check", "q1_policy", "q2_policy", "q3_policy", "q4_policy"
  ),
  description = c(
    "Unique respondent identifier (from survey platform)",
    "Survey start time (parsed)",
    "Survey end time (parsed)",
    "Time spent in survey (seconds, from platform metadata)",
    "IP address (platform metadata; treat as sensitive)",
    "Recruitment platform indicator (Prolific / MTurk / Department pool)",
    "Prolific participant ID (if applicable)",
    "MTurk worker ID (if applicable)",
    "Consent to participate (Yes/No)",
    "Age in years (numeric; missing if refused/blank)",
    "Self-reported gender (cleaned; missing if refused/blank)",
    "Self-reported party identification (blank -> NA)",
    "Attention check response (used for quality screening)",
    "Policy attitude item 1 (Likert text)",
    "Policy attitude item 2 (Likert text)",
    "Policy attitude item 3 (Likert text)",
    "Policy attitude item 4 (Likert text)"
  ),
  notes = c(
    "Do not share raw IDs publicly in a replication package",
    "Timezone set to America/New_York for class consistency",
    "Timezone set to America/New_York for class consistency",
    "Can be noisy if respondent leaves tab open",
    "Often should not be released publicly; check IRB / policy",
    "Platform differences can affect data quality",
    "Blank for non-Prolific respondents",
    "Blank for non-MTurk respondents",
    "Use to filter out non-consenting cases",
    "Converted from string; 'Prefer not to say' -> NA",
    "Converted 'Prefer not to say' and blank -> NA",
    "Blank -> NA",
    "Incorrect response indicates low attention",
    "Consider mapping to numeric 1--5 for analysis",
    "Consider mapping to numeric 1--5 for analysis",
    "Consider mapping to numeric 1--5 for analysis",
    "Consider mapping to numeric 1--5 for analysis"
  )
)

print(codebook)
write_csv(codebook, "outputs/week_codebook.csv")

# -----------------------------------------------------------------------------
# Part 4: Labeling --- variable labels and value labels
# -----------------------------------------------------------------------------
# Platforms sometimes export labelled data (especially SPSS exports).
# Even when you do not receive labels, you can add them in R for clarity.

# Add variable labels (stored as attributes)
var_label(survey_clean$response_id) <- "Unique respondent identifier"
var_label(survey_clean$duration_seconds) <- "Survey duration (seconds)"
var_label(survey_clean$age_num) <- "Age in years (numeric)"
var_label(survey_clean$party_id) <- "Party identification"
var_label(survey_clean$attention_check) <- "Attention check response"

# Create a numeric party_id variable (hard-coded mapping), then add value labels
survey_clean$party_id_num <- NA_real_
survey_clean$party_id_num[survey_clean$party_id == "Democrat"] <- 1
survey_clean$party_id_num[survey_clean$party_id == "Independent"] <- 2
survey_clean$party_id_num[survey_clean$party_id == "Republican"] <- 3

val_labels(survey_clean$party_id_num) <- c(
  Democrat = 1,
  Independent = 2,
  Republican = 3
)

# Create numeric Likert versions of the policy items (hard-coded mapping)
# Likert example:
#   Strongly disagree = 1
#   Disagree          = 2
#   Neither           = 3
#   Agree             = 4
#   Strongly agree    = 5

survey_clean$q1_num <- NA_real_
survey_clean$q1_num[survey_clean$q1_policy == "Strongly disagree"] <- 1
survey_clean$q1_num[survey_clean$q1_policy == "Disagree"] <- 2
survey_clean$q1_num[survey_clean$q1_policy == "Neither"] <- 3
survey_clean$q1_num[survey_clean$q1_policy == "Agree"] <- 4
survey_clean$q1_num[survey_clean$q1_policy == "Strongly agree"] <- 5

survey_clean$q2_num <- NA_real_
survey_clean$q2_num[survey_clean$q2_policy == "Strongly disagree"] <- 1
survey_clean$q2_num[survey_clean$q2_policy == "Disagree"] <- 2
survey_clean$q2_num[survey_clean$q2_policy == "Neither"] <- 3
survey_clean$q2_num[survey_clean$q2_policy == "Agree"] <- 4
survey_clean$q2_num[survey_clean$q2_policy == "Strongly agree"] <- 5

survey_clean$q3_num <- NA_real_
survey_clean$q3_num[survey_clean$q3_policy == "Strongly disagree"] <- 1
survey_clean$q3_num[survey_clean$q3_policy == "Disagree"] <- 2
survey_clean$q3_num[survey_clean$q3_policy == "Neither"] <- 3
survey_clean$q3_num[survey_clean$q3_policy == "Agree"] <- 4
survey_clean$q3_num[survey_clean$q3_policy == "Strongly agree"] <- 5

survey_clean$q4_num <- NA_real_
survey_clean$q4_num[survey_clean$q4_policy == "Strongly disagree"] <- 1
survey_clean$q4_num[survey_clean$q4_policy == "Disagree"] <- 2
survey_clean$q4_num[survey_clean$q4_policy == "Neither"] <- 3
survey_clean$q4_num[survey_clean$q4_policy == "Agree"] <- 4
survey_clean$q4_num[survey_clean$q4_policy == "Strongly agree"] <- 5

# Inspect a few labelled variables
survey_clean %>%
  select(response_id, party_id, party_id_num, q1_policy, q1_num) %>%
  print(n = 12)

# -----------------------------------------------------------------------------
# Part 5: Quality checks --- speeding, attention, missingness, straightlining
# -----------------------------------------------------------------------------
# Common quality checks in platform surveys:
# - Speeding (too fast)
# - Attention check failure
# - High missingness
# - Straightlining on grids
# - Duplicate IDs / duplicate IPs (platform-specific concerns)

# Check 1: Speeding flag (example: < 120 seconds)
survey_clean$flag_fast <- survey_clean$duration_seconds < 120

# Check 2: Attention check flag (toy: correct answer is "Strongly disagree")
survey_clean$flag_attention_fail <- survey_clean$attention_check != "Strongly disagree"

# Check 3: Missingness share across a set of key variables
key_vars <- survey_clean %>%
  select(age_num, gender, party_id_num, q1_num, q2_num, q3_num, q4_num)

survey_clean$missing_share <- rowMeans(is.na(key_vars))
survey_clean$flag_missing_high <- survey_clean$missing_share > 0.30

# Check 4: Straightlining on policy grid items (all identical)
survey_clean$flag_straightline <- (
  survey_clean$q1_num == survey_clean$q2_num &
    survey_clean$q2_num == survey_clean$q3_num &
    survey_clean$q3_num == survey_clean$q4_num
)

# Check 5: Consent
survey_clean$flag_no_consent <- survey_clean$consent != "Yes"

# Check 6: Duplicate platform IDs (example: Prolific_ID duplicates)
dup_prolific <- survey_clean %>%
  filter(prolific_id != "") %>%
  dplyr::count(prolific_id) 
print(dup_prolific)

# Summarize flags
flag_summary <- survey_clean %>%
  dplyr::summarise(
    n_total = dplyr::n(),
    n_fast = sum(flag_fast, na.rm = TRUE),
    n_attention_fail = sum(flag_attention_fail, na.rm = TRUE),
    n_missing_high = sum(flag_missing_high, na.rm = TRUE),
    n_straightline = sum(flag_straightline, na.rm = TRUE),
    n_no_consent = sum(flag_no_consent, na.rm = TRUE)
  )
print(flag_summary)

# Inspect which rows are flagged
survey_clean %>%
  select(response_id, platform, duration_seconds, attention_check, missing_share,
         flag_fast, flag_attention_fail, flag_missing_high, flag_straightline, flag_no_consent) %>%
  arrange(desc(flag_no_consent), desc(flag_fast), desc(flag_attention_fail)) %>%
  print(n = 12)

# -----------------------------------------------------------------------------
# Part 6: Create an analysis-ready dataset (filter out low-quality cases)
# -----------------------------------------------------------------------------
# NOTE: these rules are choices. In real projects, justify them and consider
# sensitivity analyses (e.g., results with/without straightliners).

survey_final <- survey_clean %>%
  filter(
    flag_no_consent == FALSE,
    flag_fast == FALSE,
    flag_attention_fail == FALSE,
    flag_missing_high == FALSE
    # We are not automatically excluding straightliners here.
  )

cat("\n------------------------------\n")
cat("Rows before/after filtering\n")
cat("------------------------------\n")
cat("Before:", nrow(survey_clean), "\n")
cat("After: ", nrow(survey_final), "\n\n")

write_csv(survey_final, "data_processed/week_survey_clean.csv")

# -----------------------------------------------------------------------------
# Part 7: Quick visual checks (plots)
# -----------------------------------------------------------------------------

# Plot 1: Duration distribution
p_duration <- ggplot(survey_clean, aes(x = duration_seconds)) +
  geom_histogram(bins = 12) +
  labs(
    title = "Survey Duration Distribution",
    x = "Duration (seconds)",
    y = "Count"
  )
ggsave("figures/week_duration_hist.png", p_duration, width = 8, height = 4)

# Plot 2: Missingness share distribution
p_missing <- ggplot(survey_clean, aes(x = missing_share)) +
  geom_histogram(bins = 10) +
  labs(
    title = "Missingness Share Across Key Variables",
    x = "Share missing (per respondent)",
    y = "Count"
  )
ggsave("figures/week_missingness_hist.png", p_missing, width = 8, height = 4)

# Plot 3: Share flagged by platform (speeding + attention)
platform_flags <- survey_clean %>%
  group_by(platform) %>%
  dplyr::summarize(
    n = n(),
    share_fast = mean(flag_fast, na.rm = TRUE),
    share_attention_fail = mean(flag_attention_fail, na.rm = TRUE),
    .groups = "drop"
  )
print(platform_flags)

platform_flags_long <- platform_flags %>%
  pivot_longer(cols = c(share_fast, share_attention_fail),
               names_to = "flag_type",
               values_to = "share_flagged")

p_platform <- ggplot(platform_flags_long, aes(x = platform, y = share_flagged)) +
  geom_col() +
  facet_wrap(~ flag_type) +
  labs(
    title = "Share Flagged by Platform (Toy Data)",
    x = "Platform",
    y = "Share flagged"
  )
ggsave("figures/week_flags_by_platform.png", p_platform, width = 8, height = 4)

# -----------------------------------------------------------------------------
# Part 8: One simple analysis example (optional in class)
# -----------------------------------------------------------------------------
# Example: compute a simple scale and summarize by party_id.

survey_final$policy_scale <- rowMeans(
  survey_final %>% select(q1_num, q2_num, q3_num, q4_num),
  na.rm = TRUE
)

scale_summary <- survey_final %>%
  group_by(party_id) %>%
  dplyr::summarize(
    n = n(),
    mean_scale = mean(policy_scale, na.rm = TRUE),
    median_scale = median(policy_scale, na.rm = TRUE),
    .groups = "drop"
  )

print(scale_summary)
write_csv(scale_summary, "outputs/week_policy_scale_by_party.csv")
