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

# Install required packages manually (kept commented out for teaching).
# In a real project, you should install once, not every run.
#
# install.packages(c("devtools", "renv", "logger", "tidyverse", "data.table", "estimatr", "broom"))

# Load libraries
library(logger)
library(tidyverse)
library(data.table)
library(estimatr)
library(broom)

# Dependency tracking (renv)
# Run ONE of the following depending on whether this is a new or existing project:
#
# (A) New project (run once):
# renv::init()
#
# (B) Existing renv project:
# renv::restore()
#
# (C) After you add/change packages:
# renv::snapshot()

###############################################
# SECTION 2: PROJECT DIRECTORY SETUP
###############################################

# Create necessary directories for our project
dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)         # Raw, unprocessed data (e.g., event logs)
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)   # Cleaned, processed data (analysis-ready)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)  # Plots and visualizations
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)   # Results and summaries

###############################################
# SECTION 3: LOGGING SETUP
###############################################

# Configure logging to track our analysis steps
logger::log_threshold(INFO)
logger::log_appender(appender_file("analysis_log.txt"))

###############################################
# SECTION 4: BIG DATA EXPERIMENT PIPELINE (SEQUENTIAL)
###############################################

###############################################
# STEP 0: GLOBAL SETTINGS
###############################################

# Reproducibility seed (classroom-friendly demonstration)
set.seed(123)

# "Big data" knobs (adjust upward if you want more scale)
n_users <- 100000     # number of users in the experiment
n_days  <- 14         # number of post-assignment days to log

log_info("Starting big data experiment pipeline")
log_info(paste0("n_users = ", n_users, " | n_days = ", n_days))

###############################################
# STEP 1: GENERATE SYNTHETIC USERS (UNIT TABLE)
###############################################

log_info("Generating synthetic user table")

users <- tibble(
  user_id = 1:n_users,
  platform = sample(c("ios", "android", "web"), n_users, replace = TRUE, prob = c(0.35, 0.35, 0.30)),
  cluster_id = sample(1:500, n_users, replace = TRUE),  # e.g., geography / community / school
  baseline_activity = rgamma(n_users, shape = 2, scale = 2),  # heavy-tailed activity
  signup_cohort = sample(c("cohort_A", "cohort_B", "cohort_C"), n_users, replace = TRUE, prob = c(0.40, 0.35, 0.25))
)

# Pre-treatment metric (placebo outcome) correlated with baseline_activity
users <- users %>%
  mutate(
    pre_metric = baseline_activity + rnorm(n_users, 0, 0.5)
  )

# Save raw user table
write.csv(users, "data/raw/users.csv", row.names = FALSE)
log_info("Saved: data/raw/users.csv")

###############################################
# STEP 2: BLOCKING + RANDOM ASSIGNMENT (SAVE ASSIGNMENT!)
###############################################

log_info("Creating blocked assignment table (and saving it)")

# Blocking: create deciles of baseline activity
users <- users %>%
  mutate(
    block = ntile(baseline_activity, 10)
  )

# Randomize within blocks (50/50)
assignment <- users %>%
  group_by(block) %>%
  mutate(
    treat = rbinom(n(), size = 1, prob = 0.5)
  ) %>%
  ungroup() %>%
  select(user_id, treat, block, platform, cluster_id, signup_cohort, baseline_activity, pre_metric) %>%
  mutate(
    assignment_date = as.Date("2026-04-16")
  )

# SAVE the assignment table (essential for reproducibility)
write.csv(assignment, "data/raw/assignment_table.csv", row.names = FALSE)
log_info("Saved: data/raw/assignment_table.csv")

###############################################
# STEP 3: GENERATE RAW EVENT LOGS (USER-DAY BIG TABLE)
###############################################

log_info("Generating synthetic event logs (user-day table)")

# Convert to data.table for fast cross joins and aggregation
dt_assign <- as.data.table(assignment)

# Day index table (post-assignment)
dt_days <- data.table(day = 1:n_days)
dt_days[, dummy := 1]
dt_assign[, dummy := 1]

# Cross join: users x days = user-day logs
dt_logs <- merge(dt_assign, dt_days, by = "dummy", allow.cartesian = TRUE)
dt_logs[, dummy := NULL]

# Date variable
dt_logs[, date := as.Date(assignment_date) + day - 1]

# Day-of-week effect
dt_logs[, dow := as.integer(format(date, "%u"))]  # 1=Mon ... 7=Sun

# Logging instrumentation: sometimes events are missing due to outages/bugs
dt_logs[, logged_ok := rbinom(.N, 1, 0.98)]  # 2% log dropout

# Underlying click intensity (Poisson rate)
dt_logs[, base_rate :=
          exp(-1.2 +
                0.15 * log1p(baseline_activity) +
                0.05 * (platform == "ios") +
                0.03 * (platform == "android") +
                0.02 * (dow %in% c(6,7)) +     # weekend bump
                0.01 * day                     # mild upward trend
          )
]

# Treatment effect on engagement (~5% multiplicative lift)
dt_logs[, click_rate := base_rate * exp(0.05 * treat)]

# Generate clicks (count outcome)
dt_logs[, clicks := rpois(.N, lambda = click_rate)]

# Conversion probability (logistic model)
dt_logs[, purchase_prob :=
          plogis(-5.0 +
                   0.08 * clicks +
                   0.10 * log1p(baseline_activity) +
                   0.15 * treat +
                   0.02 * (dow %in% c(6,7))
          )
]

dt_logs[, purchase := rbinom(.N, 1, purchase_prob)]

# Active day indicator
dt_logs[, active := as.integer(clicks > 0 | purchase > 0)]

# Apply logging dropout: if logged_ok == 0, events are missing
dt_logs[logged_ok == 0, clicks := NA_integer_]
dt_logs[logged_ok == 0, purchase := NA_integer_]
dt_logs[logged_ok == 0, active := NA_integer_]

# Save raw logs
fwrite(dt_logs, "data/raw/event_logs.csv")
log_info("Saved: data/raw/event_logs.csv")

###############################################
# STEP 4: BUILD AN ANALYSIS-READY DATASET (USER-LEVEL)
###############################################

log_info("Building analysis-ready dataset (user-level aggregation)")

dt_user <- dt_logs[, .(
  post_clicks = sum(clicks, na.rm = TRUE),
  post_purchases = sum(purchase, na.rm = TRUE),
  converted = as.integer(sum(purchase, na.rm = TRUE) > 0),
  days_observed = sum(!is.na(active)),
  missing_share = mean(is.na(active))
), by = .(user_id, treat, block, platform, cluster_id, signup_cohort, baseline_activity, pre_metric)]

fwrite(dt_user, "data/processed/analysis_dataset.csv")
log_info("Saved: data/processed/analysis_dataset.csv")

###############################################
# STEP 5: RANDOMIZATION CHECKS / BALANCE CHECKS
###############################################

log_info("Running randomization checks / balance checks")

dt_temp <- as_tibble(dt_user)

balance_table <- dt_temp %>%
  group_by(treat) %>%
  summarize(
    n = n(),
    mean_baseline_activity = mean(baseline_activity, na.rm = TRUE),
    mean_pre_metric = mean(pre_metric, na.rm = TRUE),
    mean_missing_share = mean(missing_share, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(balance_table, "outputs/tables/balance_means.csv", row.names = FALSE)

# Standardized mean difference (SMD): baseline_activity
mean_c <- dt_temp %>% filter(treat == 0) %>% summarize(m = mean(baseline_activity)) %>% pull(m)
mean_t <- dt_temp %>% filter(treat == 1) %>% summarize(m = mean(baseline_activity)) %>% pull(m)
sd_c   <- dt_temp %>% filter(treat == 0) %>% summarize(s = sd(baseline_activity)) %>% pull(s)
sd_t   <- dt_temp %>% filter(treat == 1) %>% summarize(s = sd(baseline_activity)) %>% pull(s)
sd_pool <- sqrt((sd_c^2 + sd_t^2) / 2)
smd_baseline_activity <- (mean_t - mean_c) / sd_pool

# SMD: pre_metric
mean_c2 <- dt_temp %>% filter(treat == 0) %>% summarize(m = mean(pre_metric)) %>% pull(m)
mean_t2 <- dt_temp %>% filter(treat == 1) %>% summarize(m = mean(pre_metric)) %>% pull(m)
sd_c2   <- dt_temp %>% filter(treat == 0) %>% summarize(s = sd(pre_metric)) %>% pull(s)
sd_t2   <- dt_temp %>% filter(treat == 1) %>% summarize(s = sd(pre_metric)) %>% pull(s)
sd_pool2 <- sqrt((sd_c2^2 + sd_t2^2) / 2)
smd_pre_metric <- (mean_t2 - mean_c2) / sd_pool2

smd_table <- tibble(
  variable = c("baseline_activity", "pre_metric"),
  smd = c(smd_baseline_activity, smd_pre_metric)
)

write.csv(smd_table, "outputs/tables/balance_smd.csv", row.names = FALSE)

###############################################
# STEP 6: ESTIMATE EXPERIMENTAL EFFECTS (ATE)
###############################################

log_info("Estimating treatment effects (ATE)")

# Difference in means (simple ITT estimator)
ate_converted <- with(dt_temp, mean(converted[treat == 1]) - mean(converted[treat == 0]))
ate_purchases <- with(dt_temp, mean(post_purchases[treat == 1]) - mean(post_purchases[treat == 0]))
ate_clicks    <- with(dt_temp, mean(post_clicks[treat == 1]) - mean(post_clicks[treat == 0]))

ate_simple <- tibble(
  outcome = c("converted", "post_purchases", "post_clicks"),
  ate_diff_in_means = c(ate_converted, ate_purchases, ate_clicks)
)

write.csv(ate_simple, "outputs/tables/ate_diff_in_means.csv", row.names = FALSE)

# Regression adjustment with robust SE (block FE + cluster-robust)
fit_conv <- lm_robust(converted ~ treat + baseline_activity + pre_metric + factor(block),
                      data = dt_temp,
                      clusters = cluster_id)

fit_pur  <- lm_robust(post_purchases ~ treat + baseline_activity + pre_metric + factor(block),
                      data = dt_temp,
                      clusters = cluster_id)

tidy_conv <- broom::tidy(fit_conv)
tidy_pur  <- broom::tidy(fit_pur)

write.csv(tidy_conv, "outputs/tables/regression_converted.csv", row.names = FALSE)
write.csv(tidy_pur,  "outputs/tables/regression_purchases.csv", row.names = FALSE)

###############################################
# STEP 7: VISUALIZATIONS (BIG DATA EXPERIMENT DIAGNOSTICS)
###############################################

log_info("Creating figures")

# Figure 1: distribution of baseline activity by treatment
p1 <- ggplot(dt_temp, aes(x = baseline_activity)) +
  geom_histogram(bins = 60) +
  facet_wrap(~ treat, ncol = 1, labeller = labeller(treat = c(`0` = "Control", `1` = "Treatment"))) +
  theme_bw() +
  labs(title = "Baseline Activity Distribution by Treatment Arm",
       x = "Baseline activity", y = "Count")

ggsave("outputs/figures/baseline_activity_by_treat.png", p1, width = 9, height = 6)

# Day-level table from logs (ignore missing purchase days)
dt_day <- dt_logs[, .(
  conversion_rate = mean(purchase, na.rm = TRUE),
  mean_clicks = mean(clicks, na.rm = TRUE),
  missing_share = mean(is.na(purchase))
), by = .(date, day, treat)]

dt_day_tbl <- as_tibble(dt_day)

# Figure 2: daily conversion rate by treatment arm
p2 <- ggplot(dt_day_tbl, aes(x = date, y = conversion_rate, group = factor(treat))) +
  geom_line() +
  theme_bw() +
  labs(title = "Daily Conversion Rate by Treatment Arm",
       x = "Date", y = "Conversion rate")

ggsave("outputs/figures/daily_conversion_rate.png", p2, width = 10, height = 4)

# Figure 3: daily missingness by treatment arm (instrumentation check)
p3 <- ggplot(dt_day_tbl, aes(x = date, y = missing_share, group = factor(treat))) +
  geom_line() +
  theme_bw() +
  labs(title = "Daily Missingness in Logged Purchases (Instrumentation Check)",
       x = "Date", y = "Share missing")

ggsave("outputs/figures/daily_missingness.png", p3, width = 10, height = 4)

###############################################
# STEP 8: PLACEBO TESTS (A/A + PLACEBO OUTCOME)
###############################################

log_info("Running placebo tests (A/A and placebo outcome)")

# Placebo outcome: pre_metric should not differ by treatment
placebo_pre <- lm_robust(pre_metric ~ treat + factor(block), data = dt_temp)
write.csv(broom::tidy(placebo_pre), "outputs/tables/placebo_pre_metric.csv", row.names = FALSE)

# A/A test: split CONTROL into two pseudo-arms many times and compute diff-in-means
set.seed(123)

control_ids <- dt_temp %>% filter(treat == 0) %>% pull(user_id)

aa_results <- tibble()

for (b in 1:200) {
  
  # Random split of control into A and A'
  a_group <- sample(control_ids, size = floor(length(control_ids) / 2), replace = FALSE)
  
  dt_sub <- dt_temp %>%
    filter(treat == 0) %>%
    mutate(aa = as.integer(user_id %in% a_group))
  
  aa_effect <- with(dt_sub, mean(converted[aa == 1]) - mean(converted[aa == 0]))
  
  aa_results <- aa_results %>%
    bind_rows(tibble(iter = b, aa_effect = aa_effect))
}

write.c
