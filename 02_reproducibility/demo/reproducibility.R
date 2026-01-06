###############################################################################
# Week 9 Assignment: Reproducible Analysis with poliscitools (R)
# Author: Jared Edgerton
# Date: Sys.Date()
#
# This script demonstrates:
#   1) Project setup for reproducible research (folders + dependency tracking)
#   2) Logging key steps in an analysis pipeline
#   3) A complete analysis workflow (data -> cleaning -> model -> bootstrap -> plots)
#   4) Saving outputs (figures, tables, session info) for replication
#   5) Writing a simple Dockerfile + README for portability
#   6) A basic reproducibility check (rerun pipeline with same seed)
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No conditional statements (no if/else).
# - Steps are explicit so students can follow and modify each piece.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Install (if needed) and load the necessary libraries.
#
# For teaching: keep installation lines commented out so students can run them
# manually if needed.

# install.packages(c("renv", "logger", "devtools", "tidyverse"))
# install.packages(c("plyr", "dplyr", "ggplot2"))

library(renv)       # Dependency management (renv.lock)
library(logger)     # Logging pipeline steps
library(devtools)   # Installing packages from GitHub
library(tidyverse)  # Data manipulation + plotting
library(plyr)
library(dplyr)
library(ggplot2)

# -----------------------------------------------------------------------------
# Part 0: Dependency Management (renv) + Custom Package Install
# -----------------------------------------------------------------------------
# renv::init() creates a project-local library and an renv.lock file.
# Teaching workflow:
# - Run renv::init() ONCE at the start of a project (not every time).
# - After that, use renv::snapshot() to record package versions.
# - On another machine, use renv::restore() to recreate the environment.
#
# NOTE: We leave renv::init() commented out to avoid re-initializing by accident.

# renv::init()

# Install your custom package from GitHub (replace with your repo).
# devtools::install_github("yourusername/poliscitools")

library(poliscitools)

# -----------------------------------------------------------------------------
# Part 1: Project Directory Setup
# -----------------------------------------------------------------------------
# Reproducible projects should separate:
# - raw data (unchanged inputs)
# - processed data (cleaned outputs)
# - figures and tables (final outputs)

dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# Part 2: Logging Setup
# -----------------------------------------------------------------------------
# Logging creates an audit trail:
# - What ran
# - In what order
# - With what parameters
# - Where outputs were written

logger::log_threshold(DEBUG)
logger::log_appender(appender_file("analysis_log.txt"))

# -----------------------------------------------------------------------------
# Part 3: Main Analysis Pipeline (Single Run)
# -----------------------------------------------------------------------------
# Pipeline overview:
#   1) Load / create data
#   2) Save raw data
#   3) Clean data
#   4) Save processed data
#   5) Run analysis models
#   6) Bootstrap a simple model (example uncertainty)
#   7) Create plots
#   8) Save tables + session info

set.seed(123)  # Reproducible randomness for the full pipeline

log_info("Starting analysis pipeline")

# -----------------------------------------------------------------------------
# Step 1: Load / generate example data
# -----------------------------------------------------------------------------
# In this template, we use poliscitools::example_data (synthetic example).
log_info("Loading example voter data")
voter_data <- poliscitools::example_data

# -----------------------------------------------------------------------------
# Step 2: Save raw data (unchanged input)
# -----------------------------------------------------------------------------
log_info("Saving raw data")
write.csv(voter_data, "data/raw/voter_data.csv", row.names = FALSE)

# -----------------------------------------------------------------------------
# Step 3: Clean the data
# -----------------------------------------------------------------------------
# clean_political_data() is assumed to be provided by poliscitools.
log_info("Cleaning voter data")
cleaned_data <- clean_political_data(voter_data)

# -----------------------------------------------------------------------------
# Step 4: Save processed data
# -----------------------------------------------------------------------------
log_info("Saving processed data")
write.csv(cleaned_data, "data/processed/cleaned_voter_data.csv", row.names = FALSE)

# -----------------------------------------------------------------------------
# Step 5: Analyze turnout
# -----------------------------------------------------------------------------
# analyze_turnout() is assumed to be provided by poliscitools.
log_info("Analyzing turnout")
turnout_results <- analyze_turnout(cleaned_data)

# -----------------------------------------------------------------------------
# Step 6: Bootstrap example (uncertainty demo)
# -----------------------------------------------------------------------------
# Bootstrap idea:
# - Resample rows with replacement
# - Fit a simple model each time
# - Store the coefficients across bootstrap runs

log_info("Bootstrapping a simple turnout model (lm)")
row_boot  <- nrow(cleaned_data)

boot_coef <- data.frame()
for (z in 1:1000) {
  boot_data  <- cleaned_data[sample(row_boot, row_boot, replace = TRUE), ]
  boot_model <- lm(turnout ~ age + party, data = boot_data)
  boot_coef  <- boot_coef |>
    rbind(t(coef(boot_model)))
}

# Clean column names for easier reading
colnames(boot_coef) <- tolower(gsub("\\(|\\)|party", "", colnames(boot_coef)))

# -----------------------------------------------------------------------------
# Step 7: Visualizations
# -----------------------------------------------------------------------------
log_info("Creating visualizations")

# Turnout by party (stacked proportions)
party_plot <- ggplot(cleaned_data, aes(x = party, fill = as.factor(turnout))) +
  geom_bar(position = "fill") +
  labs(
    title = "Voter Turnout by Party",
    y = "Proportion",
    x = "Party",
    fill = "Voted"
  ) +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2")

ggsave("outputs/figures/party_turnout.pdf", party_plot)

# Turnout by age group (stacked proportions)
age_plot <- ggplot(cleaned_data, aes(x = age_group, fill = as.factor(turnout))) +
  geom_bar(position = "fill") +
  labs(
    title = "Voter Turnout by Age Group",
    y = "Proportion",
    x = "Age Group",
    fill = "Voted"
  ) +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2")

ggsave("outputs/figures/age_turnout.pdf", age_plot)

# -----------------------------------------------------------------------------
# Step 8: Save numerical results
# -----------------------------------------------------------------------------
log_info("Saving analysis results (tables)")
saveRDS(turnout_results, "outputs/tables/turnout_results.rds")
saveRDS(boot_coef, "outputs/tables/boot_results.rds")

# -----------------------------------------------------------------------------
# Step 9: Save session information
# -----------------------------------------------------------------------------
# sessionInfo() records R version + package versions (critical for reproducibility)
log_info("Saving session information")
writeLines(capture.output(sessionInfo()), "outputs/session_info.txt")

log_info("Analysis pipeline completed successfully")

# Optional: keep a compact object in memory for students to inspect
results <- list(
  turnout_results = turnout_results,
  boot_coef = boot_coef
)

# -----------------------------------------------------------------------------
# Part 4: Dockerfile Creation (Sequential)
# -----------------------------------------------------------------------------
# Docker lets someone else run the project in the same environment.
# This section writes a simple Dockerfile as plain text.

log_info("Writing Dockerfile")

dockerfile_content <- c(
  "FROM rocker/tidyverse:latest",
  "",
  "# Install required packages",
  "RUN R -e \"install.packages(c('renv', 'logger', 'devtools'))\"",
  "",
  "# Copy project files into container",
  "COPY . /project",
  "WORKDIR /project",
  "",
  "# Install poliscitools package (replace with your GitHub repo)",
  "RUN R -e \"devtools::install_github('yourusername/poliscitools')\"",
  "",
  "# Restore renv environment (uses renv.lock)",
  "RUN R -e \"renv::restore()\"",
  "",
  "# Command to run the analysis script",
  "# NOTE: Update the filename to match the script you want executed in Docker",
  "CMD [\"Rscript\", \"-e\", \"source('reproducibility.R')\"]"
)

writeLines(dockerfile_content, "Dockerfile")

# -----------------------------------------------------------------------------
# Part 5: README Creation (Sequential)
# -----------------------------------------------------------------------------
# A README documents:
# - what the project is
# - how to install dependencies
# - how to run the analysis
# - where outputs appear

log_info("Writing README.md")

readme_content <- c(
  "# Voter Turnout Analysis (Reproducible Template)",
  "",
  "## Overview",
  "This project analyzes voter turnout patterns using the poliscitools package.",
  "",
  "## Reproducibility",
  "- Package versions are tracked with **renv** (renv.lock).",
  "- Outputs are saved in **outputs/** (figures, tables, session_info).",
  "",
  "## Installation",
  "1. Clone this repository",
  "2. Restore dependencies with:",
  "```r",
  "renv::restore()",
  "```",
  "3. Install poliscitools (if needed):",
  "```r",
  "devtools::install_github('yourusername/poliscitools')",
  "```",
  "",
  "## Usage",
  "Run the analysis by sourcing the script:",
  "```r",
  "source('reproducibility.R')",
  "```",
  "",
  "## Docker Usage",
  "```bash",
  "docker build -t voter-analysis .",
  "docker run -v $(pwd)/outputs:/project/outputs voter-analysis",
  "```",
  "",
  "## Output",
  "- Figures: outputs/figures/",
  "- Tables: outputs/tables/",
  "- Session info: outputs/session_info.txt"
)

writeLines(readme_content, "README.md")

# -----------------------------------------------------------------------------
# Part 6: Reproducibility Testing (Minimal Demo)
# -----------------------------------------------------------------------------
# Goal:
# - Rerun the key pipeline steps with the same random seed
# - Compare resulting objects for exact identity
#
# Teaching note:
# - This is a simplified check (one project, one machine).
# - Cross-platform reproducibility is a harder problem (OS, BLAS, etc.).

log_info("Running a minimal reproducibility check (two runs)")

# ---- Run 1 (reset seed) ----
set.seed(123)
voter_data_1   <- poliscitools::example_data
cleaned_data_1 <- clean_political_data(voter_data_1)
turnout_1      <- analyze_turnout(cleaned_data_1)

row_boot_1 <- nrow(cleaned_data_1)
boot_coef_1 <- data.frame()
for (z in 1:200) {  # fewer reps for the reproducibility demo
  boot_data_1  <- cleaned_data_1[sample(row_boot_1, row_boot_1, replace = TRUE), ]
  boot_model_1 <- lm(turnout ~ age + party, data = boot_data_1)
  boot_coef_1  <- boot_coef_1 |> rbind(t(coef(boot_model_1)))
}
colnames(boot_coef_1) <- tolower(gsub("\\(|\\)|party", "", colnames(boot_coef_1)))

# ---- Run 2 (reset seed) ----
set.seed(123)
voter_data_2   <- poliscitools::example_data
cleaned_data_2 <- clean_political_data(voter_data_2)
turnout_2      <- analyze_turnout(cleaned_data_2)

row_boot_2 <- nrow(cleaned_data_2)
boot_coef_2 <- data.frame()
for (z in 1:200) {
  boot_data_2  <- cleaned_data_2[sample(row_boot_2, row_boot_2, replace = TRUE), ]
  boot_model_2 <- lm(turnout ~ age + party, data = boot_data_2)
  boot_coef_2  <- boot_coef_2 |> rbind(t(coef(boot_model_2)))
}
colnames(boot_coef_2) <- tolower(gsub("\\(|\\)|party", "", colnames(boot_coef_2)))

# ---- Compare objects (no if/else; just print + log) ----
same_turnout  <- identical(turnout_1, turnout_2)
same_bootcoef <- identical(boot_coef_1, boot_coef_2)

log_info(paste("Reproducibility check — turnout identical:", same_turnout))
log_info(paste("Reproducibility check — boot_coef identical:", same_bootcoef))

print(data.frame(
  turnout_identical = same_turnout,
  bootcoef_identical = same_bootcoef
))

# -----------------------------------------------------------------------------
# Student Tasks
# -----------------------------------------------------------------------------
# 1) Extend visualizations:
#    - Add a turnout plot by state (if available in your data)
#    - Add a turnout plot by education (if available)
#
# 2) Add error handling (advanced):
#    - Validate inputs (columns exist, types are correct)
#    - Add more logging around file reads/writes
#
# 3) Extend analysis:
#    - Add additional covariates (education, income, gender)
#    - Compare models (glm vs lm, interactions)
#
# 4) Push changes to GitHub:
#    - Commit small changes often
#    - Include an updated README if workflow changes
#
# 5) Cross-platform reproducibility:
#    - Run the project on a different OS and compare outputs
#    - Use renv.lock + Docker to reduce platform differences
###############################################################################
