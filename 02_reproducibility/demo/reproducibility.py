###############################################################################
# Reproducible Analysis Template (Python)
# Author: Jared Edgerton
# Date: date.today()
#
# This script demonstrates:
#   1) Project setup for reproducible research (folders + dependency tracking)
#   2) Logging key steps in an analysis pipeline
#   3) A complete analysis workflow (data -> cleaning -> model -> bootstrap -> plots)
#   4) Saving outputs (figures, tables, environment info) for replication
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
# Install (if needed) with:
#   pip install numpy pandas matplotlib statsmodels
#
# Optional (recommended for reproducibility metadata):
#   pip install pip-tools
#
# Notes:
# - For dependency tracking, this script writes:
#     * requirements_frozen.txt (via pip freeze)
#     * python_environment.txt (Python + platform info)
# - For a true reproducible environment, pair this with:
#     * a pinned requirements.txt (or pip-tools / poetry lockfile)
#     * a Dockerfile

import os
import sys
import platform
import subprocess
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

from datetime import date

# Reproducibility seed for the full pipeline
np.random.seed(123)

# -----------------------------------------------------------------------------
# Part 1: Project Directory Setup
# -----------------------------------------------------------------------------
# Reproducible projects should separate:
# - raw data (unchanged inputs)
# - processed data (cleaned outputs)
# - figures and tables (final outputs)

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

# -----------------------------------------------------------------------------
# Part 2: Logging Setup
# -----------------------------------------------------------------------------
# Logging creates an audit trail:
# - What ran
# - In what order
# - With what parameters
# - Where outputs were written

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("analysis_log.txt", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)

logging.info("Starting analysis pipeline")

# -----------------------------------------------------------------------------
# Part 3: Dependency / Environment Tracking (Sequential)
# -----------------------------------------------------------------------------
# These files are part of a reproducibility bundle:
# - python_environment.txt: interpreter + OS info
# - requirements_frozen.txt: pip freeze snapshot (package versions as installed)

logging.info("Writing python_environment.txt")
env_lines = [
    f"Date: {date.today()}",
    f"Python: {sys.version}",
    f"Executable: {sys.executable}",
    f"Platform: {platform.platform()}",
    f"Machine: {platform.machine()}",
    f"Processor: {platform.processor()}",
]
with open("outputs/python_environment.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(env_lines) + "\n")

logging.info("Writing requirements_frozen.txt (pip freeze)")
freeze_out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
with open("outputs/requirements_frozen.txt", "w", encoding="utf-8") as f:
    f.write(freeze_out)

# -----------------------------------------------------------------------------
# Part 4: Main Analysis Pipeline (Single Run)
# -----------------------------------------------------------------------------
# Pipeline overview:
#   1) Generate data
#   2) Save raw data
#   3) Clean data
#   4) Save processed data
#   5) Run analysis models
#   6) Bootstrap a simple model (example uncertainty)
#   7) Create plots
#   8) Save tables

# -----------------------------------------------------------------------------
# Step 1: Generate example voter data (synthetic)
# -----------------------------------------------------------------------------
# Columns:
# - age: numeric
# - party: categorical (Democrat, Republican, Independent)
# - turnout: binary (0/1)
# - age_group: derived categorical
#
# Data-generating idea (teaching simplification):
# - turnout increases with age
# - turnout differs by party

logging.info("Generating synthetic voter dataset")

n = 5000

age = np.random.randint(18, 90, size=n)

party = np.random.choice(
    ["Democrat", "Republican", "Independent"],
    size=n,
    replace=True,
    p=[0.46, 0.46, 0.08],
)

party_effect = np.zeros(n, dtype=float)
party_effect[party == "Democrat"] = 0.05
party_effect[party == "Republican"] = 0.03
party_effect[party == "Independent"] = -0.02

age_effect = (age - 18) / (90 - 18) * 0.35

base = 0.35
p_turnout = base + age_effect + party_effect + np.random.normal(0, 0.05, size=n)
p_turnout = np.clip(p_turnout, 0.01, 0.99)

turnout = (np.random.rand(n) < p_turnout).astype(int)

voter_data = pd.DataFrame(
    {
        "age": age,
        "party": party,
        "turnout": turnout,
    }
)

# -----------------------------------------------------------------------------
# Step 2: Save raw data (unchanged input)
# -----------------------------------------------------------------------------
logging.info("Saving raw data to data/raw/voter_data.csv")
voter_data.to_csv("data/raw/voter_data.csv", index=False)

# -----------------------------------------------------------------------------
# Step 3: Clean the data (sequential)
# -----------------------------------------------------------------------------
# Typical cleaning steps:
# - enforce types
# - standardize category strings
# - drop missingness
# - create derived variables

logging.info("Cleaning voter data")

cleaned_data = voter_data.copy()

cleaned_data["age"] = pd.to_numeric(cleaned_data["age"], errors="coerce")
cleaned_data["turnout"] = pd.to_numeric(cleaned_data["turnout"], errors="coerce")

cleaned_data["party"] = cleaned_data["party"].astype(str).str.strip()
cleaned_data["party"] = cleaned_data["party"].str.replace(r"\s+", " ", regex=True)

cleaned_data = cleaned_data.dropna(subset=["age", "party", "turnout"]).reset_index(drop=True)

# Age group bins (explicit cut points)
cleaned_data["age_group"] = pd.cut(
    cleaned_data["age"],
    bins=[17, 29, 44, 64, 120],
    labels=["18-29", "30-44", "45-64", "65+"],
    include_lowest=True,
)

# -----------------------------------------------------------------------------
# Step 4: Save processed data
# -----------------------------------------------------------------------------
logging.info("Saving processed data to data/processed/cleaned_voter_data.csv")
cleaned_data.to_csv("data/processed/cleaned_voter_data.csv", index=False)

# -----------------------------------------------------------------------------
# Step 5: Analyze turnout (descriptive table)
# -----------------------------------------------------------------------------
# Example output:
# - turnout rate by party
# - turnout rate by age group

logging.info("Computing turnout summaries")

turnout_by_party = (
    cleaned_data
    .groupby("party", as_index=False)
    .agg(
        turnout_rate=("turnout", "mean"),
        n=("turnout", "size"),
    )
)

turnout_by_age = (
    cleaned_data
    .groupby("age_group", as_index=False)
    .agg(
        turnout_rate=("turnout", "mean"),
        n=("turnout", "size"),
    )
)

turnout_results = {
    "turnout_by_party": turnout_by_party,
    "turnout_by_age_group": turnout_by_age,
}

logging.info("Saving turnout summary tables")
turnout_by_party.to_csv("outputs/tables/turnout_by_party.csv", index=False)
turnout_by_age.to_csv("outputs/tables/turnout_by_age_group.csv", index=False)

# -----------------------------------------------------------------------------
# Step 6: Bootstrap a simple model (uncertainty demo)
# -----------------------------------------------------------------------------
# Bootstrap idea:
# - Resample rows with replacement
# - Fit a simple model each time
# - Store coefficients across bootstrap runs
#
# Model:
# - Linear probability model (OLS) for teaching simplicity:
#     turnout ~ age + C(party)

logging.info("Bootstrapping a simple turnout model (OLS)")

row_boot = cleaned_data.shape[0]
B = 1000

boot_rows = np.random.randint(0, row_boot, size=(B, row_boot))

boot_coefs = []

for z in range(B):
    boot_data = cleaned_data.iloc[boot_rows[z]].copy()
    boot_model = smf.ols("turnout ~ age + C(party)", data=boot_data).fit()
    boot_coefs.append(boot_model.params)

boot_coef = pd.DataFrame(boot_coefs).reset_index(drop=True)

# Round for stable printing and comparison
boot_coef_round = boot_coef.round(10)

logging.info("Saving bootstrap coefficients")
boot_coef_round.to_csv("outputs/tables/boot_coefficients.csv", index=False)

# -----------------------------------------------------------------------------
# Step 7: Visualizations
# -----------------------------------------------------------------------------
# Turnout by party (stacked proportions)
# Turnout by age group (stacked proportions)

logging.info("Creating visualizations")

# --- Party turnout plot (stacked proportions) ---
party_ct = pd.crosstab(cleaned_data["party"], cleaned_data["turnout"], normalize="index")
party_ct = party_ct.rename(columns={0: "No", 1: "Yes"})

plt.figure()
party_ct.plot(kind="bar", stacked=True)
plt.title("Voter Turnout by Party")
plt.xlabel("Party")
plt.ylabel("Proportion")
plt.tight_layout()
plt.savefig("outputs/figures/party_turnout.png", dpi=150)
plt.close()

# --- Age-group turnout plot (stacked proportions) ---
age_ct = pd.crosstab(cleaned_data["age_group"], cleaned_data["turnout"], normalize="index")
age_ct = age_ct.rename(columns={0: "No", 1: "Yes"})

plt.figure()
age_ct.plot(kind="bar", stacked=True)
plt.title("Voter Turnout by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Proportion")
plt.tight_layout()
plt.savefig("outputs/figures/age_turnout.png", dpi=150)
plt.close()

logging.info("Saved figures: outputs/figures/party_turnout.png and outputs/figures/age_turnout.png")

# -----------------------------------------------------------------------------
# Step 8: Save a compact “results bundle”
# -----------------------------------------------------------------------------
# For teaching, we write a single CSV for quick inspection as well.

logging.info("Saving a compact results bundle")
turnout_bundle = turnout_by_party.copy()
turnout_bundle = turnout_bundle.rename(columns={"turnout_rate": "turnout_rate_party"})
turnout_bundle.to_csv("outputs/tables/results_bundle.csv", index=False)

logging.info("Analysis pipeline completed successfully")

# -----------------------------------------------------------------------------
# Part 5: Dockerfile Creation (Sequential)
# -----------------------------------------------------------------------------
# Docker lets someone else run the project in the same environment.
# This section writes a simple Dockerfile as plain text.

logging.info("Writing Dockerfile")

dockerfile_content = [
    "FROM python:3.12-slim",
    "",
    "# Set working directory",
    "WORKDIR /project",
    "",
    "# Copy project files into container",
    "COPY . /project",
    "",
    "# Install dependencies",
    "# Option A: install from a pinned requirements.txt you maintain",
    "# COPY requirements.txt /project/requirements.txt",
    "# RUN pip install --no-cache-dir -r requirements.txt",
    "",
    "# Option B (teaching convenience): install a minimal set directly",
    "RUN pip install --no-cache-dir numpy pandas matplotlib statsmodels",
    "",
    "# Run the analysis script",
    "CMD [\"python\", \"reproducibility.py\"]",
]

with open("Dockerfile", "w", encoding="utf-8") as f:
    f.write("\n".join(dockerfile_content) + "\n")

# -----------------------------------------------------------------------------
# Part 6: README Creation (Sequential)
# -----------------------------------------------------------------------------
# A README documents:
# - what the project is
# - how to install dependencies
# - how to run the analysis
# - where outputs appear

logging.info("Writing README.md")

readme_content = [
    "# Voter Turnout Analysis (Reproducible Template — Python)",
    "",
    "## Overview",
    "This project analyzes voter turnout patterns using a fully reproducible pipeline.",
    "",
    "## Reproducibility",
    "- Outputs are saved in `outputs/` (figures, tables, environment info).",
    "- Package snapshot is saved via `pip freeze` to `outputs/requirements_frozen.txt`.",
    "",
    "## Installation",
    "1. Create and activate a virtual environment (recommended).",
    "2. Install dependencies:",
    "```bash",
    "pip install numpy pandas matplotlib statsmodels",
    "```",
    "",
    "## Usage",
    "Run the analysis script:",
    "```bash",
    "python reproducibility.py",
    "```",
    "",
    "## Docker Usage",
    "```bash",
    "docker build -t voter-analysis-py .",
    "docker run -v $(pwd)/outputs:/project/outputs voter-analysis-py",
    "```",
    "",
    "## Output",
    "- Figures: `outputs/figures/`",
    "- Tables: `outputs/tables/`",
    "- Environment: `outputs/python_environment.txt`, `outputs/requirements_frozen.txt`",
]

with open("README.md", "w", encoding="utf-8") as f:
    f.write("\n".join(readme_content) + "\n")

# -----------------------------------------------------------------------------
# Part 7: Reproducibility Testing (Minimal Demo)
# -----------------------------------------------------------------------------
# Goal:
# - Rerun the key pipeline steps with the same random seed
# - Compare resulting objects for exact identity
#
# Teaching note:
# - This is a simplified check (one project, one machine).
# - Cross-platform reproducibility is a harder problem (OS, BLAS, etc.).

logging.info("Running a minimal reproducibility check (two runs)")

# ---- Run 1 (reset seed) ----
np.random.seed(123)

n1 = 5000
age_1 = np.random.randint(18, 90, size=n1)
party_1 = np.random.choice(["Democrat", "Republican", "Independent"], size=n1, replace=True, p=[0.46, 0.46, 0.08])

party_effect_1 = np.zeros(n1, dtype=float)
party_effect_1[party_1 == "Democrat"] = 0.05
party_effect_1[party_1 == "Republican"] = 0.03
party_effect_1[party_1 == "Independent"] = -0.02

age_effect_1 = (age_1 - 18) / (90 - 18) * 0.35
base_1 = 0.35
p_turnout_1 = base_1 + age_effect_1 + party_effect_1 + np.random.normal(0, 0.05, size=n1)
p_turnout_1 = np.clip(p_turnout_1, 0.01, 0.99)
turnout_1 = (np.random.rand(n1) < p_turnout_1).astype(int)

voter_data_1 = pd.DataFrame({"age": age_1, "party": party_1, "turnout": turnout_1})

cleaned_1 = voter_data_1.copy()
cleaned_1["age"] = pd.to_numeric(cleaned_1["age"], errors="coerce")
cleaned_1["turnout"] = pd.to_numeric(cleaned_1["turnout"], errors="coerce")
cleaned_1["party"] = cleaned_1["party"].astype(str).str.strip()
cleaned_1["party"] = cleaned_1["party"].str.replace(r"\s+", " ", regex=True)
cleaned_1 = cleaned_1.dropna(subset=["age", "party", "turnout"]).reset_index(drop=True)
cleaned_1["age_group"] = pd.cut(cleaned_1["age"], bins=[17, 29, 44, 64, 120], labels=["18-29", "30-44", "45-64", "65+"], include_lowest=True)

turnout_by_party_1 = cleaned_1.groupby("party", as_index=False).agg(turnout_rate=("turnout", "mean"), n=("turnout", "size"))
row_boot_1 = cleaned_1.shape[0]
B_check = 200
boot_rows_1 = np.random.randint(0, row_boot_1, size=(B_check, row_boot_1))
boot_coefs_1 = []
for z in range(B_check):
    boot_data_1 = cleaned_1.iloc[boot_rows_1[z]].copy()
    boot_model_1 = smf.ols("turnout ~ age + C(party)", data=boot_data_1).fit()
    boot_coefs_1.append(boot_model_1.params)
boot_coef_1 = pd.DataFrame(boot_coefs_1).round(10).reset_index(drop=True)

# ---- Run 2 (reset seed) ----
np.random.seed(123)

n2 = 5000
age_2 = np.random.randint(18, 90, size=n2)
party_2 = np.random.choice(["Democrat", "Republican", "Independent"], size=n2, replace=True, p=[0.46, 0.46, 0.08])

party_effect_2 = np.zeros(n2, dtype=float)
party_effect_2[party_2 == "Democrat"] = 0.05
party_effect_2[party_2 == "Republican"] = 0.03
party_effect_2[party_2 == "Independent"] = -0.02

age_effect_2 = (age_2 - 18) / (90 - 18) * 0.35
base_2 = 0.35
p_turnout_2 = base_2 + age_effect_2 + party_effect_2 + np.random.normal(0, 0.05, size=n2)
p_turnout_2 = np.clip(p_turnout_2, 0.01, 0.99)
turnout_2 = (np.random.rand(n2) < p_turnout_2).astype(int)

voter_data_2 = pd.DataFrame({"age": age_2, "party": party_2, "turnout": turnout_2})

cleaned_2 = voter_data_2.copy()
cleaned_2["age"] = pd.to_numeric(cleaned_2["age"], errors="coerce")
cleaned_2["turnout"] = pd.to_numeric(cleaned_2["turnout"], errors="coerce")
cleaned_2["party"] = cleaned_2["party"].astype(str).str.strip()
cleaned_2["party"] = cleaned_2["party"].str.replace(r"\s+", " ", regex=True)
cleaned_2 = cleaned_2.dropna(subset=["age", "party", "turnout"]).reset_index(drop=True)
cleaned_2["age_group"] = pd.cut(cleaned_2["age"], bins=[17, 29, 44, 64, 120], labels=["18-29", "30-44", "45-64", "65+"], include_lowest=True)

turnout_by_party_2 = cleaned_2.groupby("party", as_index=False).agg(turnout_rate=("turnout", "mean"), n=("turnout", "size"))
row_boot_2 = cleaned_2.shape[0]
boot_rows_2 = np.random.randint(0, row_boot_2, size=(B_check, row_boot_2))
boot_coefs_2 = []
for z in range(B_check):
    boot_data_2 = cleaned_2.iloc[boot_rows_2[z]].copy()
    boot_model_2 = smf.ols("turnout ~ age + C(party)", data=boot_data_2).fit()
    boot_coefs_2.append(boot_model_2.params)
boot_coef_2 = pd.DataFrame(boot_coefs_2).round(10).reset_index(drop=True)

# ---- Compare objects (no if/else; just print + log) ----
same_turnout = turnout_by_party_1.round(10).equals(turnout_by_party_2.round(10))
same_bootcoef = boot_coef_1.equals(boot_coef_2)

logging.info(f"Reproducibility check — turnout_by_party identical: {same_turnout}")
logging.info(f"Reproducibility check — boot_coef identical: {same_bootcoef}")

print(pd.DataFrame(
    {
        "turnout_by_party_identical": [same_turnout],
        "bootcoef_identical": [same_bootcoef],
    }
))
