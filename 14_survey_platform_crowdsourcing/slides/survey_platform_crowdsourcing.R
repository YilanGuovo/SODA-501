# make_week04_slides.R
# Generates Week 04 lecture slides (Surveys, Platforms, and Crowdsourcing)

# Install required packages if not already installed
if (!requireNamespace("xaringan", quietly = TRUE)) install.packages("xaringan")
if (!requireNamespace("rmarkdown", quietly = TRUE)) install.packages("rmarkdown")
if (!requireNamespace("pagedown", quietly = TRUE)) install.packages("pagedown")

# Write the R Markdown source for the slides
writeLines(
"---
title: 'Surveys, Platforms, and Crowdsourcing'
author: 'Jared Edgerton'
output:
  xaringan::moon_reader:
    css: [default, metropolis, metropolis-fonts]
    nature:
      highlightStyle: github
      highlightLines: true
      countIncrementalSlides: false
      slideNumberFormat: ''
---

# Why Surveys Still Matter

Surveys remain a central tool for:
- Measuring attitudes and beliefs
- Capturing latent preferences
- Studying populations at scale

But modern surveys are rarely simple random samples.

---

# The Shift to Platforms

Much contemporary survey data are collected via:
- Online labor markets
- Survey platforms
- Panel providers
- Hybrid recruitment strategies

Platforms reshape who answers and how.

---

# Data as a Measurement Process

Survey data reflect:
- Question wording
- Response options
- Ordering and framing
- Mode of administration

Measurement decisions precede analysis.

---

# From Instrument to Dataset

Surveys arrive as:
- Exported files (CSV, TSV, SPSS)
- Codebooks or questionnaires
- Metadata files
- Platform-generated diagnostics

Understanding structure comes first.

---

# Reading Survey Exports (Python)

````python
import pandas as pd

df = pd.read_csv('survey_export.csv')
df.head()
````

The raw export is not analysis-ready data.

---

# Reading Survey Exports (R)

````r
library(readr)

df <- read_csv('survey_export.csv')
head(df)
````

Always inspect before cleaning.

---

# Codebooks Are Data

Codebooks specify:
- Variable meanings
- Scales and categories
- Missing value codes
- Skip logic

Ignoring codebooks leads to silent errors.

---

# Labeling and Recoding

Cleaning often requires:
- Relabeling categories
- Harmonizing scales
- Converting strings to factors
- Explicit missing-value handling

---

# Recoding Example (Python)

````python
df['gender'] = df['gender'].map({
    1: 'Male',
    2: 'Female',
    3: 'Other'
})
````

---

# Recoding Example (R)

````r
df$gender <- recode(
  df$gender,
  '1' = 'Male',
  '2' = 'Female',
  '3' = 'Other'
)
````

---

# Crowdsourcing and Human Subjects

Crowdsourced data involve:
- Heterogeneous respondents
- Variable attention and effort
- Strategic responding

Quality is uneven and must be assessed.

---

# Quality Checks Are Essential

Common checks include:
- Completion time thresholds
- Attention checks
- Straight-lining detection
- Response consistency

These are measurement decisions.

---

# Simple Quality Check (Python)

````python
df = df[df['duration_seconds'] > 60]
````

---

# Simple Quality Check (R)

````r
df <- df |> dplyr::filter(duration_seconds > 60)
````

---

# Missingness Is Informative

Missing data may reflect:
- Survey design
- Respondent fatigue
- Sensitivity of questions
- Platform defaults

Missingness is rarely random.

---

# Platforms Shape Data

Platforms influence:
- Who participates
- Incentives and compensation
- Sampling frames
- Longitudinal availability

Platform choice is a design choice.

---

# Documentation Is Non-Negotiable

Every survey pipeline should record:
- Recruitment method
- Platform used
- Field dates
- Exclusion criteria
- Known limitations

Survey data without documentation are not reusable.

---

# What We Emphasize in Practice

- Treat surveys as constructed instruments
- Inspect raw exports carefully
- Make recoding explicit
- Document quality filters

---

# Discussion

- Where do surveys introduce bias?
- How do platforms shape who responds?
- Which quality checks feel defensible?
",
  "week04_slides.Rmd"
)

# Render the R Markdown file
rmarkdown::render(
  "week04_slides.Rmd",
  output_format = "xaringan::moon_reader"
)

# Convert HTML to PDF
pagedown::chrome_print(
  "week04_slides.html",
  output = "week_04_slides.pdf"
)

# Clean up temporary files
file.remove("week04_slides.Rmd", "week04_slides.html")

cat("PDF slides have been created as 'week_04_slides.pdf'\n")
