# make_week12_slides.R
# Generates Week 12 lecture slides (LLM-Assisted Data Extraction: Human-in-the-Loop)

# Install required packages if not already installed
if (!requireNamespace("xaringan", quietly = TRUE)) install.packages("xaringan")
if (!requireNamespace("rmarkdown", quietly = TRUE)) install.packages("rmarkdown")
if (!requireNamespace("pagedown", quietly = TRUE)) install.packages("pagedown")

# Write the R Markdownp source for the slides
writeLines(
"---
title: 'LLM-Assisted Data Extraction'
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

# Why Use LLMs for Data Extraction

Many social-science data sources are:
- Messy
- Unstructured
- Text-heavy
- Expensive to code by hand

LLMs offer scalable *assistance*, not automation.

---

# Human-in-the-Loop Philosophy

In this course, LLMs are used to:
- Propose structured outputs
- Accelerate labeling
- Surface uncertainty

Humans remain responsible for validation.

---

# Structured Extraction

Structured extraction means:
- Defining a schema in advance
- Forcing outputs into fields
- Rejecting free-form text

Schemas discipline model behavior.

---

# Schema Example

````python
schema = {
    'actor': 'string',
    'action': 'string',
    'target': 'string',
    'date': 'YYYY-MM-DD',
    'confidence': 'float'
}
````

Explicit schemas reduce ambiguity.

---

# Prompt Design for Extraction

Effective prompts:
- Describe the task narrowly
- Specify output format
- Include examples when possible

Prompting is part of the method.

---

# Prompt Example

````python
prompt = '
Extract the following fields from the text below.
Return JSON matching this schema:
{schema}

Text:
{text}
'
````

---

# Batch Processing

LLMs are typically applied:
- In batches
- With rate limits
- With cost constraints

Pipelines must manage scale.

---

# Batch Processing Pattern

````python
results = []
for doc in documents:
    response = call_llm(prompt_template, doc)
    results.append(response)
````

Batch size affects cost and reliability.

---

# Uncertainty and Confidence

LLMs can:
- Provide confidence scores
- Flag ambiguous cases
- Abstain when uncertain

Uncertainty should be captured explicitly.

---

# Uncertainty Example

````python
if response['confidence'] < 0.6:
    flag_for_review(response)
````

Low-confidence cases are routed to humans.

---

# Human Validation

Human review is used to:
- Spot-check outputs
- Correct systematic errors
- Refine prompts and schemas

Validation is iterative.

---

# Spot Audits

Auditing strategies include:
- Random sampling
- Stratified checks
- Edge-case review

Audits reveal failure modes.

---

# Failure Modes

Common failures include:
- Hallucinated fields
- Overconfident errors
- Inconsistent formatting
- Sensitivity to phrasing

Failures must be documented.

---

# Evaluation Metrics

Extraction quality can be assessed with:
- Precision / recall
- Field-level accuracy
- Agreement with human labels

Evaluation depends on task goals.

---

# LLMs Are Not Neutral

LLM outputs reflect:
- Training data biases
- Prompt framing
- Model defaults

Human oversight is essential.

---

# Documentation Requirements

Every LLM pipeline should record:
- Model and version
- Prompt text
- Schema definition
- Validation procedure
- Known limitations

Transparency enables reuse.

---

# What We Emphasize in Practice

- Use LLMs as assistants
- Enforce structure aggressively
- Validate with humans
- Document failure modes

---

# Discussion

- Where do LLMs help most?
- When do they fail quietly?
- How much validation is enough?
",
  "week12_slides.Rmd"
)

# Render the R Markdown file
rmarkdown::render(
  "week12_slides.Rmd",
  output_format = "xaringan::moon_reader"
)

# Convert HTML to PDF
pagedown::chrome_print(
  "week12_slides.html",
  output = "week_12_slides.pdf"
)

# Clean up temporary files
file.remove("week12_slides.Rmd", "week12_slides.html")

cat("PDF slides have been created as 'week_12_slides.pdf'\n")
