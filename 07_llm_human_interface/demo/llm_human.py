###############################################################################
# LLM-Assisted Data Extraction (Human-in-the-Loop)
# Structured Extraction from Messy Text Using LLMs (Python)
# Author: Jared Edgerton
# Date: date.today()
#
# Coding lab focus:
#   1) Structured extraction from messy text using LLMs
#   2) Prompt design for schemas (what to include + what to avoid)
#   3) Batch processing (loop over many documents)
#   4) Uncertainty checks (confidence, missingness, ambiguity flags)
#   5) Human validation / spot-audits (audit sheets + corrections)
#   6) Documenting failure modes (systematic error categories)
#   7) Evaluation patterns (precision/recall + audit statistics)
#
# Application paper: Gilardi et al. (2023) PNAS
# Application paper: Heseltine & Clemm von Hohenberg (2024) Research & Politics
# Pre-class video: LLM extraction patterns + evaluation (precision/recall, auditing)
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No conditional statements (no if/else).
# - Steps are explicit so students can follow and modify each piece.
#
# Practical note:
# - This tutorial uses OpenAI Structured Outputs with a Pydantic schema.
# - You must set OPENAI_API_KEY in your environment before running:
#     * Windows PowerShell:  setx OPENAI_API_KEY "YOUR_KEY"
#     * macOS/Linux:         export OPENAI_API_KEY="YOUR_KEY"
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# If you do not have these installed, run:
#   pip install openai pydantic pandas numpy scikit-learn
#
# Optional (nice-to-have):
#   pip install python-dateutil

import os
import json
import numpy as np
import pandas as pd

from datetime import date

from pydantic import BaseModel, Field
from typing import List, Literal, Optional

from sklearn.metrics import classification_report

from openai import OpenAI

# Reproducibility (for sampling / splitting / ordering in this script)
np.random.seed(123)

# -----------------------------------------------------------------------------
# Part 1: Define a Schema (What fields do we want?)
# -----------------------------------------------------------------------------
# Goal:
# - Pick a schema that is:
#   (a) useful for the research question
#   (b) specific enough to reduce hallucinations
#   (c) simple enough to annotate reliably
#
# Here we demonstrate an "event extraction" schema from short messy text.

# Event type taxonomy (keep small at first; expand later with a codebook).
EventType = Literal[
    "protest",
    "election",
    "policy_change",
    "violence",
    "disaster",
    "other"
]

# Location granularity (teaching simplification).
GeoPrecision = Literal[
    "country_only",
    "admin1_or_state",
    "city_or_local",
    "unknown"
]

class EvidenceSpan(BaseModel):
    # Short quote(s) from the input that support key fields.
    # Keep evidence short so humans can validate quickly.
    field: Literal[
        "event_type",
        "date",
        "location",
        "actors",
        "outcome"
    ]
    quote: str

class EventExtraction(BaseModel):
    # Core identifiers
    doc_id: str

    # Structured fields
    event_type: EventType
    event_date_iso: Optional[str] = Field(
        default=None,
        description="ISO date YYYY-MM-DD if available; otherwise null."
    )
    date_is_approximate: bool = Field(
        description="True if the date is estimated/inferred (e.g., 'early April')."
    )

    country: Optional[str] = None
    admin1_or_state: Optional[str] = None
    city_or_local: Optional[str] = None
    geo_precision: GeoPrecision

    actors: List[str] = Field(
        description="Key actors mentioned (individuals, orgs, groups)."
    )

    outcome_summary: Optional[str] = Field(
        default=None,
        description="One-sentence outcome summary (what happened)."
    )

    # Uncertainty + auditing fields (human-in-the-loop)
    extraction_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model self-rated confidence (0 to 1)."
    )
    uncertainty_flags: List[str] = Field(
        description="List of issues that make extraction uncertain (e.g., missing date, vague location)."
    )
    evidence: List[EvidenceSpan] = Field(
        description="Short quotes supporting key fields."
    )

# -----------------------------------------------------------------------------
# Part 2: Create Messy Text Inputs (Mini Corpus)
# -----------------------------------------------------------------------------
# In real projects:
# - you read from PDFs, scraped HTML, interview transcripts, etc.
# - you may chunk long documents into sections before extraction
#
# Here we hard-code messy snippets to keep the tutorial self-contained.

docs = [
    {
        "doc_id": "doc_001",
        "text": "Breaking: Thousands rallied in Santiago on 2026-03-14 demanding pension reform. Police reported minor clashes; 12 were arrested."
    },
    {
        "doc_id": "doc_002",
        "text": "On March 2nd, lawmakers passed the 'Clean Air Act' amendment in the national assembly. Environmental groups praised the vote."
    },
    {
        "doc_id": "doc_003",
        "text": "Election officials said voting will take place next Sunday. Turnout is expected to be high in the capital."
    },
    {
        "doc_id": "doc_004",
        "text": "A 6.2 magnitude earthquake struck near the coastal city overnight, damaging dozens of homes and cutting power to 40,000 residents."
    },
    {
        "doc_id": "doc_005",
        "text": "Witnesses described gunfire outside a nightclub late Friday; at least two people were injured, but details remain unclear."
    },
    {
        "doc_id": "doc_006",
        "text": "The governor announced a new curfew order effective immediately. Critics called it an overreach."
    },
    {
        "doc_id": "doc_007",
        "text": "Early April saw renewed demonstrations in the northern province after fuel prices rose again."
    },
    {
        "doc_id": "doc_008",
        "text": "Floodwaters inundated low-lying neighborhoods; emergency shelters opened at local schools, officials said."
    },
    {
        "doc_id": "doc_009",
        "text": "Opposition leaders met with international observers in Brussels to discuss election monitoring."
    },
    {
        "doc_id": "doc_010",
        "text": "Police said the suspect was arrested after a stabbing in downtown; the mayor urged calm."
    },
    {
        "doc_id": "doc_011",
        "text": "Parliament reversed the prior ban on rideshare apps, citing labor market flexibility."
    },
    {
        "doc_id": "doc_012",
        "text": "A protest was planned for tomorrow, but organizers postponed it due to severe weather warnings."
    },
    {
        "doc_id": "doc_013",
        "text": "Following a landslide, the ministry declared a state of emergency in two districts."
    },
    {
        "doc_id": "doc_014",
        "text": "The court ruling sparked demonstrations across the city center; human rights groups condemned the decision."
    },
    {
        "doc_id": "doc_015",
        "text": "The article mentions reforms and elections in passing but gives no clear time or place."
    },
]

docs_df = pd.DataFrame(docs)

print("\n------------------------------")
print("Input corpus (first 5 docs)")
print("------------------------------")
print(docs_df.head())

# -----------------------------------------------------------------------------
# Part 3: Prompt Design (Schemas + Guardrails)
# -----------------------------------------------------------------------------
# Key prompt principles for structured extraction:
# 1) Define the task and scope precisely.
# 2) Define the schema in words (even if the API enforces it).
# 3) Tell the model what to do when information is missing:
#    - use null
#    - set uncertainty_flags
#    - lower confidence
# 4) Require evidence spans (short quotes) so humans can spot-check quickly.
#
# We use Structured Outputs, so the model is constrained to the Pydantic schema,
# but prompts still matter for meaning, missingness, and evidence quality.

system_prompt = (
    "You are an expert research assistant. "
    "Extract a single event record from the user text. "
    "Follow the schema exactly. "
    "If a field is unknown, set it to null (for optional fields) and add an uncertainty flag. "
    "Always provide evidence quotes for event_type, date, location, actors, and outcome."
)

# -----------------------------------------------------------------------------
# Part 4: LLM Structured Extraction (Batch Processing)
# -----------------------------------------------------------------------------
# We call the LLM once per document (simple and explicit).
# In real projects:
# - you may batch requests
# - you may chunk long documents and aggregate
#
# This example uses OpenAI Structured Outputs parsing.

client = OpenAI()

model_name = "gpt-4o-2024-08-06"

extractions = []

print("\n------------------------------")
print("Running LLM extraction (one doc at a time)")
print("------------------------------")

for i in range(len(docs_df)):
    doc_id = docs_df.loc[i, "doc_id"]
    text = docs_df.loc[i, "text"]

    user_prompt = (
        f"Document ID: {doc_id}\n\n"
        f"Text:\n{text}\n\n"
        "Return exactly one extracted record."
    )

    response = client.responses.parse(
        model=model_name,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=EventExtraction,
    )

    extracted = response.output_parsed
    extra_dict = extracted.model_dump()
    extra_dict["raw_text"] = text

    extra_dict["evidence_json"] = json.dumps(extra_dict["evidence"], ensure_ascii=False)
    extra_dict["uncertainty_flags_json"] = json.dumps(extra_dict["uncertainty_flags"], ensure_ascii=False)

    extra_dict.pop("evidence")
    extra_dict.pop("uncertainty_flags")

    extractions.append(extra_dict)

extractions_df = pd.DataFrame(extractions)

print("\n------------------------------")
print("Extracted records (first 5 rows)")
print("------------------------------")
print(extractions_df.head())

# Save raw extractions (for traceability)
os.makedirs("outputs", exist_ok=True)
extractions_df.to_csv("outputs/extractions_raw.csv", index=False)

# -----------------------------------------------------------------------------
# Part 5: Uncertainty Checks (Automatic Flags for Human Review)
# -----------------------------------------------------------------------------
# Human-in-the-loop pattern:
# - Use the model's own uncertainty fields
# - Add mechanical checks (missing date, missing country, etc.)
# - Use these checks to decide what gets spot-audited

# Convert confidence to numeric (safety)
extractions_df["extraction_confidence"] = pd.to_numeric(extractions_df["extraction_confidence"], errors="coerce")

# Simple mechanical checks (vectorized; no if/else)
extractions_df["flag_low_confidence"] = extractions_df["extraction_confidence"] < 0.70
extractions_df["flag_missing_date"] = extractions_df["event_date_iso"].isna()
extractions_df["flag_missing_country"] = extractions_df["country"].isna()
extractions_df["flag_geo_unknown"] = extractions_df["geo_precision"].isin(["unknown", "country_only"])

# If any flags are true, mark for review
flag_cols = ["flag_low_confidence", "flag_missing_date", "flag_missing_country", "flag_geo_unknown"]
extractions_df["needs_human_review"] = extractions_df[flag_cols].any(axis=1)

print("\n------------------------------")
print("Review flag counts")
print("------------------------------")
print(extractions_df[flag_cols + ["needs_human_review"]].sum(numeric_only=True))

extractions_df.to_csv("outputs/extractions_with_flags.csv", index=False)

# -----------------------------------------------------------------------------
# Part 6: Human Validation / Spot-Audits (Create an Audit Sheet)
# -----------------------------------------------------------------------------
# Pattern:
# - Always audit a random sample (coverage)
# - Always audit any cases flagged by uncertainty checks (risk-based auditing)
#
# We create a single CSV for humans to review:
# - includes extracted fields + raw_text + evidence
# - includes blank columns for human corrections / notes

audit_random_n = 5
audit_random = extractions_df.sample(n=audit_random_n, random_state=123)

audit_flagged = extractions_df[extractions_df["needs_human_review"]].copy()

audit_sheet = pd.concat([audit_random, audit_flagged], ignore_index=True).drop_duplicates(subset=["doc_id"])
audit_sheet = audit_sheet.sort_values("doc_id").reset_index(drop=True)

audit_sheet["human_is_correct"] = ""          # fill with 1/0
audit_sheet["human_correct_event_type"] = ""  # optional correction
audit_sheet["human_correct_date_iso"] = ""    # optional correction
audit_sheet["human_correct_location"] = ""    # optional correction (free text)
audit_sheet["failure_mode"] = ""              # e.g., date_missing, location_vague, actor_hallucination
audit_sheet["reviewer_notes"] = ""            # free text

audit_sheet.to_csv("outputs/human_audit_sheet.csv", index=False)

print("\n------------------------------")
print("Wrote outputs/human_audit_sheet.csv")
print("------------------------------")

# -----------------------------------------------------------------------------
# Part 7: Evaluation Patterns (Precision/Recall + Auditing)
# -----------------------------------------------------------------------------
# In real projects, you build a gold-standard labeled dataset for evaluation.
# Here we hard-code a tiny "gold" set for demonstration.
#
# Students should compare:
# - model event_type vs gold event_type
# - and then compute precision/recall by class

gold = pd.DataFrame([
    {"doc_id": "doc_001", "event_type_gold": "protest"},
    {"doc_id": "doc_002", "event_type_gold": "policy_change"},
    {"doc_id": "doc_003", "event_type_gold": "election"},
    {"doc_id": "doc_004", "event_type_gold": "disaster"},
    {"doc_id": "doc_005", "event_type_gold": "violence"},
    {"doc_id": "doc_006", "event_type_gold": "policy_change"},
    {"doc_id": "doc_007", "event_type_gold": "protest"},
    {"doc_id": "doc_008", "event_type_gold": "disaster"},
])

eval_df = gold.merge(extractions_df[["doc_id", "event_type"]], on="doc_id", how="left")
eval_df = eval_df.rename(columns={"event_type": "event_type_pred"})

print("\n------------------------------")
print("Evaluation table (gold vs predicted)")
print("------------------------------")
print(eval_df)

print("\n------------------------------")
print("Classification report (event_type)")
print("------------------------------")
print(classification_report(eval_df["event_type_gold"], eval_df["event_type_pred"]))
