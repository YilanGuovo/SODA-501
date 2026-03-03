###############################################################################
# LLM-Assisted Data Extraction (Human-in-the-Loop)
# FREE / LOCAL VERSION (Hugging Face)
# Structured Extraction from Messy Text Using LOCAL LLMs (Python)
# Author: Jared Edgerton
# Date: date.today()
#
# Teaching version:
# - Explicit, sequential steps
# - Avoids high-level pipeline() wrapper
#
# Install (recommend pinning for class):
#   pip install -U "transformers>=4.40.0" "torch" "sentencepiece" "pydantic>=2.0" pandas numpy scikit-learn
###############################################################################

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

# %%
import os
import json
import re

import numpy as np
import pandas as pd

import torch

from datetime import date
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from sklearn.metrics import classification_report

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed

np.random.seed(123)
set_seed(123)

# ---------------------------------------------------------------------
# Part 1: Define a Schema (What fields do we want?)
# ---------------------------------------------------------------------
EventType = Literal[
    "protest",
    "election",
    "policy_change",
    "violence",
    "disaster",
    "other"
]

GeoPrecision = Literal[
    "country_only",
    "admin1_or_state",
    "city_or_local",
    "unknown"
]

class EvidenceSpan(BaseModel):
    field: Literal["event_type", "date", "location", "actors", "outcome"]
    quote: str

class EventExtraction(BaseModel):
    doc_id: str

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

    actors: List[str] = Field(description="Key actors mentioned (individuals, orgs, groups).")

    outcome_summary: Optional[str] = Field(
        default=None,
        description="One-sentence outcome summary (what happened)."
    )

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

# ---------------------------------------------------------------------
# Part 2: Create Messy Text Inputs (Mini Corpus)
# ---------------------------------------------------------------------
docs = [
    {"doc_id": "doc_001", "text": "Breaking: Thousands rallied in Santiago on 2026-03-14 demanding pension reform. Police reported minor clashes; 12 were arrested."},
    {"doc_id": "doc_002", "text": "On March 2nd, lawmakers passed the 'Clean Air Act' amendment in the national assembly. Environmental groups praised the vote."},
    {"doc_id": "doc_003", "text": "Election officials said voting will take place next Sunday. Turnout is expected to be high in the capital."},
    {"doc_id": "doc_004", "text": "A 6.2 magnitude earthquake struck near the coastal city overnight, damaging dozens of homes and cutting power to 40,000 residents."},
    {"doc_id": "doc_005", "text": "Witnesses described gunfire outside a nightclub late Friday; at least two people were injured, but details remain unclear."},
    {"doc_id": "doc_006", "text": "The governor announced a new curfew order effective immediately. Critics called it an overreach."},
    {"doc_id": "doc_007", "text": "Early April saw renewed demonstrations in the northern province after fuel prices rose again."},
    {"doc_id": "doc_008", "text": "Floodwaters inundated low-lying neighborhoods; emergency shelters opened at local schools, officials said."},
    {"doc_id": "doc_009", "text": "Opposition leaders met with international observers in Brussels to discuss election monitoring."},
    {"doc_id": "doc_010", "text": "Police said the suspect was arrested after a stabbing in downtown; the mayor urged calm."},
    {"doc_id": "doc_011", "text": "Parliament reversed the prior ban on rideshare apps, citing labor market flexibility."},
    {"doc_id": "doc_012", "text": "A protest was planned for tomorrow, but organizers postponed it due to severe weather warnings."},
    {"doc_id": "doc_013", "text": "Following a landslide, the ministry declared a state of emergency in two districts."},
    {"doc_id": "doc_014", "text": "The court ruling sparked demonstrations across the city center; human rights groups condemned the decision."},
    {"doc_id": "doc_015", "text": "The article mentions reforms and elections in passing but gives no clear time or place."},
]

docs_df = pd.DataFrame(docs)

print("\n------------------------------")
print("Input corpus (first 5 docs)")
print("------------------------------")
print(docs_df.head())
print("docs_df shape:", docs_df.shape)

# ---------------------------------------------------------------------
# Part 3: Prompt Design (Schemas + Guardrails)
# ---------------------------------------------------------------------
json_template = {
    "doc_id": "doc_XXX",
    "event_type": "other",
    "event_date_iso": None,
    "date_is_approximate": False,
    "country": None,
    "admin1_or_state": None,
    "city_or_local": None,
    "geo_precision": "unknown",
    "actors": [],
    "outcome_summary": None,
    "extraction_confidence": 0.5,
    "uncertainty_flags": [],
    "evidence": [
        {"field": "event_type", "quote": ""},
        {"field": "date", "quote": ""},
        {"field": "location", "quote": ""},
        {"field": "actors", "quote": ""},
        {"field": "outcome", "quote": ""}
    ]
}

system_instructions = (
    "Task: Extract ONE event record from the text.\n"
    "Output MUST be valid JSON only (no markdown, no extra text).\n"
    "Allowed event_type: protest, election, policy_change, violence, disaster, other.\n"
    "Allowed geo_precision: country_only, admin1_or_state, city_or_local, unknown.\n"
    "If unknown: use null for optional fields, add an uncertainty flag, and lower extraction_confidence.\n"
    "Evidence quotes must be short substrings copied from the text.\n"
)

# ---------------------------------------------------------------------
# Part 4: Local LLM Structured Extraction (Batch Processing)
# ---------------------------------------------------------------------
# Model choice: small, free, runs on CPU (slow but fine for class demos)
model_name = "google/flan-t5-small"

print("\n------------------------------")
print("Loading tokenizer + model")
print("------------------------------")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda") if use_gpu else torch.device("cpu")
model = model.to(device)

print("Model:", model_name)
print("Device:", device)

extractions = []

print("\n------------------------------")
print("Running LOCAL LLM extraction (one doc at a time)")
print("------------------------------")

for i in range(len(docs_df)):
    doc_id = docs_df.loc[i, "doc_id"]
    text = docs_df.loc[i, "text"]

    prompt = (
        f"{system_instructions}\n"
        f"JSON template:\n{json.dumps(json_template, ensure_ascii=False)}\n\n"
        f"Document ID: {doc_id}\n"
        f"Text: {text}\n\n"
        "Return JSON only."
    )

    # 1) Tokenize (explicit)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 2) Generate (explicit)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False
        )

    # 3) Decode (explicit)
    out_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # 4) Recover JSON substring (explicit)
    match = re.search(r"\{.*\}", out_text, flags=re.DOTALL)
    json_str = match.group(0) if match else out_text

    # 5) Validate with Pydantic (explicit)
    # Pydantic v2:
    parse_ok = True
    parse_error = ""
    try:
        extracted_obj = EventExtraction.model_validate_json(json_str)
        extra_dict = extracted_obj.model_dump()
    except Exception as e:
        parse_ok = False
        parse_error = str(e)

        # Fallback record (keeps pipeline running)
        extra_dict = {
            "doc_id": doc_id,
            "event_type": "other",
            "event_date_iso": None,
            "date_is_approximate": False,
            "country": None,
            "admin1_or_state": None,
            "city_or_local": None,
            "geo_precision": "unknown",
            "actors": [],
            "outcome_summary": None,
            "extraction_confidence": 0.0,
            "uncertainty_flags": ["parse_failed_local_model_output"],
            "evidence": [
                {"field": "event_type", "quote": ""},
                {"field": "date", "quote": ""},
                {"field": "location", "quote": ""},
                {"field": "actors", "quote": ""},
                {"field": "outcome", "quote": ""}
            ]
        }

    # 6) Attach trace fields (explicit)
    extra_dict["raw_text"] = text
    extra_dict["local_model_raw_output"] = out_text
    extra_dict["parse_ok"] = parse_ok
    extra_dict["parse_error"] = parse_error

    # 7) Flatten list fields for CSV (explicit)
    extra_dict["evidence_json"] = json.dumps(extra_dict["evidence"], ensure_ascii=False)
    extra_dict["uncertainty_flags_json"] = json.dumps(extra_dict["uncertainty_flags"], ensure_ascii=False)
    extra_dict.pop("evidence")
    extra_dict.pop("uncertainty_flags")

    extractions.append(extra_dict)

# 8) Build dataframe + save
extractions_df = pd.DataFrame(extractions)

print("\n------------------------------")
print("Extracted records (first 5 rows)")
print("------------------------------")
print(extractions_df.head())
print("extractions_df shape:", extractions_df.shape)

os.makedirs("outputs", exist_ok=True)
extractions_df.to_csv("outputs/extractions_raw.csv", index=False)

# ---------------------------------------------------------------------
# Part 5: Uncertainty Checks (Automatic Flags for Human Review)
# ---------------------------------------------------------------------
extractions_df["extraction_confidence"] = pd.to_numeric(extractions_df["extraction_confidence"], errors="coerce")

extractions_df["flag_parse_failed"] = ~extractions_df["parse_ok"]
extractions_df["flag_low_confidence"] = extractions_df["extraction_confidence"] < 0.70
extractions_df["flag_missing_date"] = extractions_df["event_date_iso"].isna()
extractions_df["flag_missing_country"] = extractions_df["country"].isna()
extractions_df["flag_geo_unknown"] = extractions_df["geo_precision"].isin(["unknown", "country_only"])

flag_cols = [
    "flag_parse_failed",
    "flag_low_confidence",
    "flag_missing_date",
    "flag_missing_country",
    "flag_geo_unknown"
]
extractions_df["needs_human_review"] = extractions_df[flag_cols].any(axis=1)

print("\n------------------------------")
print("Review flag counts")
print("------------------------------")
print(extractions_df[flag_cols + ["needs_human_review"]].sum(numeric_only=True))

extractions_df.to_csv("outputs/extractions_with_flags.csv", index=False)

# ---------------------------------------------------------------------
# Part 6: Human Validation / Spot-Audits (Create an Audit Sheet)
# ---------------------------------------------------------------------
audit_random_n = 5
audit_random = extractions_df.sample(n=audit_random_n, random_state=123)

audit_flagged = extractions_df[extractions_df["needs_human_review"]].copy()

audit_sheet = pd.concat([audit_random, audit_flagged], ignore_index=True).drop_duplicates(subset=["doc_id"])
audit_sheet = audit_sheet.sort_values("doc_id").reset_index(drop=True)

audit_sheet["human_is_correct"] = ""
audit_sheet["human_correct_event_type"] = ""
audit_sheet["human_correct_date_iso"] = ""
audit_sheet["human_correct_location"] = ""
audit_sheet["failure_mode"] = ""
audit_sheet["reviewer_notes"] = ""

audit_sheet.to_csv("outputs/human_audit_sheet.csv", index=False)

print("\n------------------------------")
print("Wrote outputs/human_audit_sheet.csv")
print("------------------------------")

# ---------------------------------------------------------------------
# Part 7: Evaluation Patterns (Precision/Recall + Auditing)
# ---------------------------------------------------------------------
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
eval_df["event_type_pred"] = eval_df["event_type_pred"].fillna("MISSING")

print("\n------------------------------")
print("Evaluation table (gold vs predicted)")
print("------------------------------")
print(eval_df)

print("\n------------------------------")
print("Classification report (event_type)")
print("------------------------------")
print(classification_report(eval_df["event_type_gold"], eval_df["event_type_pred"], zero_division=0))