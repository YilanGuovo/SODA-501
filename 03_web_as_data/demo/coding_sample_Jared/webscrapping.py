###############################################################################
# Web Scraping + Google Scholar Tutorial: Python (Penn State Faculty Example)
# Author: Jared Edgerton
# Date: date.today()
#
# This script demonstrates:
#   1) Web scraping a Wikipedia infobox table (warm-up example)
#   2) Web scraping Penn State faculty pages
#   3) Pulling citation metrics from Google Scholar
#   4) Simple plotting with matplotlib
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No conditional statements (no if/else).
# - You will see the same steps repeated for each professor so students can
#   follow the logic and edit one piece at a time.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Install (if needed) and load the necessary libraries.
#
# If you do not have these installed, run (in Terminal / Anaconda Prompt):
#   pip install requests lxml pandas matplotlib scholarly

import re
import requests
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date
from lxml import html
from scholarly import scholarly


# -----------------------------------------------------------------------------
# Part 1: Web Scraping (Wikipedia Warm-up + Penn State Faculty Pages)
# -----------------------------------------------------------------------------
# We will do web scraping in two stages:
#
# A) Wikipedia warm-up (table scraping)
# - Read a Wikipedia page
# - Extract the "infobox" table
# - Clean it into a Key/Value table
#
# B) Penn State faculty pages (text + targeted HTML scraping)
# - Read each faculty member’s PSU profile page
# - Pull the page text and extract:
#     * A title line (regex)
#     * A PSU email address (regex)
# - Pull structured items like:
#     * "Areas of Interest" (XPath)
#     * "Research Interests" (XPath)
#
# Note:
# - Department websites do not always have the same structure.
# - Some pages may have "Areas of Interest" while others have "Research Interests".


# -----------------------------------------------------------------------------
# Part 1A: Wikipedia Warm-up (Scraping an Infobox Table)
# -----------------------------------------------------------------------------
# In Wikipedia, many biography pages include an "infobox" on the right side.
# That infobox is typically stored as an HTML table with class "infobox".

# URL of the Wikipedia page
wiki_url = "https://en.wikipedia.org/wiki/Thomas_Brunell"

# Read all HTML tables that match the "infobox" class
wiki_tables = pd.read_html(wiki_url, attrs={"class": "infobox"})

# Take the first infobox table on the page
wiki_table = wiki_tables[0]

# Give the columns simple names (X1, X2, ...) so we can clean consistently
wiki_table.columns = [f"X{i+1}" for i in range(wiki_table.shape[1])]

# Clean the data:
# - Keep only rows where both X1 and X2 exist (not missing)
# - Rename X1 -> Key and X2 -> Value
cleaned_data = (
    wiki_table
    .dropna(subset=["X1", "X2"])
    .rename(columns={"X1": "Key", "X2": "Value"})[["Key", "Value"]]
    .reset_index(drop=True)
)

# At this point, cleaned_data is a simple Key/Value table.
# You can inspect it:
# print(cleaned_data)


# -----------------------------------------------------------------------------
# Part 1B: Hard-code four Penn State faculty (social sciences broadly)
# -----------------------------------------------------------------------------
# These are the four faculty members we will use throughout the script.
# (We will repeat the same scraping steps for each person.)

matt_name = "Matt Golder"
matt_dept = "Political Science (College of the Liberal Arts)"
matt_url  = "https://polisci.la.psu.edu/people/mrg19/"

sona_name = "Sona N. Golder"
sona_dept = "Political Science (College of the Liberal Arts)"
sona_url  = "https://polisci.la.psu.edu/people/sng11/"

derek_name = "Derek Kreager"
derek_dept = "Sociology & Criminology (College of the Liberal Arts)"
derek_url  = "https://sociology.la.psu.edu/people/derek-kreager/"

jeremy_name = "Jeremy Staff"
jeremy_dept = "Sociology & Criminology (College of the Liberal Arts)"
jeremy_url  = "https://sociology.la.psu.edu/people/jeremy-staff/"

# A basic browser-like header can reduce the chance of being blocked.
headers = {"User-Agent": "Mozilla/5.0 (Teaching Script)"}

# -----------------------------------------------------------------------------
# Step 1: Scrape Matt Golder (one complete example, step-by-step)
# -----------------------------------------------------------------------------
# 1) Request the PSU profile page HTML
matt_html = requests.get(matt_url, headers=headers).text

# 2) Parse the HTML with lxml so we can use XPath (like we did in R)
matt_tree = html.fromstring(matt_html)

# 3) Pull the full page text (useful for regex extraction)
matt_text = " ".join(matt_tree.xpath("//body//text()"))
matt_text = re.sub(r"\s+", " ", matt_text).strip()

# 4) Extract a job title line (regex)
title_pattern = (
    r"(?:Distinguished|Liberal Arts|Roy C\.|Arnold S\.|James P\.)?"
    r"\s*(?:Associate\s+)?Professor[^\n\r]{0,120}"
)
matt_title = " ".join(re.findall(title_pattern, matt_text)[:1]).strip()

# 5) Extract a PSU email address (regex)
email_pattern = r"[A-Za-z0-9._%+-]+@psu\.edu"
matt_email = " ".join(re.findall(email_pattern, matt_text)[:1]).strip()

# 6) Extract "Areas of Interest" (XPath)
matt_areas = matt_tree.xpath(
    "//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li//text()"
)
matt_areas = list(filter(None, map(str.strip, matt_areas)))

# 7) Extract "Research Interests" (XPath)
matt_research = matt_tree.xpath(
    "//h2[normalize-space()='Research Interests']/following-sibling::*[1]//text()"
    " | //h3[normalize-space()='Research Interests']/following-sibling::*[1]//text()"
)
matt_research = list(filter(None, map(str.strip, matt_research)))

# 8) Combine whatever we found into one string (semicolon-separated)
matt_interests_list = matt_areas + matt_research
matt_interests = "; ".join(matt_interests_list)

# 9) Count how many interest items we captured
matt_n_interest_items = len(matt_interests_list)

# 10) Store results as one row (DataFrame)
matt_row = pd.DataFrame([{
    "name": matt_name,
    "department": matt_dept,
    "url": matt_url,
    "scraped_title": matt_title,
    "scraped_email": matt_email,
    "scraped_interests": matt_interests,
    "n_interest_items": matt_n_interest_items
}])


# -----------------------------------------------------------------------------
# Step 2: Scrape Sona N. Golder (repeat the same workflow)
# -----------------------------------------------------------------------------
sona_html = requests.get(sona_url, headers=headers).text
sona_tree = html.fromstring(sona_html)

sona_text = " ".join(sona_tree.xpath("//body//text()"))
sona_text = re.sub(r"\s+", " ", sona_text).strip()

sona_title = " ".join(re.findall(title_pattern, sona_text)[:1]).strip()
sona_email = " ".join(re.findall(email_pattern, sona_text)[:1]).strip()

sona_areas = sona_tree.xpath(
    "//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li//text()"
)
sona_areas = list(filter(None, map(str.strip, sona_areas)))

sona_research = sona_tree.xpath(
    "//h2[normalize-space()='Research Interests']/following-sibling::*[1]//text()"
    " | //h3[normalize-space()='Research Interests']/following-sibling::*[1]//text()"
)
sona_research = list(filter(None, map(str.strip, sona_research)))

sona_interests_list = sona_areas + sona_research
sona_interests = "; ".join(sona_interests_list)
sona_n_interest_items = len(sona_interests_list)

sona_row = pd.DataFrame([{
    "name": sona_name,
    "department": sona_dept,
    "url": sona_url,
    "scraped_title": sona_title,
    "scraped_email": sona_email,
    "scraped_interests": sona_interests,
    "n_interest_items": sona_n_interest_items
}])


# -----------------------------------------------------------------------------
# Step 3: Scrape Derek Kreager (repeat the same workflow)
# -----------------------------------------------------------------------------
derek_html = requests.get(derek_url, headers=headers).text
derek_tree = html.fromstring(derek_html)

derek_text = " ".join(derek_tree.xpath("//body//text()"))
derek_text = re.sub(r"\s+", " ", derek_text).strip()

derek_title = " ".join(re.findall(title_pattern, derek_text)[:1]).strip()
derek_email = " ".join(re.findall(email_pattern, derek_text)[:1]).strip()

derek_areas = derek_tree.xpath(
    "//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li//text()"
)
derek_areas = list(filter(None, map(str.strip, derek_areas)))

derek_research = derek_tree.xpath(
    "//h2[normalize-space()='Research Interests']/following-sibling::*[1]//text()"
    " | //h3[normalize-space()='Research Interests']/following-sibling::*[1]//text()"
)
derek_research = list(filter(None, map(str.strip, derek_research)))

derek_interests_list = derek_areas + derek_research
derek_interests = "; ".join(derek_interests_list)
derek_n_interest_items = len(derek_interests_list)

derek_row = pd.DataFrame([{
    "name": derek_name,
    "department": derek_dept,
    "url": derek_url,
    "scraped_title": derek_title,
    "scraped_email": derek_email,
    "scraped_interests": derek_interests,
    "n_interest_items": derek_n_interest_items
}])


# -----------------------------------------------------------------------------
# Step 4: Scrape Jeremy Staff (repeat the same workflow)
# -----------------------------------------------------------------------------
jeremy_html = requests.get(jeremy_url, headers=headers).text
jeremy_tree = html.fromstring(jeremy_html)

jeremy_text = " ".join(jeremy_tree.xpath("//body//text()"))
jeremy_text = re.sub(r"\s+", " ", jeremy_text).strip()

jeremy_title = " ".join(re.findall(title_pattern, jeremy_text)[:1]).strip()
jeremy_email = " ".join(re.findall(email_pattern, jeremy_text)[:1]).strip()

jeremy_areas = jeremy_tree.xpath(
    "//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li//text()"
)
jeremy_areas = list(filter(None, map(str.strip, jeremy_areas)))

jeremy_research = jeremy_tree.xpath(
    "//h2[normalize-space()='Research Interests']/following-sibling::*[1]//text()"
    " | //h3[normalize-space()='Research Interests']/following-sibling::*[1]//text()"
)
jeremy_research = list(filter(None, map(str.strip, jeremy_research)))

jeremy_interests_list = jeremy_areas + jeremy_research
jeremy_interests = "; ".join(jeremy_interests_list)
jeremy_n_interest_items = len(jeremy_interests_list)

jeremy_row = pd.DataFrame([{
    "name": jeremy_name,
    "department": jeremy_dept,
    "url": jeremy_url,
    "scraped_title": jeremy_title,
    "scraped_email": jeremy_email,
    "scraped_interests": jeremy_interests,
    "n_interest_items": jeremy_n_interest_items
}])


# -----------------------------------------------------------------------------
# Step 5: Combine the scraped rows into one data frame and inspect
# -----------------------------------------------------------------------------
scraped_profiles = pd.concat(
    [matt_row, sona_row, derek_row, jeremy_row],
    ignore_index=True
)

print(scraped_profiles)


# -----------------------------------------------------------------------------
# Step 6: Quick plot (interest items captured per faculty member)
# -----------------------------------------------------------------------------
# This mirrors the bar chart from the R version.

scraped_profiles_sorted = scraped_profiles.sort_values("n_interest_items")

plt.figure()
plt.barh(scraped_profiles_sorted["name"], scraped_profiles_sorted["n_interest_items"])
plt.title("Interest Items Captured from PSU Profile Pages")
plt.xlabel("Number of interest items captured")
plt.ylabel("Faculty member")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# Part 2: Pulling Google Scholar Data (Citations Over Time)
# -----------------------------------------------------------------------------
# Goal:
# - For each professor, we will:
#   (1) Define the Google Scholar ID
#   (2) Pull a profile summary
#   (3) Pull publications (and view the first 5)
#   (4) Pull citation history by year
#   (5) Combine all citation histories into one table and plot them
#
# NOTE:
# - Google Scholar scraping can fail due to bot checks / rate limits.
# - This script intentionally does not include defensive programming so students
#   can see the "main path" clearly.

# -----------------------------------------------------------------------------
# Step 1: Hard-code Google Scholar IDs
# -----------------------------------------------------------------------------
matt_scholar_id   = "yPbxmSwAAAAJ"
sona_scholar_id   = "Cuz1fTcAAAAJ"
derek_scholar_id  = "9c6_ChYAAAAJ"
jeremy_scholar_id = "nm4ZRCgAAAAJ"


# -----------------------------------------------------------------------------
# Step 2: Pull Google Scholar profiles (sequentially)
# -----------------------------------------------------------------------------
matt_author = scholarly.search_author_id(matt_scholar_id)
matt_author = scholarly.fill(matt_author, sections=["basics", "indices", "counts", "publications"])

sona_author = scholarly.search_author_id(sona_scholar_id)
sona_author = scholarly.fill(sona_author, sections=["basics", "indices", "counts", "publications"])

derek_author = scholarly.search_author_id(derek_scholar_id)
derek_author = scholarly.fill(derek_author, sections=["basics", "indices", "counts", "publications"])

jeremy_author = scholarly.search_author_id(jeremy_scholar_id)
jeremy_author = scholarly.fill(jeremy_author, sections=["basics", "indices", "counts", "publications"])


print("\n------------------------------")
print("Google Scholar Profile Summaries")
print("------------------------------")

matt_profile = pd.DataFrame([{
    "name": matt_author.get("name", ""),
    "affiliation": matt_author.get("affiliation", ""),
    "citedby": matt_author.get("citedby", ""),
    "hindex": matt_author.get("hindex", ""),
    "i10index": matt_author.get("i10index", "")
}])
print("\n" + matt_name)
print(matt_profile)

sona_profile = pd.DataFrame([{
    "name": sona_author.get("name", ""),
    "affiliation": sona_author.get("affiliation", ""),
    "citedby": sona_author.get("citedby", ""),
    "hindex": sona_author.get("hindex", ""),
    "i10index": sona_author.get("i10index", "")
}])
print("\n" + sona_name)
print(sona_profile)

derek_profile = pd.DataFrame([{
    "name": derek_author.get("name", ""),
    "affiliation": derek_author.get("affiliation", ""),
    "citedby": derek_author.get("citedby", ""),
    "hindex": derek_author.get("hindex", ""),
    "i10index": derek_author.get("i10index", "")
}])
print("\n" + derek_name)
print(derek_profile)

jeremy_profile = pd.DataFrame([{
    "name": jeremy_author.get("name", ""),
    "affiliation": jeremy_author.get("affiliation", ""),
    "citedby": jeremy_author.get("citedby", ""),
    "hindex": jeremy_author.get("hindex", ""),
    "i10index": jeremy_author.get("i10index", "")
}])
print("\n" + jeremy_name)
print(jeremy_profile)


# -----------------------------------------------------------------------------
# Step 3: Pull Google Scholar publications (sequentially)
# -----------------------------------------------------------------------------
# We will look at the first 5 publications for each professor.
# scholarly stores publications as a list in author["publications"].

print("\n------------------------------")
print("Recent Publications (first 5)")
print("------------------------------")

matt_pubs_df = pd.json_normalize(matt_author["publications"][:5])
print("\n" + matt_name)
print(matt_pubs_df[["bib.title", "bib.year"]].head(5))

sona_pubs_df = pd.json_normalize(sona_author["publications"][:5])
print("\n" + sona_name)
print(sona_pubs_df[["bib.title", "bib.year"]].head(5))

derek_pubs_df = pd.json_normalize(derek_author["publications"][:5])
print("\n" + derek_name)
print(derek_pubs_df[["bib.title", "bib.year"]].head(5))

jeremy_pubs_df = pd.json_normalize(jeremy_author["publications"][:5])
print("\n" + jeremy_name)
print(jeremy_pubs_df[["bib.title", "bib.year"]].head(5))


# -----------------------------------------------------------------------------
# Step 4: Pull citation history (citations by year) and combine
# -----------------------------------------------------------------------------
# scholarly stores citation history in author["cites_per_year"] as a dictionary:
#   {year: citations, year: citations, ...}

matt_ct = pd.DataFrame(list(matt_author["cites_per_year"].items()), columns=["year", "cites"])
matt_ct["name"] = matt_name
matt_ct = matt_ct.sort_values("year")

sona_ct = pd.DataFrame(list(sona_author["cites_per_year"].items()), columns=["year", "cites"])
sona_ct["name"] = sona_name
sona_ct = sona_ct.sort_values("year")

derek_ct = pd.DataFrame(list(derek_author["cites_per_year"].items()), columns=["year", "cites"])
derek_ct["name"] = derek_name
derek_ct = derek_ct.sort_values("year")

jeremy_ct = pd.DataFrame(list(jeremy_author["cites_per_year"].items()), columns=["year", "cites"])
jeremy_ct["name"] = jeremy_name
jeremy_ct = jeremy_ct.sort_values("year")

citation_df = pd.concat([matt_ct, sona_ct, derek_ct, jeremy_ct], ignore_index=True)

print("\nCombined citation data (first 10 rows):")
print(citation_df.head(10))


# -----------------------------------------------------------------------------
# Step 5: Plot citations over time for each professor
# -----------------------------------------------------------------------------
# This mirrors the multi-line ggplot from the R version.

plt.figure()
plt.plot(matt_ct["year"], matt_ct["cites"], marker="o", label=matt_name)
plt.plot(sona_ct["year"], sona_ct["cites"], marker="o", label=sona_name)
plt.plot(derek_ct["year"], derek_ct["cites"], marker="o", label=derek_name)
plt.plot(jeremy_ct["year"], jeremy_ct["cites"], marker="o", label=jeremy_name)

plt.title("Google Scholar Citation History (Recent Years)")
plt.xlabel("Year")
plt.ylabel("Citations")
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# Step 6: Median citations per year for each professor
# -----------------------------------------------------------------------------
median_cites = (
    citation_df
    .groupby("name", as_index=False)["cites"]
    .median()
    .rename(columns={"cites": "median_cites"})
)

print("\nMedian citations per year (by faculty):")
print(median_cites)

