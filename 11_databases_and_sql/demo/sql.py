###############################################################################
# SQL Tutorial in Python: SQLite + Campaign Finance (Simulated Data)
# Date: date.today()
#
# This script demonstrates:
#   1) Creating a local SQLite database (tables + indexes)
#   2) Inserting simulated campaign finance data
#   3) Writing and running SQL queries from Python
#   4) Visualizing query outputs with matplotlib
###############################################################################

# %%
import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date, timedelta

os.chdir(r"D:\Yilan\1\Courses\SODA\501\soda_501\11_databases_and_sql\demo")
print(os.getcwd())

os.makedirs("database", exist_ok=True)
os.makedirs("figures", exist_ok=True)


# -----------------------------------------------------------------------------
# Part 1: Create and Populate a Local SQLite Database
# -----------------------------------------------------------------------------

# Step 1: Connect to a database file
con = sqlite3.connect("database\campaign_finance.db")
cur = con.cursor()

# Step 2: Drop tables (so the script can be rerun from scratch)
cur.execute("DROP TABLE IF EXISTS contributions;")
cur.execute("DROP TABLE IF EXISTS contributors;")
cur.execute("DROP TABLE IF EXISTS candidates;")
con.commit()

# Step 3: Create tables
cur.execute("""
  CREATE TABLE candidates (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    party TEXT,
    office TEXT,
    winner INTEGER
  );
""")

cur.execute("""
  CREATE TABLE contributors (
    id INTEGER PRIMARY KEY,
    name TEXT,
    occupation TEXT,
    employer TEXT,
    state TEXT
  );
""")

cur.execute("""
  CREATE TABLE contributions (
    id INTEGER PRIMARY KEY,
    contributor_id INTEGER,
    candidate_id INTEGER,
    amount REAL,
    date TEXT,
    FOREIGN KEY (contributor_id) REFERENCES contributors(id),
    FOREIGN KEY (candidate_id) REFERENCES candidates(id)
  );
""")
con.commit()

# -----------------------------------------------------------------------------
# Step 4: Generate simulated data
# -----------------------------------------------------------------------------
np.random.seed(123)

# ---- Candidates table (100 candidates) ----
candidate_ids = np.arange(1, 101)

candidate_names = np.array([f"Candidate {i}" for i in candidate_ids])

candidate_parties = np.random.choice(
    ["Democrat", "Republican", "Independent"],
    size=100,
    replace=True,
    p=[0.45, 0.45, 0.10]
)

candidate_offices = np.random.choice(
    ["Senate", "House", "Governor", "State Senate", "State House"],
    size=100,
    replace=True
)

candidate_winner = np.random.choice(
    [1, 0],
    size=100,
    replace=True,
    p=[0.5, 0.5]
)

candidates = pd.DataFrame({
    "id": candidate_ids,
    "name": candidate_names,
    "party": candidate_parties,
    "office": candidate_offices,
    "winner": candidate_winner
})

# ---- Contributors table (100,000 contributors) ----
contributor_ids = np.arange(1, 100001)

contributor_names = np.array([f"Contributor {i}" for i in contributor_ids])

contributor_occupations = np.random.choice(
    ["Engineer", "Teacher", "Doctor", "Lawyer", "Business Owner"],
    size=100000,
    replace=True
)

contributor_employers = np.array(
    [f"Company {i}" for i in np.random.randint(1, 5001, size=100000)]
)

state_abb = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
]

contributor_states = np.random.choice(state_abb, size=100000, replace=True)

contributors = pd.DataFrame({
    "id": contributor_ids,
    "name": contributor_names,
    "occupation": contributor_occupations,
    "employer": contributor_employers,
    "state": contributor_states
})

# ---- Contributions table (1,000,000 contributions) ----
contribution_ids = np.arange(1, 1000001)

contribution_contributor_ids = np.random.randint(1, 100001, size=1000000)
contribution_candidate_ids = np.random.randint(1, 101, size=1000000)

contribution_amounts = np.round(
    np.random.lognormal(mean=np.log(1000), sigma=1, size=1000000),
    2
)

start_date = date(2024, 1, 1)
end_date = date(2024, 12, 31)
n_days = (end_date - start_date).days + 1

random_day_offsets = np.random.randint(0, n_days, size=1000000)
contribution_dates = np.array([
    (start_date + timedelta(days=int(d))).isoformat()
    for d in random_day_offsets
])

contributions = pd.DataFrame({
    "id": contribution_ids,
    "contributor_id": contribution_contributor_ids,
    "candidate_id": contribution_candidate_ids,
    "amount": contribution_amounts,
    "date": contribution_dates
})

# -----------------------------------------------------------------------------
# Step 5: Insert data into the database
# -----------------------------------------------------------------------------
candidates.to_sql("candidates", con, if_exists="append", index=False, chunksize=5000)
contributors.to_sql("contributors", con, if_exists="append", index=False, chunksize=5000)
contributions.to_sql("contributions", con, if_exists="append", index=False, chunksize=5000)
con.commit()

# -----------------------------------------------------------------------------
# Step 6: Create indexes
# -----------------------------------------------------------------------------
cur.execute("CREATE INDEX IF NOT EXISTS idx_contrib_contributor_id ON contributions (contributor_id);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_contrib_candidate_id   ON contributions (candidate_id);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_contrib_amount         ON contributions (amount);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_contrib_date           ON contributions (date);")
con.commit()

# -----------------------------------------------------------------------------
# Part 3: Required outputs for Question 3
# -----------------------------------------------------------------------------
print("\n==============================")
print("QUESTION 3: DATABASE + SCHEMA")
print("==============================")

# Row counts using SELECT COUNT(*)
count_candidates = pd.read_sql_query(
    "SELECT COUNT(*) AS n_rows FROM candidates;", con
)
count_contributors = pd.read_sql_query(
    "SELECT COUNT(*) AS n_rows FROM contributors;", con
)
count_contributions = pd.read_sql_query(
    "SELECT COUNT(*) AS n_rows FROM contributions;", con
)

print("\nRow count: candidates")
print(count_candidates)

print("\nRow count: contributors")
print(count_contributors)

print("\nRow count: contributions")
print(count_contributions)

# Show schema for each table
schema_candidates = pd.read_sql_query("PRAGMA table_info(candidates);", con)
schema_contributors = pd.read_sql_query("PRAGMA table_info(contributors);", con)
schema_contributions = pd.read_sql_query("PRAGMA table_info(contributions);", con)

print("\nSchema: candidates")
print(schema_candidates)

print("\nSchema: contributors")
print(schema_contributors)

print("\nSchema: contributions")
print(schema_contributions)

# Brief explanation of keys connecting the tables
print("\nHow the keys connect the tables:")
print(
    "The contributions table is the linking table between contributors and candidates. "
    "Its contributor_id matches contributors.id, so each contribution can be tied to the "
    "person who donated. Its candidate_id matches candidates.id, so each contribution can "
    "also be tied to the candidate who received the money. Together, these two foreign keys "
    "let us join donation records to both donor information and candidate information."
)

# -----------------------------------------------------------------------------
# Part 4: Required join + aggregation for Question 4
# -----------------------------------------------------------------------------
print("\n======================================")
print("QUESTION 4: JOIN + AGGREGATION + PLOT")
print("======================================")

# Required query:
# - join contributions to candidates
# - restrict to amount > 1000
# - output party, total_amount, num_contributions
query_q4 = """
  SELECT
    ca.party AS party,
    SUM(co.amount) AS total_amount,
    COUNT(*) AS num_contributions
  FROM contributions co
  JOIN candidates ca
    ON co.candidate_id = ca.id
  WHERE co.amount > 1000
  GROUP BY ca.party
  ORDER BY total_amount DESC;
"""

party_summary = pd.read_sql_query(query_q4, con)

print("\nSQL for Question 4:")
print(query_q4)

print("Output table:")
print(party_summary)

# Visualization: bar plot of total amount by party
plt.figure()
plt.bar(party_summary["party"], party_summary["total_amount"])
plt.title("Total Contributions by Party (Amount > 1000)")
plt.xlabel("Party")
plt.ylabel("Total Amount")
plt.tight_layout()
plt.savefig("figures\contributions_by_party.png", dpi=150)
plt.show()

print("\nSaved plot: contributions_by_party.png")

# -----------------------------------------------------------------------------
# Part 5: Indexes + query plan for Question 5
# -----------------------------------------------------------------------------
print("\n===================================")
print("QUESTION 5: INDEXES + QUERY PLAN")
print("===================================")

# Verify which indexes exist on contributions
indexes_contributions = pd.read_sql_query("""
  SELECT
    name,
    sql
  FROM sqlite_master
  WHERE type = 'index'
    AND tbl_name = 'contributions';
""", con)

print("\nIndexes on contributions:")
print(indexes_contributions)

# Choose one query that filters by candidate_id
filter_query = """
  SELECT *
  FROM contributions
  WHERE candidate_id = 10;
"""

query_plan = pd.read_sql_query(
    "EXPLAIN QUERY PLAN " + filter_query,
    con
)

print("\nFiltered query:")
print(filter_query)

print("EXPLAIN QUERY PLAN output:")
print(query_plan)

print("\nInterpretation of the query plan:")
print(
    "This query filters the contributions table on candidate_id, which is one of the columns "
    "we indexed. If SQLite reports SEARCH contributions USING INDEX idx_contrib_candidate_id, "
    "that means it is using the candidate_id index rather than scanning the full table. "
    "Using an index is helpful here because the contributions table is very large, so indexed "
    "lookup should reduce the amount of data SQLite must examine. If the plan instead showed "
    "a full table scan, then an index on candidate_id would be especially useful for speeding "
    "up repeated filters and joins on that column. In general, indexes help most when queries "
    "frequently restrict rows on the indexed field."
)

# -----------------------------------------------------------------------------
# Optional: keep a couple of your original examples if you want
# -----------------------------------------------------------------------------
print("\n------------------------------")
print("Sample joined rows")
print("------------------------------")
print(pd.read_sql_query("""
  SELECT
    co.id,
    co.candidate_id,
    ca.name AS candidate_name,
    co.amount,
    co.date
  FROM contributions co
  JOIN candidates ca
    ON co.candidate_id = ca.id
  LIMIT 5;
""", con))

# -----------------------------------------------------------------------------
# Close the database connection
# -----------------------------------------------------------------------------
con.close()
# %%
