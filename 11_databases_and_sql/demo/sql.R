###############################################################################
# SQL Tutorial in R: SQLite + Campaign Finance (Simulated Data)
# Author: Jared Edgerton
# Date: Sys.Date()
#
# This script demonstrates:
#   1) Creating a local SQLite database (tables + indexes)
#   2) Inserting simulated campaign finance data
#   3) Writing and running SQL queries from R
#   4) Visualizing query outputs with ggplot2
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No conditional statements (no if/else).
# - Steps are repeated explicitly so students can follow and modify each piece.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Install (if needed) and load the necessary libraries.

# install.packages(c("DBI", "RSQLite", "dplyr", "ggplot2", "lubridate"))
library(DBI)
library(RSQLite)
library(dplyr)
library(ggplot2)
library(lubridate)

# -----------------------------------------------------------------------------
# Part 1: Create and Populate a Local SQLite Database
# -----------------------------------------------------------------------------
# SQLite is a lightweight database that lives in a single file (e.g., .db).
# DBI provides a consistent database interface; RSQLite is the SQLite backend.

# -----------------------------------------------------------------------------
# Step 1: Connect to a database file
# -----------------------------------------------------------------------------
# If the file does not exist, SQLite creates it automatically.
con <- dbConnect(RSQLite::SQLite(), "campaign_finance.db")

# -----------------------------------------------------------------------------
# Step 2: Drop tables (so the script can be rerun from scratch)
# -----------------------------------------------------------------------------
# SQL keyword notes:
# - DROP TABLE removes a table
# - IF EXISTS prevents errors if the table does not exist

dbExecute(con, "DROP TABLE IF EXISTS contributions;")
dbExecute(con, "DROP TABLE IF EXISTS contributors;")
dbExecute(con, "DROP TABLE IF EXISTS candidates;")

# -----------------------------------------------------------------------------
# Step 3: Create tables
# -----------------------------------------------------------------------------
# SQL keyword notes:
# - CREATE TABLE creates a new table
# - PRIMARY KEY uniquely identifies each row
# - FOREIGN KEY enforces relationships between tables (relational structure)

dbExecute(con, "
  CREATE TABLE candidates (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    party TEXT,
    office TEXT,
    winner INTEGER  -- 1 = winner, 0 = not winner (SQLite stores booleans as integers)
  );
")

dbExecute(con, "
  CREATE TABLE contributors (
    id INTEGER PRIMARY KEY,
    name TEXT,
    occupation TEXT,
    employer TEXT,
    state TEXT
  );
")

dbExecute(con, "
  CREATE TABLE contributions (
    id INTEGER PRIMARY KEY,
    contributor_id INTEGER,
    candidate_id INTEGER,
    amount REAL,
    date TEXT,
    FOREIGN KEY (contributor_id) REFERENCES contributors(id),
    FOREIGN KEY (candidate_id) REFERENCES candidates(id)
  );
")

# -----------------------------------------------------------------------------
# Step 4: Generate simulated data
# -----------------------------------------------------------------------------
# Teaching idea:
# - We are building a "toy" campaign finance database that is large enough
#   to make SQL meaningful, but still simple enough to understand.

set.seed(123)

# Candidates table (100 candidates)
candidates <- data.frame(
  id = 1:100,
  name = paste("Candidate", 1:100),
  party = sample(
    c("Democrat", "Republican", "Independent"),
    100,
    replace = TRUE,
    prob = c(0.45, 0.45, 0.10)
  ),
  office = sample(
    c("Senate", "House", "Governor", "State Senate", "State House"),
    100,
    replace = TRUE
  ),
  winner = sample(c(1, 0), 100, replace = TRUE, prob = c(0.5, 0.5))
)

# Contributors table (100,000 contributors)
contributors <- data.frame(
  id = 1:100000,
  name = paste("Contributor", 1:100000),
  occupation = sample(
    c("Engineer", "Teacher", "Doctor", "Lawyer", "Business Owner"),
    100000,
    replace = TRUE
  ),
  employer = paste("Company", sample(1:5000, 100000, replace = TRUE)),
  state = sample(state.abb, 100000, replace = TRUE)
)

# Contributions table (1,000,000 contributions)
# - amount is log-normal to mimic a skewed donation distribution
# - date is sampled uniformly across 2024
contrib_dates <- seq(as.Date("2024-01-01"), as.Date("2024-12-31"), by = "day")

contributions <- data.frame(
  id = 1:1000000,
  contributor_id = sample(1:100000, 1000000, replace = TRUE),
  candidate_id = sample(1:100, 1000000, replace = TRUE),
  amount = round(rlnorm(1000000, meanlog = log(1000), sdlog = 1), 2),
  date = as.character(sample(contrib_dates, 1000000, replace = TRUE))
)

# -----------------------------------------------------------------------------
# Step 5: Insert data into the database
# -----------------------------------------------------------------------------
# dbWriteTable() writes a data.frame into a database table.

dbWriteTable(con, "candidates", candidates, append = TRUE)
dbWriteTable(con, "contributors", contributors, append = TRUE)
dbWriteTable(con, "contributions", contributions, append = TRUE)

# -----------------------------------------------------------------------------
# Step 6: Create indexes (performance optimization)
# -----------------------------------------------------------------------------
# Indexes speed up queries that filter/join on these columns.
# SQL keyword notes:
# - CREATE INDEX builds an index structure for faster lookups
# - IF NOT EXISTS prevents errors if an index already exists

dbExecute(con, "CREATE INDEX IF NOT EXISTS idx_contrib_contributor_id ON contributions (contributor_id);")
dbExecute(con, "CREATE INDEX IF NOT EXISTS idx_contrib_candidate_id   ON contributions (candidate_id);")
dbExecute(con, "CREATE INDEX IF NOT EXISTS idx_contrib_amount         ON contributions (amount);")
dbExecute(con, "CREATE INDEX IF NOT EXISTS idx_contrib_date           ON contributions (date);")

# -----------------------------------------------------------------------------
# Step 7: Quick sanity checks (counts + small samples)
# -----------------------------------------------------------------------------
cat("\n------------------------------\n")
cat("Sanity checks: table sizes\n")
cat("------------------------------\n")

print(dbGetQuery(con, "SELECT COUNT(*) AS n_candidates FROM candidates;"))
print(dbGetQuery(con, "SELECT COUNT(*) AS n_contributors FROM contributors;"))
print(dbGetQuery(con, "SELECT COUNT(*) AS n_contributions FROM contributions;"))

cat("\n------------------------------\n")
cat("Sanity checks: sample rows\n")
cat("------------------------------\n")

print(dbGetQuery(con, "SELECT * FROM candidates LIMIT 5;"))
print(dbGetQuery(con, "
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
"))

# -----------------------------------------------------------------------------
# Part 2: SQL Queries for Analysis
# -----------------------------------------------------------------------------
# In each example below, we:
#   (1) Write a SQL query as a string
#   (2) Run it with dbGetQuery(con, query)
#   (3) Print or plot the result
#
# SQL syntax checklist you’ll see repeatedly:
# - SELECT: which columns to return
# - FROM: which table to start from
# - JOIN ... ON: how to combine tables
# - WHERE: filter rows
# - GROUP BY: aggregate by groups
# - ORDER BY: sort results
# - LIMIT: keep only the top rows

# -----------------------------------------------------------------------------
# Query 1: Top 10 occupations by total contribution amount
# -----------------------------------------------------------------------------
# Concepts:
# - SUM() aggregates amounts by occupation
# - COUNT(*) counts rows per group
# - GROUP BY creates groups
# - ORDER BY sorts totals descending

query_1 <- "
  SELECT
    c.occupation,
    SUM(co.amount) AS total_amount,
    COUNT(*) AS num_contributions
  FROM contributors c
  JOIN contributions co
    ON c.id = co.contributor_id
  GROUP BY c.occupation
  ORDER BY total_amount DESC
  LIMIT 10;
"
top_occupations <- dbGetQuery(con, query_1)

cat("\n------------------------------\n")
cat("Top 10 occupations by total contribution amount\n")
cat("------------------------------\n")
print(top_occupations)

# -----------------------------------------------------------------------------
# Query 2: Percentage of contributions by party for donations > $1000
# -----------------------------------------------------------------------------
# Concepts:
# - WHERE filters to only contributions > 1000
# - subquery (SELECT SUM(...)) computes the total for the denominator
# - grouping by party creates one row per party

query_2 <- "
  SELECT
    ca.party,
    SUM(co.amount) AS total_amount,
    SUM(co.amount) * 100.0 / (
      SELECT SUM(amount)
      FROM contributions
      WHERE amount > 1000
    ) AS percentage
  FROM contributions co
  JOIN candidates ca
    ON co.candidate_id = ca.id
  WHERE co.amount > 1000
  GROUP BY ca.party;
"
party_pct <- dbGetQuery(con, query_2)

cat("\n------------------------------\n")
cat("Percent of total contributions by party (amount > $1000)\n")
cat("------------------------------\n")
print(party_pct)

# -----------------------------------------------------------------------------
# Query 3: Candidates receiving contributions from the most distinct states
# -----------------------------------------------------------------------------
# Concepts:
# - COUNT(DISTINCT state) counts unique states
# - join chain: candidates -> contributions -> contributors
# - ORDER BY sorts by (num_states, then contribution_count)

query_3 <- "
  SELECT
    ca.id AS candidate_id,
    ca.name,
    ca.party,
    COUNT(DISTINCT c.state) AS num_states,
    COUNT(co.id) AS contribution_count
  FROM candidates ca
  JOIN contributions co
    ON ca.id = co.candidate_id
  JOIN contributors c
    ON co.contributor_id = c.id
  GROUP BY ca.id, ca.name, ca.party
  ORDER BY num_states DESC, contribution_count DESC
  LIMIT 5;
"
candidates_most_states <- dbGetQuery(con, query_3)

cat("\n------------------------------\n")
cat("Candidates with contributions from the most distinct states\n")
cat("------------------------------\n")
print(candidates_most_states)

# -----------------------------------------------------------------------------
# Query 4: Contributors donating to multiple parties (cross-party donors)
# -----------------------------------------------------------------------------
# Concepts:
# - HAVING filters groups after aggregation
# - GROUP_CONCAT summarizes unique parties into one string (SQLite feature)
# - CASE WHEN creates party-specific sums

query_4 <- "
  SELECT
    c.name,
    GROUP_CONCAT(DISTINCT ca.party) AS parties,
    SUM(CASE WHEN ca.party = 'Democrat' THEN co.amount ELSE 0 END) AS dem_amount,
    SUM(CASE WHEN ca.party = 'Republican' THEN co.amount ELSE 0 END) AS rep_amount
  FROM contributors c
  JOIN contributions co
    ON c.id = co.contributor_id
  JOIN candidates ca
    ON co.candidate_id = ca.id
  GROUP BY c.id
  HAVING COUNT(DISTINCT ca.party) > 1
  LIMIT 20;
"
cross_party <- dbGetQuery(con, query_4)

cat("\n------------------------------\n")
cat("Cross-party contributors (sample)\n")
cat("------------------------------\n")
print(cross_party)

# -----------------------------------------------------------------------------
# Query 5: Moving average of contribution amounts for top 3 candidates
# -----------------------------------------------------------------------------
# Concepts:
# - WITH ... AS creates CTEs (Common Table Expressions)
# - window functions use OVER (PARTITION BY ... ORDER BY ...)
# - ROWS BETWEEN 29 PRECEDING AND CURRENT ROW defines a rolling window
# - This query:
#   (1) builds cumulative totals per candidate
#   (2) computes 30-day moving average of contribution amounts by candidate
#   (3) keeps only the top 3 candidates by cumulative amount

query_5 <- "
  WITH ranked_contributions AS (
    SELECT
      ca.id,
      ca.name,
      co.date,
      co.amount,
      SUM(co.amount) OVER (
        PARTITION BY ca.id
        ORDER BY co.date
      ) AS cumulative_amount
    FROM candidates ca
    JOIN contributions co
      ON ca.id = co.candidate_id
  ),
  windowed_avg AS (
    SELECT
      id,
      name,
      date,
      amount,
      AVG(amount) OVER (
        PARTITION BY id
        ORDER BY date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
      ) AS moving_avg
    FROM ranked_contributions
  )
  SELECT
    id,
    name,
    date,
    amount,
    moving_avg
  FROM windowed_avg
  WHERE id IN (
    SELECT id
    FROM ranked_contributions
    GROUP BY id
    ORDER BY MAX(cumulative_amount) DESC
    LIMIT 3
  )
  ORDER BY id, date
  LIMIT 100;
"
moving_avg <- dbGetQuery(con, query_5)

cat("\n------------------------------\n")
cat("Moving average contributions (top 3 candidates, sample rows)\n")
cat("------------------------------\n")
print(moving_avg)

# -----------------------------------------------------------------------------
# Query 6: Visualize total contributions by party
# -----------------------------------------------------------------------------
# Concepts:
# - This query aggregates contribution totals by party (SUM + GROUP BY)
# - Then we plot with ggplot2

query_6 <- "
  SELECT
    ca.party,
    SUM(co.amount) AS total_amount
  FROM contributions co
  JOIN candidates ca
    ON co.candidate_id = ca.id
  GROUP BY ca.party;
"
party_totals <- dbGetQuery(con, query_6)

cat("\n------------------------------\n")
cat("Total contributions by party\n")
cat("------------------------------\n")
print(party_totals)

party_plot <- ggplot(party_totals, aes(x = party, y = total_amount, fill = party)) +
  geom_col() +
  theme_minimal() +
  labs(
    title = "Total Contributions by Party",
    x = "Party",
    y = "Total Amount ($)"
  )

print(party_plot)
ggsave("contributions_by_party.png", party_plot, width = 7, height = 4)

cat("\nSaved plot: contributions_by_party.png\n")

# -----------------------------------------------------------------------------
# Close the database connection
# -----------------------------------------------------------------------------
dbDisconnect(con)
