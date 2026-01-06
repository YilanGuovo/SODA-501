###############################################################################
# Regular Expressions Tutorial: R
# Author: Jared Edgerton
# Date: Sys.Date()
#
# Regular expressions (regex) are powerful tools for pattern matching and
# text manipulation. This script introduces regex in R using the stringr package.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Install (if needed) and load stringr, which provides a consistent interface
# for working with strings and regular expressions in R.

# install.packages("stringr")
library(stringr)

# -----------------------------------------------------------------------------
# Part 1: Basic Patterns
# -----------------------------------------------------------------------------

# Basic matching
text <- "The quick brown fox jumps over the lazy dog"
str_detect(text, "quick")  # TRUE
str_detect(text, "cat")    # FALSE

# Character classes
digits <- "There are 42 apples and 7 oranges"
str_extract_all(digits, "\\d+")  # Extracts one or more digits

# Wildcards
wildcard <- "The cat and the hat sat on the mat"
str_extract_all(wildcard, "..at")  # Matches any two chars + "at"

# -----------------------------------------------------------------------------
# Part 2: Quantifiers and Anchors
# -----------------------------------------------------------------------------

# Quantifiers
text <- "The color is gray, or maybe grey"
str_extract_all(text, "gr[a|e]y?")  # NOTE: This pattern treats | literally in []
# A more typical version would be:
# str_extract_all(text, "gr[ae]y?")         # gray or grey
# or:
# str_extract_all(text, "gr(a|e)y?")        # using a group

# Anchors
names <- c("John Doe", "Jane Doe", "Joe Smith")
str_detect(names, "^J")     # Starts with "J"
str_detect(names, "Doe$")   # Ends with "Doe"

# -----------------------------------------------------------------------------
# Part 3: Groups and Backreferences
# -----------------------------------------------------------------------------

# Groups
text <- "The date is 2023-05-15"
str_match(text, "(\\d{4})-(\\d{2})-(\\d{2})")
# Returns a matrix: full match + capture groups (year, month, day)

# Backreferences
html <- "<p>This is a paragraph</p><p>This is another paragraph</p>"
result <- str_replace_all(html, '<(\\w+)>(.*?)</\\1>', '[\\2]')
print(result)
# Expected: "[This is a paragraph][This is another paragraph]"

# -----------------------------------------------------------------------------
# Practice Tasks
# -----------------------------------------------------------------------------

# 1) Email Validation:
# Create a regex pattern to validate email addresses. Test it with various valid
# and invalid email formats.

validate_email <- function(email) {
  pattern <- "^[\\w\\.]+@[\\w\\.]+\\.\\w+$"
  str_detect(email, pattern)
}

# Test the function
emails <- c(
  "user@example.com",
  "invalid.email",
  "another.user@subdomain.example.co.uk"
)
sapply(emails, validate_email)

# 2) URL Extraction:
# Write a function that extracts all URLs from a given text. Consider different
# URL formats (http, https, www, etc.).

extract_urls <- function(text) {
  pattern <- "https?://[\\w\\d\\.-]+\\.\\w{2,}|www\\.[\\w\\d\\.-]+\\.\\w{2,}"
  str_extract_all(text, pattern)
}

# Test the function
sample_text <- "Check out https://www.example.com and http://another-example.org for more info."
extract_urls(sample_text)

# -----------------------------------------------------------------------------
# Incomplete Tasks (for students to complete)
# -----------------------------------------------------------------------------

# 1) Phone Number Formatting:
# Create a regex pattern to identify and format phone numbers in the text.
# The function should work for different formats:
#   - (123) 456-7890
#   - 123-456-7890
#   - 1234567890
# and standardize them to: (XXX) XXX-XXXX.
#
# TODO: Implement this function.
# format_phone_numbers <- function(text) {
#   # pattern <- ...
#   # Use str_replace_all / str_replace to standardize
# }

# 2) Password Strength Checker:
# Implement a function that uses regex to check the strength of a password.
# Criteria:
#   - At least 8 characters long
#   - At least one uppercase letter
#   - At least one lowercase letter
#   - At least one digit
#   - At least one special character (@$!%*?&)
#
# TODO: Implement this function.
# check_password_strength <- function(password) {
#   # Use str_detect with multiple patterns (or a single combined pattern)
# }
