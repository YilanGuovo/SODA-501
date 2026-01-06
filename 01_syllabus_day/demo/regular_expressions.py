###############################################################################
# Regular Expressions Tutorial: Python
# Author: Jared Edgerton
# Date: date.today()
#
# This file mirrors the R/stringr version using Python's built-in `re` module.
###############################################################################

import re
from datetime import date



# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# In Python, regular expressions are handled by the built-in `re` module (no install).
# We'll define a few small helper functions so the code reads similarly to stringr.

def str_detect(x, pattern):
    """Return True/False (or a list of True/False) depending on whether pattern is found."""
    if isinstance(x, (list, tuple)):
        return [re.search(pattern, s) is not None for s in x]
    return re.search(pattern, x) is not None

def str_extract_all(x, pattern):
    """Return all matches (or a list of match-lists if x is a list/tuple)."""
    if isinstance(x, (list, tuple)):
        return [re.findall(pattern, s) for s in x]
    return re.findall(pattern, x)

def str_match(x, pattern):
    """
    Return a list like: [full_match, group1, group2, ...]
    (similar spirit to stringr::str_match).
    """
    m = re.search(pattern, x)
    if not m:
        return None
    return [m.group(0), *m.groups()]

def str_replace_all(x, pattern, replacement):
    """Replace all occurrences of pattern with replacement."""
    return re.sub(pattern, replacement, x)


# -----------------------------------------------------------------------------
# Part 1: Basic Patterns
# -----------------------------------------------------------------------------

# Basic matching
text = "The quick brown fox jumps over the lazy dog"
print(str_detect(text, "quick"))  # True
print(str_detect(text, "cat"))    # False

# Character classes
digits = "There are 42 apples and 7 oranges"
print(str_extract_all(digits, "\\d+"))  # ["42", "7"]

# Wildcards
wildcard = "The cat and the hat sat on the mat"
print(str_extract_all(wildcard, "..at"))  # ["cat", "hat", "sat", "mat"]


# -----------------------------------------------------------------------------
# Part 2: Quantifiers and Anchors
# -----------------------------------------------------------------------------

# Quantifiers
text = "The color is gray, or maybe grey"
print(str_extract_all(text, "gr[a|e]y?"))  # ["gray", "grey"]

# Anchors
names = ["John Doe", "Jane Doe", "Joe Smith"]
print(str_detect(names, "^J"))     # [True, True, True]
print(str_detect(names, "Doe$"))   # [True, True, False]


# -----------------------------------------------------------------------------
# Part 3: Groups and Backreferences
# -----------------------------------------------------------------------------

# Groups
text = "The date is 2023-05-15"
print(str_match(text, "(\\d{4})-(\\d{2})-(\\d{2})"))
# Example output: ["2023-05-15", "2023", "05", "15"]

# Backreferences
html = "<p>This is a paragraph</p><p>This is another paragraph</p>"
result = str_replace_all(html, "<(\\w+)>(.*?)</\\1>", "[\\2]")
print(result)
# Expected: "[This is a paragraph][This is another paragraph]"


# -----------------------------------------------------------------------------
# Practice Tasks
# -----------------------------------------------------------------------------

# 1) Email Validation:
# Create a regex pattern to validate email addresses. Test it with various valid
# and invalid email formats.

def validate_email(email):
    pattern = "^[\\w\\.]+@[\\w\\.]+\\.\\w+$"
    return str_detect(email, pattern)

# Test the function
emails = ["user@example.com", "invalid.email", "another.user@subdomain.example.co.uk"]
print([validate_email(e) for e in emails])


# 2) URL Extraction:
# Write a function that extracts all URLs from a given text. Consider different
# URL formats (http, https, www, etc.).

def extract_urls(text):
    pattern = "https?://[\\w\\d\\.-]+\\.\\w{2,}|www\\.[\\w\\d\\.-]+\\.\\w{2,}"
    return str_extract_all(text, pattern)

# Test the function
sample_text = "Check out https://www.example.com and http://another-example.org for more info."
print(extract_urls(sample_text))


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
# def format_phone_numbers(text):
#     # pattern = ...
#     # Use re.sub / str_replace_all to standardize
#     pass

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
# def check_password_strength(password):
#     # Use str_detect with multiple patterns (or one combined pattern)
#     pass
