###############################################################################
# Text as Data Pipelines Tutorial: Python
# Author: Jared Edgerton
# Date: (use your local date/time)
#
# This script demonstrates (lightweight workflow):
#   1) Tokenization + basic text preprocessing
#   2) A classic topic model (LDA)
#   3) Word-embedding regression (Word2Vec -> document vectors -> Ridge regression)
#   4) A BERT-based topic model (BERTopic)
#
# Week context:
# - Text as Data Pipelines
# - Coding lab: tokenization; embeddings; topic models; basic transformer workflow.
# - Pre-class video: practical text pipeline architecture.
#
# Teaching note (important):
# - This file is intentionally written as a sequential workflow so students can
#   see how the pipeline unfolds.
# - No user-defined functions (no def ...).
# - Minimal "magic": explicit steps and prints.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Recommended installs (run once in terminal):
#
#   pip install pandas numpy matplotlib scikit-learn gensim
#
# For BERTopic (heavier; may take a bit to install):
#
#   pip install bertopic sentence-transformers umap-learn hdbscan
#
# If hdbscan fails on Windows, consider:
#   - conda install -c conda-forge hdbscan
#   - then pip install bertopic sentence-transformers umap-learn
#
# NOTE: Students can run Parts 1–3 without BERTopic if installation is a barrier.
# (BERTopic is included because it is part of this week's material.)

import os
import re
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gensim.models import Word2Vec

# Reproducibility
random.seed(123)
np.random.seed(123)

# Create project folders (safe to run repeatedly)
os.makedirs("data_raw", exist_ok=True)
os.makedirs("data_processed", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("src", exist_ok=True)

# -----------------------------------------------------------------------------
# Part 0: Create a small synthetic text dataset (for a self-contained tutorial)
# -----------------------------------------------------------------------------
# In a real pipeline, this would be where you load text data from a file/API/DB.
# Example:
#   df = pd.read_csv("data_raw/my_corpus.csv")
#   docs = df["text"].tolist()

topic_words_elections = [
    "election", "vote", "turnout", "campaign", "candidate", "ballot", "district",
    "polling", "primary", "debate", "incumbent", "swing", "party", "coalition"
]
topic_words_health = [
    "health", "hospital", "vaccine", "clinic", "patient", "disease", "outbreak",
    "public", "policy", "insurance", "treatment", "nurse", "doctor", "risk"
]
topic_words_misinformation = [
    "misinformation", "platform", "social", "media", "bot", "rumor", "viral",
    "network", "algorithm", "moderation", "content", "narrative", "false", "signal"
]
common_words = [
    "study", "evidence", "analysis", "data", "method", "results", "model",
    "effects", "estimate", "theory", "research", "observed"
]

docs = []
true_topic = []
y = []

# We generate 300 short documents: 100 per topic
# y is a synthetic "outcome" that depends on topic + random noise
# (This is for the regression section later.)

for i in range(100):
    # Elections docs
    tokens = (
        random.choices(topic_words_elections, k=random.randint(30, 45)) +
        random.choices(common_words, k=random.randint(10, 20))
    )
    random.shuffle(tokens)
    docs.append(" ".join(tokens))
    true_topic.append("elections")
    y.append(1.0 + np.random.normal(0, 0.5))

for i in range(100):
    # Health docs
    tokens = (
        random.choices(topic_words_health, k=random.randint(30, 45)) +
        random.choices(common_words, k=random.randint(10, 20))
    )
    random.shuffle(tokens)
    docs.append(" ".join(tokens))
    true_topic.append("health")
    y.append(0.0 + np.random.normal(0, 0.5))

for i in range(100):
    # Misinformation docs
    tokens = (
        random.choices(topic_words_misinformation, k=random.randint(30, 45)) +
        random.choices(common_words, k=random.randint(10, 20))
    )
    random.shuffle(tokens)
    docs.append(" ".join(tokens))
    true_topic.append("misinformation")
    y.append(-1.0 + np.random.normal(0, 0.5))

df = pd.DataFrame(
    {
        "doc_id": np.arange(1, len(docs) + 1),
        "text": docs,
        "true_topic": true_topic,
        "y_outcome": y,
    }
)

print("\n--- Synthetic corpus preview ---")
print(df.head())
print("\n--- True topic distribution ---")
print(df["true_topic"].value_counts())

df.to_csv("data_raw/week_synthetic_corpus.csv", index=False)

# -----------------------------------------------------------------------------
# Part 1: Tokenization + basic preprocessing
# -----------------------------------------------------------------------------
# For many "classic" text-as-data workflows, we build a document-term matrix (DTM)
# with CountVectorizer or TF-IDF. This implicitly defines a tokenizer + vocabulary.

vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # words with >=2 letters
    min_df=2
)

X_counts = vectorizer.fit_transform(df["text"])
vocab = vectorizer.get_feature_names_out()

print("\n--- Document-term matrix (counts) ---")
print("Shape:", X_counts.shape)  # (n_docs, n_terms)
print("Vocabulary size:", len(vocab))
print("Example vocab terms:", vocab[:20])

# Top terms by total count (quick diagnostic)
term_totals = np.asarray(X_counts.sum(axis=0)).ravel()
top_idx = term_totals.argsort()[::-1][:15]
top_terms = pd.DataFrame({"term": vocab[top_idx], "total_count": term_totals[top_idx]})
print("\n--- Top terms by total count ---")
print(top_terms)

top_terms.to_csv("outputs/week_top_terms.csv", index=False)

# -----------------------------------------------------------------------------
# Part 2: Classic topic model (LDA)
# -----------------------------------------------------------------------------
n_topics = 6
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=123,
    learning_method="batch"
)
lda.fit(X_counts)

# Topic-word distributions
topic_word = lda.components_  # shape: (K, n_terms)

print("\n--- LDA topics: top words ---")
n_top_words = 10
for k in range(n_topics):
    top_word_idx = topic_word[k].argsort()[::-1][:n_top_words]
    words = vocab[top_word_idx]
    weights = topic_word[k][top_word_idx]
    print(f"\nTopic {k}:")
    for w, wt in zip(words, weights):
        print(f"  {w:15s} {wt:,.2f}")

# Document-topic proportions
doc_topic = lda.transform(X_counts)  # shape: (n_docs, K)
df_lda = df.copy()
df_lda["lda_topic"] = doc_topic.argmax(axis=1)
df_lda["lda_topic_prob"] = doc_topic.max(axis=1)

print("\n--- LDA: dominant topic counts ---")
print(df_lda["lda_topic"].value_counts().sort_index())

df_lda.to_csv("data_processed/week_with_lda_topics.csv", index=False)

topic_counts = df_lda["lda_topic"].value_counts().sort_index()
plt.figure(figsize=(8, 4))
plt.bar(topic_counts.index.astype(str), topic_counts.values)
plt.title("LDA: Dominant Topic Counts (Synthetic Corpus)")
plt.xlabel("Dominant topic")
plt.ylabel("Number of documents")
plt.tight_layout()
plt.savefig("figures/week_lda_dominant_topic_counts.png", dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# Part 3: Word embedding regression (Word2Vec -> doc vectors -> Ridge regression)
# -----------------------------------------------------------------------------
tokenized_docs = []
for text in df["text"].tolist():
    tokens = re.findall(r"[a-z]+", text.lower())
    tokenized_docs.append(tokens)

print("\n--- Tokenization check ---")
print("Example tokens:", tokenized_docs[0][:20])

w2v = Word2Vec(
    sentences=tokenized_docs,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1,  # skip-gram
    seed=123
)

print("\n--- Word2Vec vocabulary size ---")
print(len(w2v.wv.index_to_key))

doc_vectors = []
for tokens in tokenized_docs:
    vecs = w2v.wv[tokens]
    doc_vec = vecs.mean(axis=0)
    doc_vectors.append(doc_vec)

doc_vectors = np.vstack(doc_vectors)
print("\n--- Document embedding matrix ---")
print("Shape:", doc_vectors.shape)

X_train, X_test, y_train, y_test = train_test_split(
    doc_vectors, df["y_outcome"].values, test_size=0.25, random_state=123
)

ridge = Ridge(alpha=1.0, random_state=123)
ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Word2Vec regression (Ridge) out-of-sample performance ---")
print("MAE: ", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R^2: ", round(r2, 4))

metrics = pd.DataFrame(
    {"model": ["word2vec_ridge"], "mae": [mae], "rmse": [rmse], "r2": [r2]}
)
metrics.to_csv("outputs/week_word2vec_regression_metrics.csv", index=False)

plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.title("Word2Vec Regression: Predicted vs Actual")
plt.xlabel("Actual y_outcome")
plt.ylabel("Predicted y_outcome")
plt.tight_layout()
plt.savefig("figures/week_word2vec_pred_vs_actual.png", dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# Part 4: BERT-based topic model (BERTopic)
# -----------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
import hdbscan

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(df["text"].tolist(), show_progress_bar=True)

print("\n--- Transformer embedding matrix ---")
print("Shape:", embeddings.shape)

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=123
)

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=15,
    metric="euclidean",
    cluster_selection_method="eom"
)

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=True,
    verbose=True
)

topics, probs = topic_model.fit_transform(df["text"].tolist(), embeddings)

df_bert = df.copy()
df_bert["bertopic_topic"] = topics
df_bert["bertopic_max_prob"] = np.max(probs, axis=1)

print("\n--- BERTopic: topic counts ---")
print(pd.Series(topics).value_counts().sort_index())

topic_info = topic_model.get_topic_info()
print("\n--- BERTopic: topic info (head) ---")
print(topic_info.head(10))

df_bert.to_csv("data_processed/week_with_bertopic.csv", index=False)
topic_info.to_csv("outputs/week_bertopic_topic_info.csv", index=False)

topic_counts_bt = topic_info.loc[topic_info["Topic"] != -1, ["Topic", "Count"]]
plt.figure(figsize=(8, 4))
plt.bar(topic_counts_bt["Topic"].astype(str), topic_counts_bt["Count"])
plt.title("BERTopic: Topic Counts (Excluding Outliers)")
plt.xlabel("Topic")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/week_bertopic_topic_counts.png", dpi=200)
plt.close()
