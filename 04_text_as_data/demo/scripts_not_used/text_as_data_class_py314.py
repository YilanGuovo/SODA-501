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

#pip install numpy pandas matplotlib scikit-learn scipy sentence-transformers wordcloud pyLDAvis
# (On Windows + Python 3.14, gensim often fails to build; this script uses a
# sentence-transformers embedding regression instead of gensim Word2Vec.)

os.chdir(r"D:\Yilan\1\Courses\SODA\501\soda_501\04_text_as_data\demo")
os.getcwd()

import os
import re
import random
import tarfile
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.manifold import TSNE


from wordcloud import WordCloud
from bertopic import BERTopic
from umap import UMAP
import hdbscan


# Required for Part 3 alternative
from sentence_transformers import SentenceTransformer

# Optional libraries (script will still run without them)
try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

try:
    import pyLDAvis
    import pyLDAvis.lda_model
except Exception:
    pyLDAvis = None

try:
    from bertopic import BERTopic
    from umap import UMAP
    import hdbscan
except Exception:
    BERTopic = None
    UMAP = None
    hdbscan = None

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
# Part 0: Load the CMU Movie Summary Corpus from MovieSummaries.tar.gz
# -----------------------------------------------------------------------------
# Expected archive location (based on your screenshot):
#   04_text_as_data/demo/MovieSummaries.tar.gz
#
# You can download here: http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz
#
# This block:
#   1) extracts the archive (if needed)
#   2) loads plot summaries + metadata
#   3) builds df with columns: doc_id, text, y_outcome, true_topic (optional)
#
# Teaching note:
# - We keep this explicit (no helper functions) so students can follow every step.
# -----------------------------------------------------------------------------



# 0.1 Paths
# 0.1 Paths (support both .tar.gz and .tar)
archive_dir = os.path.join("04_text_as_data", "demo")
archive_candidates = [
    os.path.join(archive_dir, "MovieSummaries.tar.gz"),
    os.path.join(archive_dir, "MovieSummaries.tar"),
    os.path.join(archive_dir, "MovieSummaries.tar.gzip"),
]
archive_path = None
for p in archive_candidates:
    if os.path.exists(p):
        archive_path = p
        break
if archive_path is None:
    # Fall back to the original expected name (will raise later if missing)
    archive_path = os.path.join(archive_dir, "MovieSummaries.tar.gz")
extract_dir  = os.path.join("04_text_as_data", "demo", "MovieSummaries_extracted")

os.makedirs(extract_dir, exist_ok=True)

print("\n--- Movie corpus: archive path ---")
print(archive_path)

# 0.2 Extract (only if the expected files are not already present)
# We will check for plot_summaries.txt and movie.metadata.tsv *somewhere* in extract_dir.
# Because tar archives sometimes include a top-level folder, we’ll (a) list members,
# then (b) extract all if needed.

need_extract = True
for root, dirs, files in os.walk(extract_dir):
    if ("plot_summaries.txt" in files) and ("movie.metadata.tsv" in files):
        need_extract = False
        break

if need_extract:
    print("\n--- Extracting MovieSummaries.tar ---")
    with tarfile.open(archive_path, "r:*") as tar:
        tar.extractall(path=extract_dir)
    print("Extraction complete:", extract_dir)
else:
    print("\n--- Extraction skipped (files already present) ---")

# 0.3 Locate the extracted files (they might be in a subfolder)
plots_path = None
meta_path  = None

for root, dirs, files in os.walk(extract_dir):
    if "plot_summaries.txt" in files:
        plots_path = os.path.join(root, "plot_summaries.txt")
    if "movie.metadata.tsv" in files:
        meta_path = os.path.join(root, "movie.metadata.tsv")

print("\n--- Located extracted files ---")
print("plots_path:", plots_path)
print("meta_path: ", meta_path)

if plots_path is None or meta_path is None:
    raise FileNotFoundError("Could not find plot_summaries.txt and/or movie.metadata.tsv after extraction.")

# 0.4 Load plot summaries (tab-separated: wikipedia_movie_id \t plot_summary)
plots = pd.read_csv(
    plots_path,
    sep="\t",
    header=None,
    names=["wikipedia_movie_id", "text"],
    encoding="utf-8",
    quoting=3,
    on_bad_lines="skip"
)

print("\n--- Plots loaded ---")
print("plots shape:", plots.shape)
print(plots.head())

# 0.5 Load metadata (tab-separated, no header)
# CMU metadata columns (by position) are commonly:
# 0 wikipedia_movie_id
# 1 freebase_movie_id
# 2 movie_name
# 3 release_date
# 4 box_office_revenue
# 5 runtime
# 6 languages
# 7 countries
# 8 genres (dict-like string)
meta = pd.read_csv(
    meta_path,
    sep="\t",
    header=None,
    encoding="utf-8",
    quoting=3,
    on_bad_lines="skip"
)

print("\n--- Metadata loaded ---")
print("meta shape:", meta.shape)
print(meta.head())

# Name the key columns we’ll actually use
meta = meta.rename(columns={
    0: "wikipedia_movie_id",
    2: "movie_name",
    3: "release_date",
    8: "genres_raw"
})

# Keep only the columns we care about (this reduces confusion for students)
meta = meta[["wikipedia_movie_id", "movie_name", "release_date", "genres_raw"]]

# 0.6 Merge plots + metadata
df = plots.merge(meta, on="wikipedia_movie_id", how="left")

print("\n--- Merged movie df ---")
print("df shape:", df.shape)
print(df[["wikipedia_movie_id", "movie_name", "release_date"]].head())

# 0.7 Clean text (drop missing/very short summaries)
df["text"] = df["text"].astype(str)
df["text_len"] = df["text"].str.len()

df = df[df["text_len"] >= 200].copy()   # keep reasonably informative plots
df = df.dropna(subset=["text"]).copy()

print("\n--- After filtering short plots ---")
print("df shape:", df.shape)
print(df["text_len"].describe())

# 0.8 Create an outcome for the regression section (simple + interpretable)
# Here: y_outcome = 1 if "Action" appears in the genres dict, else 0
# genres_raw is often a dict-like string; we parse it with ast.literal_eval.

y_outcome = []
true_topic = []

for g in df["genres_raw"].tolist():
    # Default values if parsing fails
    is_action = 0
    topic_label = "other"

    if isinstance(g, str) and len(g) > 0:
        try:
            g_dict = ast.literal_eval(g)   # dict of {freebase_genre_id: genre_name}
            g_text = " ".join([str(v) for v in g_dict.values()]).lower()

            if "action" in g_text:
                is_action = 1
                topic_label = "action"
            elif "comedy" in g_text:
                topic_label = "comedy"
            elif "drama" in g_text:
                topic_label = "drama"
            elif "horror" in g_text:
                topic_label = "horror"
            else:
                topic_label = "other"

        except Exception:
            # If genre parsing fails, keep defaults
            pass

    y_outcome.append(is_action)
    true_topic.append(topic_label)

df["y_outcome"] = y_outcome
df["true_topic"] = true_topic

print("\n--- Outcome + rough topic labels ---")
print(df["y_outcome"].value_counts(dropna=False))
print(df["true_topic"].value_counts().head(10))

# 0.9 Optional: downsample for classroom runtime (BERTopic + embeddings can be heavy)
# Keep it reproducible with seed=123.
max_docs = 800  # adjust (e.g., 800 for quick laptops, 3000 for your machine)
if df.shape[0] > max_docs:
    df = df.sample(n=max_docs, random_state=123).copy()

# 0.10 Add doc_id and keep a clean set of columns for later steps
df = df.reset_index(drop=True)
df["doc_id"] = np.arange(1, df.shape[0] + 1)

df = df[["doc_id", "wikipedia_movie_id", "movie_name", "release_date", "text", "true_topic", "y_outcome"]].copy()

print("\n--- Final movie corpus preview ---")
print(df.head())
print("\n--- Corpus size ---")
print(df.shape)

df.to_csv("data_raw/week_movie_corpus.csv", index=False)


# -----------------------------------------------------------------------------
# Part 1: Tokenization + basic preprocessing
# -----------------------------------------------------------------------------
# For many "classic" text-as-data workflows, we build a document-term matrix (DTM)
# with CountVectorizer or TF-IDF. This implicitly defines a tokenizer + vocabulary.

df = pd.read_csv("data_raw/week_movie_corpus.csv")

#os.makedirs("/data_raw", exist_ok=True)
#os.makedirs("/data_processed", exist_ok=True)
#os.makedirs("/figures", exist_ok=True)
#os.makedirs("/outputs", exist_ok=True)
#os.makedirs("/Week 4/src", exist_ok=True)

vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # words with >=2 letters
    min_df=5
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

# Generate a dictionary of word frequencies from the CountVectorizer results
word_freqs = dict(zip(vocab, term_totals))

# Create the Word Cloud
# Create the Word Cloud (optional; skip if wordcloud isn't installed)
if WordCloud is not None:
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
    ).generate_from_frequencies(word_freqs)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Document Terms (min_df=5)", fontsize=16)
    plt.savefig(
        "figures/wordcloud.png",
        dpi=200
    )
    plt.show()
else:
    print("[Skipped] WordCloud not installed (pip install wordcloud).")


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
f = open("outputs/week_lda_top_words.txt", "w", encoding="utf-8")
for k in range(n_topics):
    top_word_idx = topic_word[k].argsort()[::-1][:n_top_words]
    words = vocab[top_word_idx]
    weights = topic_word[k][top_word_idx]

    print(f"\nTopic {k}:")
    f.write(f"\nTopic {k}:\n")

    for w, wt in zip(words, weights):
        print(f"  {w:15s} {wt:,.2f}")
        f.write(f"  {w:15s} {wt:,.2f}\n")
f.close()

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
plt.title("LDA: Dominant Topic Counts (Movie Plots)")
plt.xlabel("Dominant topic")
plt.ylabel("Number of documents")
plt.tight_layout()
plt.savefig("figures/week_lda_dominant_topic_counts.png", dpi=200)
plt.show()
plt.close()


# Interactive LDA visualization (optional; skip if pyLDAvis isn't installed)
if pyLDAvis is not None:
    lda_display = pyLDAvis.lda_model.prepare(
        lda,
        X_counts,
        vectorizer,
        mds="tsne",
    )
    pyLDAvis.save_html(
        lda_display,
        "Penn State/Spring 2026/SoDA 501/Week 4/outputs/lda_visualization.html"
    )
    print("\nSaved: outputs/lda_visualization.html")
else:
    print("\n[Skipped] pyLDAvis not installed (pip install pyLDAvis).")


# -----------------------------------------------------------------------------
# Part 3: Embedding regression (SentenceTransformer -> Ridge regression)
# -----------------------------------------------------------------------------
# NOTE (Windows + Python 3.14): gensim Word2Vec often fails to install due to
# compiled extensions. This section uses sentence-transformers to create document
# embeddings instead (works well for modern NLP pipelines).

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_vectors = embed_model.encode(df["text"].tolist(), show_progress_bar=True)

print("--- Document embedding matrix (SentenceTransformer) ---")
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

print("--- Embedding regression (Ridge) out-of-sample performance ---")
print("MAE: ", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R^2: ", round(r2, 4))

metrics = pd.DataFrame(
    {"model": ["sbert_ridge"], "mae": [mae], "rmse": [rmse], "r2": [r2]}
)
metrics.to_csv("outputs/week_embedding_regression_metrics.csv", index=False)

# Simple diagnostic plot (matplotlib only; no seaborn dependency)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, s=12)
plt.xlabel("Actual Outcome (y_test)")
plt.ylabel("Predicted Outcome (y_pred)")
plt.title("Ridge Regression: Actual vs. Predicted Values (SBERT embeddings)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/embedding_regression_actual_vs_predicted.png", dpi=200)
plt.show()

# 2D map of document embeddings (t-SNE) for classroom visualization
tsne = TSNE(n_components=2, random_state=123, perplexity=30)
embeddings_2d = tsne.fit_transform(doc_vectors)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=10)
plt.title("t-SNE Projection of Document Embeddings (SBERT)")
plt.tight_layout()
plt.savefig("figures/doc_embeddings_tsne.png", dpi=200)
plt.show()

# -----------------------------------------------------------------------------
# Part 4: BERT-based topic model (BERTopic)
# -----------------------------------------------------------------------------
if BERTopic is None or UMAP is None or hdbscan is None:
    print("\n[Skipped] BERTopic/UMAP/HDBSCAN not installed (pip install bertopic umap-learn hdbscan).")
else:

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
        min_cluster_size=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
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

    df_bert.to_csv("Penn State/Spring 2026/SoDA 501/Week 4/data_processed/week_with_bertopic.csv", index=False)
    topic_info.to_csv("Penn State/Spring 2026/SoDA 501/Week 4/outputs/week_bertopic_topic_info.csv", index=False)

    topic_counts_bt = topic_info.loc[topic_info["Topic"] != -1, ["Topic", "Count"]]
    plt.figure(figsize=(8, 4))
    plt.bar(topic_counts_bt["Topic"].astype(str), topic_counts_bt["Count"])
    plt.title("BERTopic: Topic Counts (Excluding Outliers)")
    plt.xlabel("Topic")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("Penn State/Spring 2026/SoDA 501/Week 4/figures/week_bertopic_topic_counts.png", dpi=200)
    plt.show()
    plt.close()

    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_hierarchy.show()
    fig_hierarchy.write_html("Penn State/Spring 2026/SoDA 501/Week 4/outputs/topic_hierarchy.html")

    fig_docs = topic_model.visualize_documents(df["text"].tolist(), embeddings=embeddings)
    fig_docs.show()
    fig_docs.write_html("Penn State/Spring 2026/SoDA 501/Week 4/outputs/document_map.html")


print(numba.__version__)

