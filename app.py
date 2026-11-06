"""Phase 1 GUI: a separate web UI (Streamlit) for the hybrid arXiv search engine.

Run:  streamlit run app.py
Needs corpus.csv (built by IRPROJECT123.ipynb) in the same folder.
"""
import hashlib
import re
from collections import Counter
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer

CORPUS = Path("corpus.csv")
MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60

ABBREVIATIONS = {
    "nlp": "natural language processing",
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "mt": "machine translation",
    "qa": "question answering",
    "llm": "large language model",
}
STOP = {"the", "and", "for", "with", "using", "from", "into", "are"}


# ---------------------------------------------------------------- engine
@st.cache_resource(show_spinner="Building BM25 index and dense embeddings...")
def load_engine():
    df = pd.read_csv(CORPUS).fillna("")
    documents = (df["title"] + " " + df["content"].str[:2000]).tolist()

    # BM25 (sparse): lowercase -> case-insensitive matching
    tokenized = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)

    # Dense: MiniLM embeddings, normalized so dot product = cosine similarity
    model = SentenceTransformer(MODEL_NAME)
    # Cache file is tied to the exact corpus text -> a new corpus.csv re-embeds automatically
    digest = hashlib.md5("\n".join(documents).encode("utf-8")).hexdigest()[:12]
    cache = Path(f"embeddings_{digest}.npy")
    if cache.exists():
        emb = np.load(cache)
    else:
        emb = model.encode(documents, normalize_embeddings=True, show_progress_bar=True)
        np.save(cache, emb)

    word_counts = Counter(w for doc in tokenized for w in doc)
    vocabulary = [w for w, c in word_counts.items() if c >= 2 and w.isalpha()]
    search_texts = [" ".join(doc.lower().split()) for doc in documents]  # for phrase matching
    return {"df": df, "bm25": bm25, "model": model, "emb": emb,
            "word_counts": word_counts, "vocabulary": vocabulary,
            "search_texts": search_texts}


def process_query(query, eng):
    """Lowercase -> expand abbreviations -> correct spelling (outside quoted phrases)."""
    query = query.lower().strip()
    for short, full in ABBREVIATIONS.items():
        query = re.sub(r"\b" + short + r"\b", full, query)

    parts = re.split(r'("[^"]*")', query)
    out = []
    for part in parts:
        if part.startswith('"') and part.endswith('"'):
            out.append(part)
            continue
        words = []
        for word in re.findall(r"\w+|[^\w\s]", part):
            if word.isalpha() and len(word) >= 4 and word not in eng["word_counts"]:
                match = process.extractOne(word, eng["vocabulary"],
                                           scorer=fuzz.ratio, score_cutoff=85)
                if match:
                    word = match[0]
            words.append(word)
        out.append(" ".join(words))
    return " ".join(p for p in out if p.strip()).strip()


def search(query, method, eng):
    processed = process_query(query, eng)
    if not processed:
        return processed, []
    n = len(eng["df"])
    plain = processed.replace('"', "")

    bm25_scores = eng["bm25"].get_scores(plain.split()).copy()
    q_emb = eng["model"].encode(plain, normalize_embeddings=True)
    dense_scores = eng["emb"] @ q_emb

    # Exact phrase matching
    phrases = [" ".join(p.split()) for p in re.findall(r'"([^"]+)"', processed)]
    phrase_matches = np.zeros(n)
    for phrase in phrases:
        for i, text in enumerate(eng["search_texts"]):
            if phrase in text:
                phrase_matches[i] += 1
    bm25_scores += phrase_matches * 5

    if method == "BM25":
        scores = bm25_scores
    elif method == "Dense":
        scores = dense_scores + phrase_matches * 0.1
    else:  # Reciprocal Rank Fusion
        scores = np.zeros(n)
        for rank, i in enumerate(np.argsort(-bm25_scores)):
            scores[i] += 1 / (RRF_K + rank + 1)
        for rank, i in enumerate(np.argsort(-dense_scores)):
            scores[i] += 1 / (RRF_K + rank + 1)
        scores += phrase_matches * 0.005

    # Rerank: a quoted phrase is a filter -> only papers that contain it
    ranking = list(np.argsort(-scores))
    if phrases:
        ranking = [i for i in ranking if phrase_matches[i] == len(phrases)]

    results = []
    for i in ranking[:10]:
        row = eng["df"].iloc[i]
        results.append({
            "title": " ".join(str(row["title"]).split()),
            "link": str(row["arxiv_link"]),
            "snippet": " ".join(str(row["content"]).split())[:200],
            "score": float(scores[i]),
            "sub_topic": row.get("sub_topic", ""),
            "length": row.get("length", ""),
        })
    return processed, results


def highlight(text, query):
    text = escape(text)
    terms = {t for t in re.findall(r"\w+", query.lower()) if len(t) > 2 and t not in STOP}
    if not terms:
        return text
    pattern = r"\b(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")\b"
    return re.sub(pattern, r'<mark style="background:#ffe066;color:#000;padding:0 2px">\1</mark>',
                  text, flags=re.IGNORECASE)


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="arXiv Hybrid Search", layout="wide")
st.title("arXiv Hybrid Search Engine")
st.caption("cs.CL corpus · BM25 + dense retrieval (all-MiniLM-L6-v2) fused with Reciprocal Rank Fusion")

if not CORPUS.exists():
    st.error("corpus.csv not found. Put it in the same folder as app.py.")
    st.stop()

eng = load_engine()

with st.form("search_form"):
    query = st.text_input("Search", placeholder='Try: NLP, "machine translation", sumarization')
    method = st.radio("Model", ["Hybrid (RRF)", "BM25", "Dense"], horizontal=True)
    submitted = st.form_submit_button("Search", type="primary")

if submitted:
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        processed, results = search(query, method, eng)
        st.markdown(f"**Processed query:** `{processed}` &nbsp;·&nbsp; **Model:** {method}")
        if not results:
            st.info("No papers contain that exact phrase.")
        st.subheader("Top 10 results")
        for rank, r in enumerate(results, 1):
            title = highlight(r["title"], processed)
            snippet = highlight(r["snippet"], processed)
            link = escape(r["link"], quote=True)
            st.html(
                f'<div style="padding:12px 0;border-bottom:1px solid #8884">'
                f'<div style="font-size:1.05rem;font-weight:600">{rank}. {title}</div>'
                f'<a href="{link}" target="_blank">{escape(r["link"])}</a>'
                f'<p style="margin:6px 0">{snippet}...</p>'
                f'<small><b>Ranking score:</b> {r["score"]:.5f} &nbsp;·&nbsp; '
                f'{escape(str(r["sub_topic"]))} &nbsp;·&nbsp; {escape(str(r["length"]))}</small>'
                f'</div>'
            )
