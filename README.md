# arXiv Hybrid Search — IR Project, Phase 1 (PSUT)

A literature-search engine over 400 arXiv **cs.CL** papers. It combines a sparse retriever (BM25) and a dense retriever (all-MiniLM-L6-v2), fuses their rankings with Reciprocal Rank Fusion, and serves them through a separate web UI built with Streamlit.

## Repository contents

| File | What it is |
|---|---|
| `corpus_collection.ipynb` | Builds the corpus from the arXiv API (run in Google Colab) |
| `corpus.csv` | The corpus: 400 papers with title, full text, arXiv ID/link, sub-topic, pages, length tier, period |
| `collection_info.json` | Exact queries, date range, sampling rule and filters (reproducibility record) |
| `app.py` | Search engine + GUI (BM25, dense retrieval, RRF fusion, query handling) |
| `requirements.txt` | Python packages for `app.py` |
| `report/phase1_report.pdf` | Corpus statistics report |
| `ai-logs/phase1/` | AI usage disclosure and transcripts |

## Run the search engine

Requires Python 3.10–3.12.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. The first start builds the embeddings (1–3 minutes on CPU) and caches them in `embeddings_<hash>.npy`; later starts are fast.

## Rebuild the corpus (optional)

Open `corpus_collection.ipynb` in Google Colab and run all cells. It writes `corpus.csv` and `collection_info.json`. The collection cell resumes from `corpus.csv` if it is interrupted (for example by an arXiv HTTP 429 rate limit); delete `corpus.csv` to start over.

## How the engine works

1. **Documents**: title + first 2,000 characters of each paper's extracted text.
2. **BM25** (`rank_bm25.BM25Okapi`) over lowercased tokens.
3. **Dense**: `all-MiniLM-L6-v2` embeddings, normalized, scored by cosine similarity.
4. **Fusion**: Reciprocal Rank Fusion, `score(d) = 1/(60 + rank_BM25(d)) + 1/(60 + rank_dense(d))`.
5. **Query handling**: lowercasing (case-insensitive), abbreviation expansion (e.g. NLP → natural language processing), spelling correction against the corpus vocabulary (rapidfuzz, similarity ≥ 85), and exact phrase matching for quoted text (phrase matches get a bonus, then results are filtered to papers containing the phrase).
6. **GUI**: search box, model selector (Hybrid RRF / BM25 / Dense), top-10 results with title, arXiv link, ~200-character snippet, highlighted query terms and the ranking score.

## Example queries

`NLP` · `"machine translation"` · `sumarization` · `MACHINE Translation`
