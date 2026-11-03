import sys
print("Python:", sys.version.split()[0])

import arxiv, pymupdf, pandas, numpy, rank_bm25, symspellpy, streamlit
from sentence_transformers import SentenceTransformer
print("All libraries imported OK")

client = arxiv.Client()
search = arxiv.Search(query="cat:cs.CL", max_results=1)
paper = next(client.results(search))
print("arXiv API OK ->", paper.entry_id, "|", paper.title[:60])

model = SentenceTransformer("all-MiniLM-L6-v2")
vec = model.encode("hybrid retrieval test")
print("Embedding model OK -> vector length:", len(vec))