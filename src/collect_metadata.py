"""Step 1: collect arXiv metadata for 4 cs.CL sub-topics."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import arxiv

CATEGORY = "cs.CL"
DATE_FROM = "202301010000"   # 2023-01-01 00:00
DATE_TO   = "202512312359"   # 2025-12-31 23:59
PER_TOPIC = 150              # papers we keep per sub-topic
MAX_FETCH = 400              # how many results we look through per sub-topic

SUBTOPICS = {
    "machine_translation": '"machine translation"',
    "question_answering":  '"question answering"',
    "summarization":       'summarization',
    "sentiment_analysis":  '"sentiment analysis"',
}

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)

papers = []
seen_ids = set()
log = {"category": CATEGORY, "date_from": DATE_FROM, "date_to": DATE_TO,
       "collected_at_utc": datetime.now(timezone.utc).isoformat(), "queries": {}}

for label, phrase in SUBTOPICS.items():
    # Search the phrase in the TITLE only -> clean sub-topic labels
    query = f"cat:{CATEGORY} AND ti:{phrase} AND submittedDate:[{DATE_FROM} TO {DATE_TO}]"
    print(f"\n[{label}] query: {query}")

    search = arxiv.Search(query=query, max_results=MAX_FETCH,
                          sort_by=arxiv.SortCriterion.SubmittedDate)
    kept = skipped_dup = skipped_cat = 0

    for r in client.results(search):
        if kept >= PER_TOPIC:
            break
        if r.primary_category != CATEGORY:      # must truly belong to cs.CL
            skipped_cat += 1
            continue
        arxiv_id = r.get_short_id()
        if arxiv_id in seen_ids:                # already labeled under another sub-topic
            skipped_dup += 1
            continue
        seen_ids.add(arxiv_id)
        papers.append({
            "arxiv_id": arxiv_id,
            "title": " ".join(r.title.split()),
            "abstract": " ".join(r.summary.split()),
            "link": r.entry_id,
            "pdf_url": r.pdf_url,
            "published": r.published.date().isoformat(),
            "subtopic": label,
        })
        kept += 1

    print(f"  kept={kept}  skipped_other_primary_cat={skipped_cat}  skipped_duplicates={skipped_dup}")
    log["queries"][label] = {"query": query, "kept": kept,
                             "skipped_other_primary_cat": skipped_cat,
                             "skipped_duplicates": skipped_dup}
    time.sleep(3)

with open(OUT_DIR / "metadata.jsonl", "w", encoding="utf-8") as f:
    for p in papers:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
with open(OUT_DIR / "collection_log.json", "w", encoding="utf-8") as f:
    json.dump(log, f, indent=2)

print(f"\nTOTAL papers saved: {len(papers)} -> data/raw/metadata.jsonl")