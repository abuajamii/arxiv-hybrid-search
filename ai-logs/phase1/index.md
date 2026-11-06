# AI usage disclosure — Phase 1

| Tool | Version | Transcript | What it covers |
|---|---|---|---|
| Claude (Anthropic), claude.ai app | Claude Opus 5.5 | `<PASTE SHARE LINK or file name>` | Environment setup; choice of sub-topics and date range; reviewing the teammate's notebook against the rubric; re-designing the collection to cover all of 2023–2025 (quarterly quotas, ID-month and primary-category filters); the separate Streamlit GUI (`app.py`); highlighted terms and phrase-matching fixes; testing; repository cleanup; this README and the corpus report |
| `<teammate's tool, if any>` | `<version>` | `<link>` | `<first version of the Colab notebook: collection, BM25/dense/RRF, ipywidgets GUI>` |

## How the AI output was checked and corrected

- The first corpus covered only Apr 2025 – Feb 2026, and 17 papers had 2026 IDs despite the 2023–2025 filter. We measured this with an ID-month check and re-collected the corpus. The final check shows 0 papers outside 2023–2025.
- The instructor confirmed by email that the GUI must be a separate UI, so the in-notebook ipywidgets GUI was replaced by `app.py` (Streamlit).
- Phrase matching missed phrases split across PDF line breaks. This was fixed by normalizing whitespace, and verified with the query `"machine translation"`.
- All query-handling features were tested in the running app: `NLP`, `"machine translation"`, `sumarization`, `MACHINE Translation`, plus BM25-only and Dense-only runs.
- An early AI-suggested pipeline used Sentiment Analysis as the fourth sub-topic. It was dropped in favour of the team's Dialogue Systems corpus.
