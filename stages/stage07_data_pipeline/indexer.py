"""
Stage 07 (part 2) - Indexer + Scheduler

Goal: connect the scraper (scraper.py) to the dialogue knowledge base (Stage 05 /
knowledge_base.py) - every page that changed gets split into chunks and automatically updated
in the index, with no manual intervention.

This is the "bridge" between the background flow (data updates) and the real-time flow (student
dialogue) shown in ARCHITECTURE.md sections 2 and 3.

TODO (real implementation later):
  - wire this into a real scheduler (cron / APScheduler) instead of a manual loop call.

Note: embedding is already real, not a TODO - KnowledgeBase.upsert() (knowledge_base.py) embeds
every chunk with a local sentence-transformers model as soon as it's indexed here.

Note: chunk_text() is also real now, not a TODO - it splits by paragraph/sentence boundaries
(via a simple regex, same style as Stage 06's SENTENCE_SPLIT_RE) instead of blindly cutting
every ~200 characters mid-word. This matters more now that Stage 07's scraper feeds in real
page content (FAQ Q&A pairs, Subject Highlights tables, multi-paragraph overviews) instead of
one-line mock strings - naive word-count chunking would otherwise slice sentences in half.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

# Lets this file import from stage05 without setting up a full package structure this early
sys.path.append(str(Path(__file__).resolve().parent.parent / "stage05_nlu_dialogue"))

from knowledge_base import KnowledgeBase, KnowledgeChunk  # noqa: E402
from scraper import ScrapedPage, UniversityScraper  # noqa: E402

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_len: int = 400, overlap_sentences: int = 1) -> list[str]:
    """Splits on paragraph breaks first (real pages have real paragraph structure - overview
    text, FAQ answers, etc.), then further splits any paragraph still longer than max_len by
    sentence, accumulating sentences into a chunk until it would exceed max_len. A small
    sentence overlap is carried into the next chunk so a search query that lands near a chunk
    boundary doesn't lose the surrounding context.

    This intentionally still isn't "smart" chunking (no semantic-boundary detection, no
    table-aware handling for Subject Highlights tables) - it's a straightforward improvement
    over the previous blind word-count cut, sized for real programme-page content."""
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= max_len:
            chunks.append(paragraph)
            continue

        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            if current and current_len + len(sentence) + 1 > max_len:
                chunks.append(" ".join(current))
                current = current[-overlap_sentences:] if overlap_sentences else []
                current_len = sum(len(s) + 1 for s in current)
            current.append(sentence)
            current_len += len(sentence) + 1
        if current:
            chunks.append(" ".join(current))

    return chunks or [text]


class UniversityDataIndexer:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def index_changed_pages(self, changed_pages: list[ScrapedPage]) -> int:
        """Splits each changed page into chunks and updates them in the knowledge base. Returns
        the number of chunks updated.

        Collects every chunk first and hands them to the knowledge base in one bulk call, rather
        than upserting each one as it's produced. A real scrape of lincoln.edu.my yields ~14,800
        chunks, and one-at-a-time upsert() meant ~14,800 separate embedding-model calls - see
        KnowledgeBase.upsert_many() for why batching that is a large, no-downside speedup. Note
        these chunks deliberately leave `boost` and `is_direct_answer` at their defaults: scraped
        page text is raw web content, not curated spoken-ready answers (see those fields'
        docstrings in knowledge_base.py)."""
        chunks = [
            KnowledgeChunk(
                chunk_id=f"{page.url}#{i}",
                text=piece,
                source_url=page.url,
                last_updated=time.time(),
            )
            for page in changed_pages
            for i, piece in enumerate(chunk_text(page.raw_text))
        ]
        self.kb.upsert_many(chunks)
        return len(chunks)


def run_update_cycle(
    scraper: UniversityScraper, indexer: UniversityDataIndexer, discover: bool = True
) -> int:
    """
    Represents one run of the periodic scheduler (every N hours) described in
    ARCHITECTURE.md. Returns the number of chunks updated this cycle (0 if the website didn't
    change). Set discover=False to only scrape the seed URLs (faster - useful for a quick check)
    instead of also following each seed page's programme links.
    """
    urls = scraper.discover_urls() if discover else None
    pages = scraper.scrape_once(urls)
    changed = scraper.detect_changes(pages)
    return indexer.index_changed_pages(changed)


if __name__ == "__main__":
    # Real self-test: hits the actual lincoln.edu.my site over the network (see scraper.py's
    # module docstring for why this can't run inside the network-restricted sandbox this was
    # built in). Uses a single real school page to keep the test quick and polite.
    from scraper import REAL_SEED_URLS

    kb = KnowledgeBase()
    scraper = UniversityScraper(
        REAL_SEED_URLS[:1], snapshot_path=Path(__file__).resolve().parent / "snapshot_selftest.json"
    )
    indexer = UniversityDataIndexer(kb)

    updated_1 = run_update_cycle(scraper, indexer, discover=False)
    print(f"Cycle 1: {updated_1} chunk(s) updated (expected > 0, first time)")
    assert updated_1 > 0

    results = kb.search("computer science")
    print(f"Dialogue search after the update: {[r.text[:80] for r in results]}")
    assert results, "The live dialogue must see the updated data immediately"

    updated_2 = run_update_cycle(scraper, indexer, discover=False)  # site unlikely to have
    # changed in the few seconds between these two calls
    print(f"Cycle 2 (no site change expected): {updated_2} chunk(s) updated (expected 0)")
    assert updated_2 == 0

    # Clean up the self-test snapshot so re-running this file always exercises the "brand-new
    # page" path above, rather than depending on leftover state from a prior run.
    Path(__file__).resolve().parent.joinpath("snapshot_selftest.json").unlink(missing_ok=True)

    print("\nOK: the periodic update feeds the dialogue knowledge base automatically from the "
          "real site, without re-indexing when nothing changed.")
