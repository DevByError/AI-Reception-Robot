"""
Stage 07 (part 1) - University Website Scraper

Goal: periodically crawl the university's public pages, extract structured data (text +
source), and compare it against the last saved snapshot to know if anything changed (diff) - so
we don't re-index everything every time.

Real target: Lincoln University College, https://www.lincoln.edu.my/ (see the "project
university name" memory note for how this was confirmed - the real site itself is the source of
truth for the spelling "College").

Real site structure (confirmed by fetching the actual site on 2026-08-12):
  - Homepage nav -> ~16 school/centre pages, e.g. https://www.lincoln.edu.my/school-of-ai-computing-multimedia/
  - Each school page has a "Programmes" section linking to individual programme pages, e.g.
    https://www.lincoln.edu.my/phd-in-information-technology/
  - Programme pages hold the real content worth indexing: Duration, Intakes, Programme
    Overview, Subject Highlights, Entry Requirements, Career Opportunities, Program Aim, FAQ.
  - robots.txt (https://www.lincoln.edu.my/robots.txt) only disallows /wp-admin/ - permissive.
  - It's WordPress + Elementor. Elementor doesn't always use semantic <nav>/<main> tags, so
    extract_text() below uses heuristics (documented inline) rather than exact CSS selectors -
    this couldn't be verified against raw HTML/class names during development (the sandbox this
    was built in can't reach lincoln.edu.my directly; only markdown-rendered fetches worked).
    Run this for real and check the printed output - if boilerplate (nav labels, "Lorem ipsum"
    testimonial placeholders, etc.) is leaking into the extracted text, that's the place to
    tighten with real class names once you can inspect them.

Not a TODO anymore (previously was):
  - fetch_page() makes a real HTTP request and respects robots.txt.
  - extract_text() uses BeautifulSoup with content heuristics.
  - snapshots persist to disk (snapshot.json next to this file), not just in memory - so change
    detection survives the process restarting between scheduled runs.

Still a TODO:
  - extract_text()'s boilerplate-stripping heuristics are unverified against the real site's
    actual HTML/CSS - refine once you can see what's leaking through in a real run.
  - discover_programme_urls()'s link-selection heuristic (same root cause: no real class names
    to anchor on) may pick up a few stray non-programme links - spot check its output.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "AI-Reception-Robot-Scraper/1.0 (university reception robot; "
    "+https://github.com/DevByError/AI-Reception-Robot)"
)
REQUEST_TIMEOUT_S = 15
REQUEST_DELAY_S = 1.0  # polite delay between requests - robots.txt has no Crawl-delay, but
                        # hammering a small institutional site every few hours is bad practice
                        # regardless of what's technically allowed.

SNAPSHOT_PATH = Path(__file__).resolve().parent / "snapshot.json"

# Tags that never contain real page content on a typical WordPress/Elementor site.
_JUNK_TAGS = ["script", "style", "nav", "header", "footer", "form", "aside", "noscript", "iframe"]

# A standalone line shorter than this is more likely a nav label, button, or breadcrumb than a
# real sentence of content - filtered out of extract_text()'s output. Headings are short too, so
# this is a heuristic trade-off, not a precise rule.
_MIN_LINE_WORDS = 3

_robot_parser_cache: dict[str, urllib.robotparser.RobotFileParser] = {}


def _allowed_by_robots(url: str) -> bool:
    """Fail safe: if robots.txt can't be read at all, treat the URL as disallowed rather than
    scraping blind."""
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin not in _robot_parser_cache:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(urljoin(origin, "/robots.txt"))
        try:
            rp.read()
        except Exception:
            return False
        _robot_parser_cache[origin] = rp
    return _robot_parser_cache[origin].can_fetch(USER_AGENT, url)


def fetch_page(url: str) -> str:
    """Real implementation: fetches actual HTML over HTTP, respecting robots.txt. Returns "" on
    any failure instead of raising - an unattended periodic job shouldn't crash the whole cycle
    because one page 404'd or timed out; detect_changes() naturally treats empty/unchanged
    content as nothing to index."""
    if not _allowed_by_robots(url):
        print(f"[scraper] robots.txt disallows {url} - skipping")
        return ""
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_S)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        print(f"[scraper] failed to fetch {url}: {exc}")
        return ""


def extract_text(html: str) -> str:
    """Real implementation: BeautifulSoup-based extraction, tuned for a WordPress/Elementor
    site (see module docstring for what was actually observed on lincoln.edu.my). Strips obvious
    non-content tags, prefers a <main>/<article>/#content container if one exists, then drops
    short standalone lines that look like nav/button labels rather than real sentences."""
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_JUNK_TAGS):
        tag.decompose()

    container = soup.find("main") or soup.find("article") or soup.find(id="content") or soup.body
    if container is None:
        container = soup

    lines = [line.strip() for line in container.get_text(separator="\n").splitlines()]
    kept = []
    for line in lines:
        if not line:
            continue
        if len(line.split()) < _MIN_LINE_WORDS:
            continue
        kept.append(line)

    # Collapse consecutive duplicate lines (Elementor sometimes repeats a heading in both a
    # visual heading element and an aria-label/accessibility copy).
    deduped = []
    for line in kept:
        if not deduped or deduped[-1] != line:
            deduped.append(line)

    return "\n".join(deduped)


def extract_link_urls(html: str, base_url: str, same_domain_only: bool = True) -> list[str]:
    """Pulls every same-domain link out of the page's real content (after junk-tag removal),
    used by discover_programme_urls() to find a school page's programme links without needing a
    hand-maintained list. Heuristic, not exact - see module docstring."""
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_JUNK_TAGS):
        tag.decompose()

    base_domain = urlparse(base_url).netloc
    # Normalize every discovered link to the seed page's own scheme (https, in practice) - a
    # real run against lincoln.edu.my turned up at least one internal link the site itself
    # authored as "http://" while every other link on the same page is "https://" (e.g.
    # /master-in-financial-technology/). Without this, the same real page could get scraped
    # under two different URLs (http and https variants) and look like two separate sources.
    base_scheme = urlparse(base_url).scheme
    urls: list[str] = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if same_domain_only and parsed.netloc != base_domain:
            continue
        # Skip obvious non-page assets (PDFs/brochures, images, WP admin/login).
        if any(parsed.path.lower().endswith(ext) for ext in (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip")):
            continue
        if "/wp-admin" in parsed.path or "/wp-login" in parsed.path:
            continue
        clean = f"{base_scheme}://{parsed.netloc}{parsed.path}"
        if clean not in seen:
            seen.add(clean)
            urls.append(clean)
    return urls


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ScrapedPage:
    url: str
    raw_text: str
    content_hash: str


class UniversityScraper:
    def __init__(self, seed_urls: list[str], snapshot_path: Path = SNAPSHOT_PATH):
        """seed_urls: the pages to always scrape directly (school/centre pages + a few general
        pages). Programme pages don't need to be listed here - discover_urls() finds them by
        following each seed page's real links."""
        self.seed_urls = seed_urls
        self.snapshot_path = snapshot_path
        self._last_hashes: dict[str, str] = self._load_snapshot()

    def _load_snapshot(self) -> dict[str, str]:
        if self.snapshot_path.exists():
            try:
                return json.loads(self.snapshot_path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_snapshot(self) -> None:
        self.snapshot_path.write_text(json.dumps(self._last_hashes, indent=2))

    def discover_urls(self) -> list[str]:
        """Fetches each seed (school/centre) page and pulls real programme links out of it, so
        the full site of ~150+ programme pages doesn't need to be hand-typed - it's discovered
        the same way a real visitor would find it, by following the school pages' own links.
        Falls back to just the seed URLs if a seed page can't be fetched."""
        all_urls = list(self.seed_urls)
        seen = set(all_urls)
        for seed_url in self.seed_urls:
            html = fetch_page(seed_url)
            time.sleep(REQUEST_DELAY_S)
            if not html:
                continue
            for link in extract_link_urls(html, base_url=seed_url):
                # Heuristic filter: a school page's real "Programmes" links are page-like paths
                # (e.g. /phd-in-information-technology/), not the school pages themselves or
                # generic site sections. Exact class-based selection wasn't possible to verify
                # (see module docstring) so this keeps any same-domain content link that isn't
                # already a known seed and isn't an obvious non-programme section.
                if link in seen:
                    continue
                path = urlparse(link).path.strip("/")
                if not path or path in ("about-us", "contact-us", "news", "events", "blog"):
                    continue
                seen.add(link)
                all_urls.append(link)
        return all_urls

    def scrape_once(self, urls: list[str] | None = None,
                    before_each_page=None) -> list[ScrapedPage]:
        """Returns every page (regardless of whether it changed) - filtering happens in
        detect_changes(). Pass urls explicitly (e.g. the output of discover_urls()) to scrape
        more than just the seed list; defaults to the seed list alone.

        before_each_page: an optional callback invoked before each fetch. A full scrape is ~150
        rate-limited pages and runs for about a minute, so a caller that shares a machine with
        something latency-sensitive needs a way to get a word in partway through rather than
        only before it starts. Stage 08 passes one that blocks while a visitor is being served -
        scraping and parsing in the background is what starved its microphone callback and left
        a visitor's question transcribing to nothing. Defaults to None, which is the original
        uninterruptible behaviour.
        """
        pages = []
        for url in urls if urls is not None else self.seed_urls:
            if before_each_page is not None:
                before_each_page()
            html = fetch_page(url)
            text = extract_text(html)
            pages.append(ScrapedPage(url=url, raw_text=text, content_hash=compute_hash(text)))
            time.sleep(REQUEST_DELAY_S)
        return pages

    def detect_changes(self, pages: list[ScrapedPage]) -> list[ScrapedPage]:
        """Returns only the pages whose content changed since last time (or brand-new pages).
        Hashes persist to disk (snapshot.json) so this still works correctly across process
        restarts between scheduled runs, not just within one run."""
        changed = []
        for page in pages:
            if not page.raw_text:
                continue  # fetch/extract failed - don't treat "nothing extracted" as a change
            if self._last_hashes.get(page.url) != page.content_hash:
                changed.append(page)
                self._last_hashes[page.url] = page.content_hash
        self._save_snapshot()
        return changed


# Seed list: the real school/centre pages discovered from the lincoln.edu.my homepage nav on
# 2026-08-12. A few schools have inconsistent URL slugs on the real site (e.g. some pages link a
# shorter alias for the same school) - discover_urls() follows whatever links each page actually
# has rather than requiring this list to be exhaustive or canonical, so a missed/aliased school
# page here just means its programmes are found one hop later than ideal, not missed entirely.
REAL_SEED_URLS = [
    "https://www.lincoln.edu.my/school-of-ai-computing-multimedia/",
    "https://www.lincoln.edu.my/school-of-business-management/",
    "https://www.lincoln.edu.my/school-of-nursing/",
    "https://www.lincoln.edu.my/school-of-engineering/",
    "https://www.lincoln.edu.my/school-of-social-science/",
]


if __name__ == "__main__":
    # Real self-test: hits the actual lincoln.edu.my site over the network. Run this on a
    # machine with normal internet access (not inside a network-restricted sandbox - see the
    # module docstring). Uses a small subset of REAL_SEED_URLS to keep the test quick and
    # polite; scrape_once()/discover_urls() work the same way at full scale.
    test_seed = REAL_SEED_URLS[:1]
    scraper = UniversityScraper(test_seed, snapshot_path=Path(__file__).resolve().parent / "snapshot_selftest.json")

    print(f"Fetching seed page: {test_seed[0]}")
    pages_v1 = scraper.scrape_once(test_seed)
    print(f"Extracted {len(pages_v1[0].raw_text)} chars of text (first 300 shown):\n")
    print(pages_v1[0].raw_text[:300])

    changed_v1 = scraper.detect_changes(pages_v1)
    print(f"\nFirst run: {len(changed_v1)} page(s) changed (expected: 1 - new)")
    assert len(changed_v1) == 1, "a brand-new page should always count as changed"

    pages_v1_again = scraper.scrape_once(test_seed)
    changed_none = scraper.detect_changes(pages_v1_again)
    print(f"Second run (same content expected): {len(changed_none)} page(s) changed (expected: 0)")
    assert len(changed_none) == 0, "unchanged content shouldn't be reported as changed"

    print("\nDiscovering programme links from the seed page...")
    discovered = scraper.discover_urls()
    print(f"Found {len(discovered)} total URLs ({len(discovered) - len(test_seed)} discovered "
          f"programme/sub-page links beyond the seed):")
    for url in discovered[:15]:
        print(f"  {url}")
    if len(discovered) > 15:
        print(f"  ... and {len(discovered) - 15} more")

    # Clean up the self-test snapshot so re-running this file always exercises the "brand-new
    # page" path above, rather than silently depending on leftover state from a prior run.
    Path(__file__).resolve().parent.joinpath("snapshot_selftest.json").unlink(missing_ok=True)

    print("\nOK: real fetch + extract + change detection + link discovery all worked end to "
          "end against the live site. If the extracted text above has visible nav/boilerplate "
          "noise, tighten extract_text()'s heuristics once you can inspect the real HTML "
          "classes directly (see module docstring).")
