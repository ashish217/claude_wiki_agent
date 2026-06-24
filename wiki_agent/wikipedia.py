"""Minimal English-Wikipedia retrieval via the MediaWiki Action API.

Design choice (see RATIONALE.md): the assignment fixes the *signature*
``search_wikipedia(query: str)``; the *return payload* is ours to design. We
return the top-k matching articles, each with title, URL, and the plain-text
**introduction** extract. That single round-trip gives the model enough real
content to ground an answer and surfaces disambiguation candidates, without the
token blow-up of fetching full articles.

We use the live API (no key, always fresh) because the assignment says to focus
on prompt + eval quality, not on building a production search system. This also
keeps us clearly within the "no hosted search/RAG tools" constraint — we own
the retrieval loop end to end.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from typing import List

import requests

_API = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks all API clients to send a descriptive User-Agent.
_HEADERS = {
    "User-Agent": "wiki-agent/0.1 (prompt-eng take-home; contact: ashish9@gmail.com)"
}
_EXTRACT_CHAR_CAP = 1200  # keep each extract small to control token cost

_session = requests.Session()
_session.headers.update(_HEADERS)


@dataclass
class WikiResult:
    title: str
    url: str
    extract: str


def search_wikipedia(query: str, k: int = 4, timeout: float = 15.0) -> List[WikiResult]:
    """Return up to ``k`` Wikipedia articles (title, URL, intro extract) for ``query``.

    Uses ``generator=search`` so search + extract happen in one HTTP call.
    Returns an empty list when nothing matches.
    """
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": str(k),
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "exlimit": "max",
        "redirects": "1",
    }
    resp = _session.get(_API, params=params, timeout=timeout)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", [])
    # The generator returns pages in arbitrary dict order; "index" preserves
    # the search ranking.
    pages.sort(key=lambda p: p.get("index", 1_000_000))

    results: List[WikiResult] = []
    for page in pages:
        title = page.get("title", "")
        extract = (page.get("extract") or "").strip()
        if not extract:
            continue
        if len(extract) > _EXTRACT_CHAR_CAP:
            extract = extract[:_EXTRACT_CHAR_CAP].rstrip() + " […]"
        url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        results.append(WikiResult(title=title, url=url, extract=extract))
    return results


def format_results(query: str, results: List[WikiResult]) -> str:
    """Render results into the text block returned to the model as a tool result."""
    if not results:
        return f'No English Wikipedia articles were found for the query "{query}".'
    blocks = [f'Top Wikipedia results for "{query}":\n']
    for i, r in enumerate(results, 1):
        blocks.append(f"[{i}] {r.title} — {r.url}\n{r.extract}")
    return "\n\n".join(blocks)
