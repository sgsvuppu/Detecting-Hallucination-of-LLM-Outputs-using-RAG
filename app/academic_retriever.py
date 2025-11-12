"""
academic_retriever.py
Lightweight Semantic Scholar client for academic evidence retrieval and credibility scoring.
Author: Sai Srivastav
"""

import math
import time
from typing import Dict, List, Optional
import requests

# Base API configuration
S2_HOST = "https://api.semanticscholar.org/graph/v1"
FIELDS = (
    "paperId,title,abstract,authors.name,year,venue,externalIds,url,"
    "openAccessPdf,citationCount,isOpenAccess"
)


class SemanticScholar:
    """
    Simple wrapper for the Semantic Scholar public API (no API key required).

    Note: this API is rate-limited (~100 requests per 5 minutes).
    Use `sleep` to stay under the threshold.
    """

    def __init__(self, sleep: float = 0.25, timeout: int = 20):
        self.sleep = sleep
        self.timeout = timeout

    def search(
        self,
        query: str,
        limit: int = 10,
        year_from: Optional[int] = None,
    ) -> List[Dict]:
        """
        Search for research papers matching the query string.

        Args:
            query: Search text (e.g., "climate change impacts").
            limit: Max number of papers to return.
            year_from: Optionally restrict to more recent papers.

        Returns:
            List of dictionaries containing title, abstract, year, venue, etc.
        """
        params = {"query": query, "limit": limit, "fields": FIELDS}
        if year_from:
            params["year"] = f">{year_from}"

        url = f"{S2_HOST}/paper/search"
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json() or {}
        except Exception as e:
            print(f"[SemanticScholar.search] Error: {e}")
            return []

        time.sleep(self.sleep)
        return data.get("data", [])

    def details(self, paper_id: str) -> Dict:
        """
        Fetch details for a given paper ID.

        Args:
            paper_id: Semantic Scholar paper identifier.
        """
        url = f"{S2_HOST}/paper/{paper_id}"
        try:
            resp = requests.get(url, params={"fields": FIELDS}, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json() or {}
        except Exception as e:
            print(f"[SemanticScholar.details] Error: {e}")
            return {}
        time.sleep(self.sleep)
        return data


def credibility_score(p: Dict, query_similarity: float = 0.0) -> float:
    """
    Compute a heuristic 0–1 credibility score for a paper.

    Combines:
      - citation impact (log-scaled)
      - publication recency
      - venue reputation
      - query similarity (semantic relevance)
      - open access bonus

    Args:
        p: Paper dictionary from Semantic Scholar.
        query_similarity: Cosine similarity to the user query.

    Returns:
        A float score between 0.0 and 1.0.
    """
    year = p.get("year") or 0
    citations = max(0, int(p.get("citationCount") or 0))
    venue = (p.get("venue") or "").lower()
    is_open = bool(p.get("isOpenAccess"))

    # Log-scaled citation impact (log10 normalized)
    c_cit = min(1.0, math.log10(citations + 1) / 3.0)

    # Recency component: full credit for ~past decade
    c_year = 0.0
    if year:
        c_year = max(0.0, min(1.0, (year - 2012) / 12.0))

    # Venue reputation
    top_venues = {
        "nature", "science", "cell", "neurips", "icml", "iclr",
        "acl", "emnlp", "cvpr", "sigir", "kdd", "www"
    }
    c_venue = 0.7 if any(v in venue for v in top_venues) else (0.5 if venue else 0.3)

    # Open-access bonus (minor)
    c_open = 0.05 if is_open else 0.0

    # Query relevance (from semantic similarity)
    c_match = max(0.0, min(1.0, query_similarity))

    # Weighted combination
    score = (
        0.40 * c_cit +
        0.25 * c_year +
        0.20 * c_match +
        0.10 * c_venue +
        c_open
    )
    return round(min(1.0, score), 3)
