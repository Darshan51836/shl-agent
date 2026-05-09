"""
STEP 4: Retrieval module — used at runtime by the FastAPI agent
Loaded once at startup; queried on every /chat call.

Usage:
    from retriever import CatalogRetriever
    retriever = CatalogRetriever()                   # loads index + model
    results = retriever.search("Java developer", k=10)
"""
import json
import numpy as np
from typing import Optional


class CatalogRetriever:
    """
    Thin wrapper around a FAISS index + metadata list.
    Thread-safe for reading (no writes after init).
    """

    def __init__(
        self,
        index_path: str = "data/catalog_index.faiss",
        meta_path:  str = "data/catalog_index_meta.json",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError("pip install sentence-transformers faiss-cpu --break-system-packages")

        self._index = faiss.read_index(index_path)
        with open(meta_path) as f:
            self._meta: list[dict] = json.load(f)
        self._model = SentenceTransformer(model_name)
        print(f"[Retriever] Loaded {self._index.ntotal} vectors, model={model_name}")

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 10,
        filter_job_level: Optional[str] = None,
        filter_test_type: Optional[str] = None,
        filter_language:  Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search over the catalog.
        Returns up to k items, each dict includes name/url/test_type/description.
        Filters are applied post-retrieval (over-fetch then trim).
        """
        # Over-fetch so filters have material to work with
        fetch_k = min(k * 5, self._index.ntotal)
        vec = self._model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self._index.search(vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            item = self._meta[idx]

            # ── Optional hard filters ─────────────────────────────────────────
            if filter_job_level:
                lvl_lower = filter_job_level.lower()
                if not any(lvl_lower in jl.lower() for jl in item.get("job_levels", [])):
                    continue
            if filter_test_type:
                if filter_test_type.upper() not in item.get("test_type", ""):
                    continue
            if filter_language:
                lang_lower = filter_language.lower()
                if not any(lang_lower in lang.lower() for lang in item.get("languages", [])):
                    continue

            results.append({**item, "_score": float(score)})
            if len(results) >= k:
                break

        return results

    def get_by_name(self, name: str) -> Optional[dict]:
        """Exact (case-insensitive) lookup by assessment name."""
        name_lower = name.lower().strip()
        for item in self._meta:
            if item["name"].lower().strip() == name_lower:
                return item
        # Fuzzy fallback — substring
        for item in self._meta:
            if name_lower in item["name"].lower():
                return item
        return None

    def get_by_names(self, names: list[str]) -> list[dict]:
        """Look up multiple assessments by name."""
        return [r for name in names if (r := self.get_by_name(name)) is not None]

    def format_for_api(self, items: list[dict]) -> list[dict]:
        """
        Convert retrieval results → the API response schema:
        [{"name": ..., "url": ..., "test_type": ...}, ...]
        """
        return [
            {
                "name":      item["name"],
                "url":       item["url"],
                "test_type": item.get("test_type", "K"),
            }
            for item in items
        ]


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "senior Java backend developer microservices"
    retriever = CatalogRetriever()
    results = retriever.search(query, k=5)
    print(f"\nTop results for: '{query}'")
    for r in results:
        print(f"  [{r['_score']:.3f}] {r['name']}  ({r['test_type']})  {r['url']}")
