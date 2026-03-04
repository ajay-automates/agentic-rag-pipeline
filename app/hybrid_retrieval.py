"""
Hybrid Retrieval — BM25 + Vector Search + Reciprocal Rank Fusion

Combines two retrieval strategies:
1. Vector Search (ChromaDB) — Finds semantically similar chunks
2. BM25 Keyword Search — Finds exact term matches
3. Reciprocal Rank Fusion — Merges results from both strategies

Why hybrid?
- Vector search excels at meaning ("What are the benefits?" matches "PTO and health insurance")
- BM25 excels at exact terms ("SOC 2 Type II", "$2,500/month")
- Together they catch what either alone would miss
"""

import re
import math
from .ingestion import collection, search as vector_search


class BM25:
    """Okapi BM25 ranking function. Lightweight, no external dependencies."""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_ids = []
        self.doc_sources = []
        self.doc_len = []
        self.avgdl = 0
        self.df = {}
        self.idf = {}
        self.n_docs = 0

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s\.\$\%]', ' ', text)
        return [t for t in text.split() if len(t) > 1 or t.isdigit()]

    def fit(self, documents, doc_ids, doc_sources):
        self.corpus = []
        self.doc_ids = doc_ids
        self.doc_sources = doc_sources
        self.n_docs = len(documents)
        self.df = {}
        self.doc_len = []
        for doc in documents:
            tokens = self._tokenize(doc)
            self.corpus.append(tokens)
            self.doc_len.append(len(tokens))
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1
        self.avgdl = sum(self.doc_len) / self.n_docs if self.n_docs > 0 else 1
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1)

    def search(self, query, n_results=5):
        query_tokens = self._tokenize(query)
        scores = []
        for i, doc_tokens in enumerate(self.corpus):
            score = 0
            tf = {}
            for token in doc_tokens:
                tf[token] = tf.get(token, 0) + 1
            for term in query_tokens:
                if term not in tf:
                    continue
                term_freq = tf[term]
                idf = self.idf.get(term, 0)
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                score += idf * (numerator / denominator)
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:n_results]:
            if score > 0:
                results.append({"text": " ".join(self.corpus[idx]), "source": self.doc_sources[idx],
                    "bm25_score": round(score, 4), "retrieval_method": "bm25"})
        return results


bm25_index = BM25()
_bm25_initialized = False


def build_bm25_index():
    global bm25_index, _bm25_initialized
    all_docs = collection.get(include=["documents", "metadatas"])
    if not all_docs or not all_docs["documents"]:
        _bm25_initialized = False
        return
    documents = all_docs["documents"]
    ids = all_docs["ids"]
    sources = [m.get("source", "unknown") for m in all_docs["metadatas"]]
    bm25_index.fit(documents, ids, sources)
    _bm25_initialized = True


def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    """RRF merges ranked lists. Score = sum(1/(k + rank_i)) per ranker."""
    fused_scores = {}
    for rank, doc in enumerate(vector_results):
        doc_key = doc["text"][:100]
        if doc_key not in fused_scores:
            fused_scores[doc_key] = {"text": doc["text"], "source": doc["source"],
                "rrf_score": 0, "vector_rank": None, "bm25_rank": None,
                "vector_score": doc.get("relevance_score", 0), "bm25_score": 0}
        fused_scores[doc_key]["rrf_score"] += 1 / (k + rank + 1)
        fused_scores[doc_key]["vector_rank"] = rank + 1
    for rank, doc in enumerate(bm25_results):
        doc_key = doc["text"][:100]
        if doc_key not in fused_scores:
            fused_scores[doc_key] = {"text": doc["text"], "source": doc["source"],
                "rrf_score": 0, "vector_rank": None, "bm25_rank": None,
                "vector_score": 0, "bm25_score": doc.get("bm25_score", 0)}
        fused_scores[doc_key]["rrf_score"] += 1 / (k + rank + 1)
        fused_scores[doc_key]["bm25_rank"] = rank + 1
        fused_scores[doc_key]["bm25_score"] = doc.get("bm25_score", 0)
    fused = sorted(fused_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    for doc in fused:
        if doc["vector_rank"] and doc["bm25_rank"]:
            doc["retrieval_method"] = "hybrid"
        elif doc["vector_rank"]:
            doc["retrieval_method"] = "vector"
        else:
            doc["retrieval_method"] = "bm25"
    return fused


def hybrid_search(query, n_results=5):
    """Hybrid search: vector (ChromaDB) + BM25 keyword, merged via RRF."""
    global _bm25_initialized
    vector_results = vector_search(query, n_results=n_results)
    if not _bm25_initialized:
        build_bm25_index()
    bm25_results = bm25_index.search(query, n_results=n_results) if _bm25_initialized else []
    if not bm25_results:
        return vector_results
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    return [{"text": d["text"], "source": d["source"], "relevance_score": round(d["rrf_score"], 4),
        "retrieval_method": d["retrieval_method"], "vector_rank": d.get("vector_rank"),
        "bm25_rank": d.get("bm25_rank")} for d in fused[:n_results]]


def rebuild_index():
    global _bm25_initialized
    _bm25_initialized = False
    build_bm25_index()
