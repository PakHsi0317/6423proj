"""
reranker.py

This module supports re-ranking strategies applied before the generative LLM call.
"""

from typing import Dict, List, Sequence, Tuple
from sentence_transformers import CrossEncoder

# -------------------------- Cross-Encoder Cache --------------------------
_CROSS_ENCODER_CACHE: Dict[str, CrossEncoder] = {}

def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
    """
    Fetch the cached cross-encoder model to prevent reloading on every query.
    """
    if model_name not in _CROSS_ENCODER_CACHE:
        _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODER_CACHE[model_name]


# -------------------------- Reranking Strategies -------------------------
def rerank_with_cross_encoder(query: str, chunks: List[str], top_n: int) -> List[str]:
    """
    Reranks a list of documents using the cross-encoder model.
    """
    if not chunks:
        print("[INSIDE RERANKER] Warning: No chunks to rerank. Returning empty list.")
        return []

    model = get_cross_encoder()

    # Create pairs of [query, chunk] for the model
    pairs = [(query, chunk) for chunk in chunks]

    # Predict the scores
    scores = model.predict(pairs, show_progress_bar=False)

    # Combine chunks with their scores and sort
    chunk_with_scores = list(zip(chunks, scores))
    chunk_with_scores.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _score in chunk_with_scores[:top_n]]


def rerank_candidate_indices(
    query: str,
    candidate_indices: Sequence[int],
    chunks: List[str],
    mode: str,
    top_n: int,
) -> Tuple[List[int], Dict[int, float]]:
    """
    Rerank candidate chunk indices while preserving the mapping back to metadata.
    """
    candidate_indices = list(candidate_indices)
    if not candidate_indices:
        return [], {}

    top_n = min(top_n, len(candidate_indices))
    if mode == "cross_encoder":
        model = get_cross_encoder()
        pairs = [(query, chunks[idx]) for idx in candidate_indices]
        scores = model.predict(pairs, show_progress_bar=False)
        scored_candidates = list(zip(candidate_indices, scores))
        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        reranked = scored_candidates[:top_n]
        return [idx for idx, _score in reranked], {int(idx): float(score) for idx, score in reranked}

    truncated = candidate_indices[:top_n]
    return truncated, {int(idx): 0.0 for idx in truncated}


# -------------------------- Reranking Router -----------------------------
def rerank(query: str, chunks: List[str], mode: str, top_n: int) -> List[str]:
    """
    Routes to the appropriate reranker based on the mode in the config.
    """
    if mode == "cross_encoder":
        return rerank_with_cross_encoder(query, chunks, top_n)

    # We can add other re-ranking strategies to switch between them.
    return chunks
