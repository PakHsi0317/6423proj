from __future__ import annotations

from math import ceil, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def infer_source_space(source_path: str) -> str:
    """
    Infer a coarse-grained source-space label from the path.
    This keeps the current TokenSmith baseline compatible while giving us
    metadata that can support cross-space retrieval later.
    """
    normalized = Path(source_path).as_posix().lower()

    rules = (
        ("textbook", ("silberschatz", "textbook", "/chapters/", "/books/", "/book/")),
        ("slides", ("slides", "slide", "lecture")),
        ("notes", ("notes", "note")),
        ("tutorial", ("tutorial", "guide", "walkthrough")),
        ("examples", ("example", "examples", "sample", "demo")),
        ("api", ("api", "reference", "documentation", "docs")),
        ("code", (".py", ".java", ".js", ".ts", "/src/", "/code/")),
    )

    for label, markers in rules:
        if any(marker in normalized for marker in markers):
            return label
    return "general"


def estimate_chunk_tokens(text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
    """
    Lightweight token estimate used to budget context for local models.
    We prefer stored metadata when available and fall back to a chars/4 heuristic.
    """
    metadata = metadata or {}

    stored = metadata.get("estimated_tokens")
    if isinstance(stored, int) and stored > 0:
        return stored

    char_len = metadata.get("char_len")
    if isinstance(char_len, int) and char_len > 0:
        return max(1, ceil(char_len / 4))

    word_len = metadata.get("word_len")
    if isinstance(word_len, int) and word_len > 0:
        return max(1, ceil(word_len * 1.3))

    text = (text or "").strip()
    return max(1, ceil(len(text) / 4))


def infer_preferred_source_spaces(query: str, available_spaces: Sequence[str]) -> List[str]:
    """
    Heuristic routing for cross-space retrieval.
    If the requested spaces are not present in the corpus, we fall back to no filter.
    """
    q = query.lower().strip()
    available = {space for space in available_spaces if space}
    if not available:
        return []

    if any(token in q for token in ("implement", "implementation", "code", "syntax", "function", "class", "method", "api", "example")):
        preference_order = ["code", "api", "examples", "tutorial"]
    elif any(token in q for token in ("how", "steps", "procedure", "algorithm", "workflow")):
        preference_order = ["tutorial", "textbook", "examples", "code"]
    elif any(token in q for token in ("what is", "why", "explain", "definition", "compare", "difference", "goal")):
        preference_order = ["textbook", "notes", "slides"]
    else:
        preference_order = []

    preferred = [space for space in preference_order if space in available]
    return preferred


def filter_candidates_by_source_space(
    query: str,
    ordered: Sequence[int],
    metadata: Optional[List[Dict[str, Any]]],
    enabled: bool,
) -> Tuple[List[int], List[str]]:
    """
    Route the candidate list toward the most relevant source spaces when possible.
    """
    if not enabled or not ordered or not metadata:
        return list(ordered), []

    available_spaces = {
        metadata[idx].get("source_space")
        for idx in ordered
        if 0 <= idx < len(metadata) and metadata[idx].get("source_space")
    }
    preferred_spaces = infer_preferred_source_spaces(query, sorted(available_spaces))
    if not preferred_spaces:
        return list(ordered), []

    filtered = [
        idx for idx in ordered
        if 0 <= idx < len(metadata) and metadata[idx].get("source_space") in preferred_spaces
    ]
    if not filtered:
        return list(ordered), []

    return filtered, preferred_spaces


def select_context_indices(
    strategy: str,
    top_k: int,
    token_budget: int,
    ordered: Sequence[int],
    scores: Optional[Sequence[float]],
    chunks: Sequence[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Select the final context chunk indices.
    `top_k` remains as an upper bound so we can compare against the baseline fairly.
    """
    ordered = list(ordered)
    metadata = metadata or []
    if not ordered:
        return [], {
            "strategy": strategy,
            "token_budget": token_budget,
            "selected_token_estimate": 0,
            "candidate_count": 0,
        }

    score_map: Dict[int, float] = {}
    if scores:
        score_map = {
            int(idx): float(score)
            for idx, score in zip(ordered, scores)
        }

    strategy = (strategy or "top_k").lower().strip()
    if strategy == "top_k":
        selected = ordered[:top_k]
        total_tokens = sum(
            estimate_chunk_tokens(chunks[idx], metadata[idx] if idx < len(metadata) else None)
            for idx in selected
        )
        return selected, {
            "strategy": "top_k",
            "token_budget": token_budget,
            "selected_token_estimate": total_tokens,
            "candidate_count": len(ordered),
            "selected_count": len(selected),
        }

    max_score = max(score_map.values(), default=1.0)
    candidates: List[Dict[str, Any]] = []
    for rank, idx in enumerate(ordered):
        meta = metadata[idx] if 0 <= idx < len(metadata) else {}
        token_estimate = estimate_chunk_tokens(chunks[idx], meta)
        raw_score = score_map.get(idx, 1.0 / (rank + 1))
        normalized_score = raw_score / max_score if max_score > 0 else raw_score
        normalized_score = max(0.05, normalized_score)

        section_key = meta.get("section_path") or meta.get("section") or ""
        source_space = meta.get("source_space") or "general"

        utility = normalized_score / max(1.0, sqrt(token_estimate))
        candidates.append({
            "idx": idx,
            "rank": rank,
            "raw_score": raw_score,
            "tokens": token_estimate,
            "section_key": section_key,
            "source_space": source_space,
            "utility": utility,
        })

    prioritized = sorted(
        candidates,
        key=lambda cand: (-cand["utility"], cand["rank"]),
    )

    selected: List[Dict[str, Any]] = []
    used_tokens = 0
    seen_sections = set()
    seen_spaces = set()

    for cand in prioritized:
        if len(selected) >= top_k:
            break

        adjusted_utility = cand["utility"]
        if cand["section_key"] and cand["section_key"] in seen_sections:
            adjusted_utility *= 0.9
        if cand["source_space"] in seen_spaces:
            adjusted_utility *= 0.95

        fits_budget = used_tokens + cand["tokens"] <= token_budget
        if fits_budget and adjusted_utility > 0:
            selected.append(cand)
            used_tokens += cand["tokens"]
            if cand["section_key"]:
                seen_sections.add(cand["section_key"])
            seen_spaces.add(cand["source_space"])

    if len(selected) < top_k:
        selected_ids = {cand["idx"] for cand in selected}
        for cand in sorted(candidates, key=lambda item: item["rank"]):
            if len(selected) >= top_k:
                break
            if cand["idx"] in selected_ids:
                continue
            if used_tokens + cand["tokens"] <= token_budget:
                selected.append(cand)
                selected_ids.add(cand["idx"])
                used_tokens += cand["tokens"]
                if cand["section_key"]:
                    seen_sections.add(cand["section_key"])
                seen_spaces.add(cand["source_space"])

    if not selected:
        best = min(candidates, key=lambda cand: cand["rank"])
        selected = [best]
        used_tokens = best["tokens"]

    selected_sorted = sorted(selected, key=lambda cand: cand["rank"])
    selected_ids = [cand["idx"] for cand in selected_sorted]

    return selected_ids, {
        "strategy": "token_budget",
        "token_budget": token_budget,
        "selected_token_estimate": used_tokens,
        "candidate_count": len(ordered),
        "selected_count": len(selected_ids),
        "dropped_count": max(0, len(ordered) - len(selected_ids)),
    }
