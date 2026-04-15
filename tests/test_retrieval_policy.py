import argparse
from unittest.mock import MagicMock, patch

import pytest

from src.config import RAGConfig
from src.instrumentation.logging import RunLogger
from src.main import get_answer
from src.ranking.ranker import EnsembleRanker
from src.retrieval_policy import filter_candidates_by_source_space, select_context_indices


class MockRetriever:
    def __init__(self, name, scores):
        self.name = name
        self.scores = scores

    def get_scores(self, query, pool_size, chunks):
        return self.scores


def test_select_context_indices_respects_token_budget():
    chunks = [
        "L" * 200,
        "short relevant chunk",
        "another short chunk",
    ]
    metadata = [
        {"estimated_tokens": 50, "section_path": "A", "source_space": "textbook"},
        {"estimated_tokens": 20, "section_path": "B", "source_space": "textbook"},
        {"estimated_tokens": 18, "section_path": "C", "source_space": "textbook"},
    ]

    selected, info = select_context_indices(
        strategy="token_budget",
        top_k=3,
        token_budget=40,
        ordered=[0, 1, 2],
        scores=[1.0, 0.95, 0.6],
        chunks=chunks,
        metadata=metadata,
    )

    assert selected == [1, 2]
    assert info["selected_token_estimate"] <= 40


def test_filter_candidates_by_source_space_prefers_matching_spaces():
    metadata = [
        {"source_space": "textbook"},
        {"source_space": "code"},
        {"source_space": "examples"},
    ]

    filtered, preferred = filter_candidates_by_source_space(
        query="Show a code example for this API",
        ordered=[0, 1, 2],
        metadata=metadata,
        enabled=True,
    )

    assert filtered == [1, 2]
    assert preferred == ["code", "examples"]


@pytest.mark.unit
def test_get_answer_uses_budgeted_context_selection():
    cfg = RAGConfig(
        top_k=3,
        num_candidates=5,
        ensemble_method="linear",
        ranker_weights={"faiss": 1.0},
        chunk_mode="recursive_sections",
        disable_chunks=False,
        rerank_mode="none",
        context_selection_strategy="token_budget",
        context_token_budget=30,
    )
    args = argparse.Namespace(
        system_prompt_mode="baseline",
        index_prefix="test_index",
    )

    chunks = [
        "Chunk 0 " + ("very long " * 40),
        "Chunk 1: compact and relevant.",
        "Chunk 2: also compact and relevant.",
    ]
    sources = ["doc1", "doc1", "doc1"]
    metadata = [
        {"page_numbers": [1], "estimated_tokens": 60, "source_space": "textbook"},
        {"page_numbers": [2], "estimated_tokens": 12, "source_space": "textbook"},
        {"page_numbers": [3], "estimated_tokens": 12, "source_space": "textbook"},
    ]

    retrievers = [MockRetriever("faiss", {0: 0.99, 1: 0.9, 2: 0.8})]
    ranker = EnsembleRanker(
        ensemble_method="linear",
        weights={"faiss": 1.0},
    )
    artifacts = {
        "chunks": chunks,
        "sources": sources,
        "retrievers": retrievers,
        "ranker": ranker,
        "meta": metadata,
    }

    def mock_stream_generator():
        yield "Budgeted selection response."

    with patch("src.main.answer", side_effect=lambda *a, **k: mock_stream_generator()) as mock_answer_func:
        ans, chunks_info, _hyde_query = get_answer(
            question="Explain recovery",
            cfg=cfg,
            args=args,
            logger=RunLogger(),
            console=MagicMock(),
            artifacts=artifacts,
            is_test_mode=True,
        )

        assert ans == "Budgeted selection response."
        assert len(chunks_info) == 2
        assert {info["chunk_id"] for info in chunks_info} == {1, 2}

        passed_chunks = mock_answer_func.call_args[0][1]
        assert len(passed_chunks) == 2
        assert all("compact" in chunk for chunk in passed_chunks)
