#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import RAGConfig
from src.instrumentation.logging import RunLogger
from src.main import get_answer
from src.ranking.ranker import EnsembleRanker
from src.retriever import BM25Retriever, FAISSRetriever, IndexKeywordRetriever, load_artifacts
from tests.metrics.scorer import SimilarityScorer


VARIANTS = ("baseline", "cross_space", "token_budget", "both")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TokenSmith baseline and enhancement variants."
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--benchmark-file", default="tests/benchmarks.yaml", help="Path to benchmark YAML.")
    parser.add_argument("--index-prefix", default="textbook_index", help="Index prefix to load.")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
        help="Variants to evaluate.",
    )
    parser.add_argument(
        "--benchmark-ids",
        nargs="*",
        default=None,
        help="Optional subset of benchmark ids to run.",
    )
    parser.add_argument(
        "--mode",
        choices=("retrieval", "full"),
        default="retrieval",
        help="Retrieval mode skips generation and focuses on retrieval/context metrics.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["keyword", "chunk_retrieval"],
        help="Metrics to use in full mode.",
    )
    parser.add_argument(
        "--rerank-mode",
        default="none",
        help="Reranker mode to use during comparison. Default is none for reproducibility.",
    )
    parser.add_argument(
        "--output",
        default="tests/results/variant_comparison.json",
        help="Where to write the JSON summary.",
    )
    return parser.parse_args()


def load_benchmarks(path: str, selected_ids: List[str] | None) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        all_benchmarks = yaml.safe_load(handle)["benchmarks"]
    if not selected_ids:
        return all_benchmarks
    selected = set(selected_ids)
    return [benchmark for benchmark in all_benchmarks if benchmark["id"] in selected]


def build_variant_cfg(base_cfg: RAGConfig, variant: str, rerank_mode: str) -> RAGConfig:
    cfg = RAGConfig(**base_cfg.get_config_state())
    cfg.enable_history = False
    cfg.use_hyde = False
    cfg.rerank_mode = rerank_mode

    if variant == "baseline":
        cfg.context_selection_strategy = "top_k"
        cfg.enable_cross_space_routing = False
    elif variant == "cross_space":
        cfg.context_selection_strategy = "top_k"
        cfg.enable_cross_space_routing = True
    elif variant == "token_budget":
        cfg.context_selection_strategy = "token_budget"
        cfg.enable_cross_space_routing = False
    elif variant == "both":
        cfg.context_selection_strategy = "token_budget"
        cfg.enable_cross_space_routing = True
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return cfg


def load_artifacts_bundle(cfg: RAGConfig, index_prefix: str) -> Dict[str, Any]:
    artifacts_dir = cfg.get_artifacts_directory()
    faiss_index, bm25_index, chunks, sources, metadata = load_artifacts(
        artifacts_dir=artifacts_dir,
        index_prefix=index_prefix,
    )
    retrievers = [
        FAISSRetriever(faiss_index, cfg.embed_model),
        BM25Retriever(bm25_index),
    ]
    if cfg.ranker_weights.get("index_keywords", 0) > 0:
        retrievers.append(IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path))
    ranker = EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights=cfg.ranker_weights,
        rrf_k=int(cfg.rrf_k),
    )
    return {
        "chunks": chunks,
        "sources": sources,
        "retrievers": retrievers,
        "ranker": ranker,
        "meta": metadata,
    }


def summarize_variant(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    total = len(results)
    return {
        "num_questions": total,
        "avg_context_tokens": sum(item["context_tokens"] for item in results) / total,
        "avg_selected_chunks": sum(item["selected_chunks"] for item in results) / total,
        "avg_chunk_hit_count": sum(item["chunk_hit_count"] for item in results) / total,
        "avg_chunk_recall": sum(item["chunk_recall"] for item in results) / total,
        "avg_keyword_hits": sum(item["keyword_hits"] for item in results) / total,
        "avg_answer_score": sum(item.get("answer_score", 0.0) for item in results) / total,
    }


def main() -> None:
    args = parse_args()
    base_cfg = RAGConfig.from_yaml(args.config)
    benchmarks = load_benchmarks(args.benchmark_file, args.benchmark_ids)
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scorer = SimilarityScorer(enabled_metrics=args.metrics) if args.mode == "full" else None
    logger = RunLogger()
    common_args = argparse.Namespace(
        index_prefix=args.index_prefix,
        gen_model=base_cfg.gen_model,
        system_prompt_mode=base_cfg.system_prompt_mode,
        double_prompt=False,
    )

    all_results: Dict[str, Any] = {
        "config": {
            "source_config": args.config,
            "benchmark_file": args.benchmark_file,
            "index_prefix": args.index_prefix,
            "mode": args.mode,
            "rerank_mode": args.rerank_mode,
            "metrics": args.metrics if scorer else [],
        },
        "variants": {},
    }

    for variant in args.variants:
        cfg = build_variant_cfg(base_cfg, variant, args.rerank_mode)
        artifacts = load_artifacts_bundle(cfg, args.index_prefix)
        per_question: List[Dict[str, Any]] = []

        patcher = patch("src.main.answer", side_effect=lambda *a, **k: iter([""]))
        context = patcher if args.mode == "retrieval" else nullcontext()
        with context:
            for benchmark in benchmarks:
                result = get_answer(
                    question=benchmark["question"],
                    cfg=cfg,
                    args=common_args,
                    logger=logger,
                    console=None,
                    artifacts=artifacts,
                    is_test_mode=True,
                )
                answer_text, chunks_info, _hyde_query = result if isinstance(result, tuple) else (result, [], None)
                chunks_info = chunks_info or []
                chunk_ids = [item["chunk_id"] for item in chunks_info]
                ideal_ids = benchmark.get("ideal_retrieved_chunks", []) or []
                hit_count = sum(1 for chunk_id in chunk_ids if chunk_id in ideal_ids)
                answer_score = 0.0
                if scorer:
                    scores = scorer.calculate_scores(
                        answer=answer_text,
                        expected=benchmark["expected_answer"],
                        keywords=benchmark.get("keywords", []),
                        question=benchmark["question"],
                        ideal_retrieved_chunks=ideal_ids,
                        actual_retrieved_chunks=chunks_info,
                    )
                    answer_score = scores.get("final_score", 0.0)

                per_question.append({
                    "benchmark_id": benchmark["id"],
                    "question": benchmark["question"],
                    "selected_chunks": len(chunks_info),
                    "selected_chunk_ids": chunk_ids,
                    "context_tokens": sum((item.get("token_estimate") or 0) for item in chunks_info),
                    "chunk_hit_count": hit_count,
                    "chunk_recall": hit_count / len(ideal_ids) if ideal_ids else 0.0,
                    "keyword_hits": sum(
                        1 for keyword in benchmark.get("keywords", [])
                        if keyword.lower() in answer_text.lower()
                    ),
                    "answer_score": answer_score,
                })

        all_results["variants"][variant] = {
            "summary": summarize_variant(per_question),
            "questions": per_question,
        }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, ensure_ascii=False)

    print(f"Saved variant comparison to {output_path}")
    for variant in args.variants:
        summary = all_results["variants"][variant]["summary"]
        print(
            f"{variant}: avg_context_tokens={summary.get('avg_context_tokens', 0):.1f}, "
            f"avg_chunk_recall={summary.get('avg_chunk_recall', 0):.3f}, "
            f"avg_answer_score={summary.get('avg_answer_score', 0):.3f}"
        )


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


if __name__ == "__main__":
    main()
