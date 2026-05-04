import json
from datetime import datetime
from pathlib import Path

import pytest

from tests.metrics import SimilarityScorer


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_tokensmith_benchmarks(benchmarks, config, results_dir):
    """
    Run all benchmarks through the TokenSmith system.
    """
    scorer = SimilarityScorer(enabled_metrics=config["metrics"])
    print_test_config(config, scorer)

    passed = 0
    failed = 0

    for benchmark in benchmarks:
        result = run_benchmark(benchmark, config, results_dir, scorer)
        if result["passed"]:
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: {passed} passed, {failed} failed")
    print(f"{'=' * 60}\n")


def print_test_config(config, scorer):
    """Print the test configuration in a readable format."""
    active_metrics = list(scorer._get_active_metrics().keys())

    print(f"\n{'=' * 60}")
    print("  TokenSmith Benchmark Configuration")
    print(f"{'=' * 60}")
    print(f"  Generator Model:    {Path(config['gen_model']).name}")
    embed_model = config["embed_model"]
    embed_display = Path(embed_model).name if "/" in embed_model or "\\" in embed_model else embed_model
    print(f"  Embedding Model:    {embed_display}")
    print(f"  System Prompt:      {config['system_prompt_mode']}")
    print(f"  Chunks Enabled:     {not config['disable_chunks']}")
    print(f"  Golden Chunks:      {config['use_golden_chunks']}")
    print(f"  HyDE Enabled:       {config.get('use_hyde', False)}")
    print(f"  Context Strategy:   {config.get('context_selection_strategy', 'top_k')}")
    print(f"  Output Mode:        {config['output_mode']}")
    print(f"  Metrics:            {', '.join(active_metrics)}")
    print(f"{'=' * 60}\n")


def run_benchmark(benchmark, config, results_dir, scorer):
    """
    Run a single benchmark test.
    """
    benchmark_id = benchmark.get("id", "unknown")
    question = benchmark["question"]
    expected_answer = benchmark["expected_answer"]
    keywords = benchmark.get("keywords", [])
    threshold = config["threshold_override"] or benchmark["similarity_threshold"] or 0.6
    golden_chunks = benchmark.get("golden_chunks", None)
    ideal_retrieved_chunks = benchmark.get("ideal_retrieved_chunks", None)

    print(f"\n{'-' * 60}")
    print(f"  Benchmark: {benchmark_id}")
    print(f"  Question: {question}")
    print(f"  Threshold: {threshold}")
    print(f"{'-' * 60}")

    try:
        retrieved_answer, chunks_info, hyde_query = get_tokensmith_answer(
            question=question,
            config=config,
            golden_chunks=golden_chunks if config["use_golden_chunks"] else None,
        )
    except Exception as exc:
        import logging
        import traceback

        error_msg = f"Error running TokenSmith: {exc}"
        print(f"  FAILED: {error_msg}")
        log_failure(results_dir, benchmark_id, error_msg)
        traceback.print_exc()
        logging.exception("Error running TokenSmith")
        return {"passed": False}

    if not retrieved_answer or not retrieved_answer.strip():
        error_msg = f"No answer generated for benchmark '{benchmark_id}'"
        print(f"  FAILED: {error_msg}")
        log_failure(results_dir, benchmark_id, error_msg)
        return {"passed": False}

    try:
        scores = scorer.calculate_scores(
            retrieved_answer,
            expected_answer,
            keywords,
            question=question,
            ideal_retrieved_chunks=ideal_retrieved_chunks,
            actual_retrieved_chunks=chunks_info,
        )
    except Exception as exc:
        error_msg = f"Scoring error: {exc}"
        print(f"  FAILED: {error_msg}")
        log_failure(results_dir, benchmark_id, error_msg)
        return {"passed": False}

    final_score = scores.get("final_score", 0)
    passed = final_score >= threshold
    print_result(benchmark_id, passed, final_score, threshold, scores, config["output_mode"], retrieved_answer)

    result_data = {
        "test_id": benchmark_id,
        "question": question,
        "expected_answer": expected_answer,
        "retrieved_answer": retrieved_answer,
        "keywords": keywords,
        "threshold": threshold,
        "scores": scores,
        "passed": passed,
        "active_metrics": scores.get("active_metrics", []),
        "metric_weights": get_metric_weights(scorer, scores.get("active_metrics", [])),
        "chunks_info": chunks_info if chunks_info else [],
        "hyde_query": hyde_query if hyde_query else None,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "gen_model": config["gen_model"],
            "embed_model": config["embed_model"],
            "system_prompt_mode": config["system_prompt_mode"],
            "disable_chunks": config["disable_chunks"],
            "use_golden_chunks": config["use_golden_chunks"],
            "context_selection_strategy": config.get("context_selection_strategy"),
            "context_token_budget": config.get("context_token_budget"),
            "enable_cross_space_routing": config.get("enable_cross_space_routing"),
        },
    }

    save_result(results_dir, result_data)

    if not passed:
        log_failure(
            results_dir,
            benchmark_id,
            format_failure_message(
                question,
                expected_answer,
                retrieved_answer,
                final_score,
                threshold,
                scores,
            ),
        )

    return result_data


def get_tokensmith_answer(question, config, golden_chunks=None):
    """
    Get answer from TokenSmith system.
    """
    import argparse

    from src.config import RAGConfig
    from src.instrumentation.logging import get_logger
    from src.main import get_answer
    from src.ranking.ranker import EnsembleRanker
    from src.retriever import BM25Retriever, FAISSRetriever, IndexKeywordRetriever, load_artifacts

    args = argparse.Namespace(
        index_prefix=config["index_prefix"],
        gen_model=config.get("gen_model"),
        system_prompt_mode=config.get("system_prompt_mode"),
    )

    cfg = RAGConfig(
        chunk_mode=config.get("chunk_mode", "recursive_sections"),
        chunk_size_in_chars=config.get("chunk_size_in_chars", 2000),
        chunk_overlap=config.get("chunk_overlap", 300),
        top_k=config.get("top_k", 10),
        num_candidates=config.get("num_candidates", 60),
        embed_model=config.get("embed_model"),
        gen_model=config.get("gen_model"),
        ensemble_method=config.get("ensemble_method", config.get("retrieval_method", "rrf")),
        rrf_k=config.get("rrf_k", 60),
        ranker_weights=config.get("ranker_weights", {"faiss": 1, "bm25": 0}),
        rerank_mode=config.get("rerank_mode", "none"),
        rerank_top_k=config.get("rerank_top_k", 5),
        context_selection_strategy=config.get("context_selection_strategy", "token_budget"),
        context_token_budget=config.get("context_token_budget", 1400),
        enable_cross_space_routing=config.get("enable_cross_space_routing", True),
        system_prompt_mode=config.get("system_prompt_mode", "baseline"),
        max_gen_tokens=config.get("max_gen_tokens", 400),
        disable_chunks=config.get("disable_chunks", False),
        use_golden_chunks=config.get("use_golden_chunks", False),
        output_mode=config.get("output_mode", "html"),
        metrics=config.get("metrics", ["all"]),
        use_hyde=config.get("use_hyde", False),
        hyde_max_tokens=config.get("hyde_max_tokens", 300),
        use_indexed_chunks=config.get("use_indexed_chunks", False),
        extracted_index_path=config.get("extracted_index_path", "data/extracted_index.json"),
        page_to_chunk_map_path=config.get("page_to_chunk_map_path", "index/sections/textbook_index_page_to_chunk_map.json"),
    )

    if golden_chunks and config["use_golden_chunks"]:
        print(f"  Using {len(golden_chunks)} golden chunks")
    elif config["disable_chunks"]:
        print("  No chunks (baseline mode)")
    else:
        if config.get("use_hyde", False):
            print("  HyDE enabled - generating hypothetical document...")
        print("  Retrieving chunks...")

    logger = get_logger()

    artifacts_dir = cfg.get_artifacts_directory()
    faiss_index, bm25_index, chunks, sources, metadata = load_artifacts(
        artifacts_dir=artifacts_dir,
        index_prefix=config["index_prefix"],
    )

    retrievers = [
        FAISSRetriever(faiss_index, cfg.embed_model),
        BM25Retriever(bm25_index),
    ]

    if cfg.ranker_weights.get("index_keywords", 0) > 0:
        retrievers.append(
            IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path)
        )

    ranker = EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights=cfg.ranker_weights,
        rrf_k=int(cfg.rrf_k),
    )

    artifacts = {
        "chunks": chunks,
        "sources": sources,
        "retrievers": retrievers,
        "ranker": ranker,
        "metadata": metadata,
    }

    result = get_answer(
        question=question,
        cfg=cfg,
        args=args,
        logger=logger,
        artifacts=artifacts,
        console=None,
        golden_chunks=golden_chunks,
        is_test_mode=True,
    )

    if isinstance(result, tuple):
        generated, chunks_info, hyde_query = result
    else:
        generated, chunks_info, hyde_query = result, None, None

    generated = clean_answer(generated)
    return generated, chunks_info, hyde_query


def clean_answer(text):
    """
    Extract answer up to end token if present.
    """
    end_tokens = [
        "[end of text]",
        "</s>",
        "<|end|>",
        "<|endoftext|>",
        "<|im_end|>",
    ]

    earliest_pos = len(text)
    found_token = None

    for token in end_tokens:
        pos = text.find(token)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
            found_token = token

    if found_token:
        return text[:earliest_pos].strip()

    return text.strip()


def print_result(benchmark_id, passed, final_score, threshold, scores, output_mode, retrieved_answer=None):
    """Print test result based on output mode."""
    if output_mode == "terminal":
        status = "PASSED" if passed else "FAILED"
        print(f"\n  {status}")
        print(f"  Final Score: {final_score:.3f} (threshold: {threshold:.3f})")

        active_metrics = scores.get("active_metrics", [])
        if len(active_metrics) > 1:
            print("  Metric Breakdown:")
            for metric in active_metrics:
                metric_score = scores.get(f"{metric}_similarity", 0)
                print(f"    - {metric:12} : {metric_score:.3f}")

        keywords_matched = scores.get("keywords_matched", 0)
        total_keywords = len(scores.get("keywords", []))
        if total_keywords > 0:
            print(f"    - keywords    : {keywords_matched}/{total_keywords}")

        if retrieved_answer:
            print("\n  Retrieved Answer:")
            print(f"  {'-' * 58}")
            for line in retrieved_answer.split("\n"):
                print(f"  {line}")
            print(f"  {'-' * 58}")
    else:
        status = "PASS" if passed else "FAIL"
        print(f"  {status} Score: {final_score:.3f} (threshold: {threshold:.3f})")


def get_metric_weights(scorer, active_metric_names):
    """Get weights for active metrics."""
    weights = {}
    for name in active_metric_names:
        metric = scorer.registry.get_metric(name)
        if metric:
            weights[name] = metric.weight
    return weights


def save_result(results_dir, result_data):
    """Save benchmark result to JSON file (one result per line)."""
    results_file = results_dir / "benchmark_results.json"
    with open(results_file, "a", encoding="utf-8") as handle:
        json.dump(result_data, handle, indent=None, ensure_ascii=False, default=str)
        handle.write("\n")


def log_failure(results_dir, benchmark_id, message):
    """Log benchmark failure to dedicated log file."""
    failed_log = results_dir / "failed_tests.log"
    with open(failed_log, "a", encoding="utf-8") as handle:
        handle.write(f"\n{'=' * 60}\n")
        handle.write(f"BENCHMARK FAILURE: {benchmark_id}\n")
        handle.write(f"{'=' * 60}\n")
        handle.write(f"{message}\n")
        handle.write(f"{'=' * 60}\n")


def format_failure_message(question, expected, retrieved, final_score, threshold, scores):
    """Create detailed failure message."""
    lines = [
        f"Question: {question}",
        "",
        "Expected Answer:",
        f"{expected}",
        "",
        "Retrieved Answer:",
        f"{retrieved}",
        "",
        f"Final Score: {final_score:.3f} (threshold: {threshold:.3f})",
        f"Active Metrics: {', '.join(scores.get('active_metrics', []))}",
        "",
        "Individual Metric Scores:",
    ]

    for metric in scores.get("active_metrics", []):
        metric_score = scores.get(f"{metric}_similarity", 0)
        lines.append(f"  {metric}: {metric_score:.3f}")

    keywords_matched = scores.get("keywords_matched", 0)
    total_keywords = len(scores.get("keywords", []))
    if total_keywords > 0:
        lines.append(f"  keywords: {keywords_matched}/{total_keywords}")

    return "\n".join(lines)
