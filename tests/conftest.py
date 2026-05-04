import os
import sys
from pathlib import Path

import pytest
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pytest_addoption(parser):
    """Add custom command-line options for testing."""
    group = parser.getgroup("tokensmith", "TokenSmith Testing Options")

    group.addoption(
        "--config",
        default="config/config.yaml",
        help="Path to configuration YAML file (default: config/config.yaml)"
    )
    group.addoption(
        "--output-mode",
        choices=["terminal", "html"],
        default=None,
        help="Output mode: terminal for console output, html for HTML report"
    )

    group.addoption(
        "--gen-model",
        default=None,
        help="Path to generator model (overrides config)"
    )
    group.addoption(
        "--model-path",
        default=None,
        help="Deprecated alias for --gen-model"
    )
    group.addoption(
        "--embed-model",
        default=None,
        help="Path to embedding model (overrides config)"
    )

    group.addoption(
        "--disable-chunks",
        action="store_true",
        default=None,
        help="Disable chunks in the generator prompt"
    )
    group.addoption(
        "--use-golden-chunks",
        action="store_true",
        default=None,
        help="Use golden chunks from benchmarks"
    )
    group.addoption(
        "--system-prompt",
        choices=["baseline", "tutor", "concise", "detailed"],
        default=None,
        help="System prompt mode (overrides config)"
    )

    group.addoption(
        "--artifacts_dir",
        default=None,
        help="Artifacts folder for tests (overrides config)"
    )
    group.addoption(
        "--index-prefix",
        default=None,
        help="Index prefix for tests (overrides config)"
    )
    group.addoption(
        "--benchmark-ids",
        default=None,
        help="Comma-separated list of benchmark IDs to run"
    )
    group.addoption(
        "--metrics",
        action="append",
        dest="metrics_list",
        help="Metrics to use for evaluation"
    )
    group.addoption(
        "--threshold",
        type=float,
        default=None,
        help="Override similarity threshold for all tests"
    )
    group.addoption(
        "--list-metrics",
        action="store_true",
        help="List available metrics and exit"
    )


@pytest.fixture(scope="session")
def config(pytestconfig):
    """
    Load and merge configuration from YAML file and CLI arguments.

    Priority: CLI args > config.yaml
    """
    config_path = Path(pytestconfig.getoption("--config"))
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
    else:
        cfg = {}

    merged_config = {
        "top_k": cfg.get("top_k", 10),
        "num_candidates": cfg.get("num_candidates", cfg.get("pool_size", 60)),
        "ensemble_method": cfg.get("ensemble_method", "rrf"),
        "rrf_k": cfg.get("rrf_k", 60),
        "ranker_weights": cfg.get("ranker_weights", {"faiss": 0.6, "bm25": 0.4}),
        "rerank_mode": cfg.get("rerank_mode", "none"),
        "rerank_top_k": cfg.get("rerank_top_k", 5),
        "context_selection_strategy": cfg.get("context_selection_strategy", "token_budget"),
        "context_token_budget": cfg.get("context_token_budget", 1400),
        "enable_cross_space_routing": cfg.get("enable_cross_space_routing", True),
        "seg_filter": cfg.get("seg_filter", None),
        "chunk_mode": cfg.get("chunk_mode", "recursive_sections"),
        "chunk_size_in_chars": cfg.get("chunk_size_in_chars", cfg.get("recursive_chunk_size", 1000)),
        "chunk_overlap": cfg.get("chunk_overlap", cfg.get("recursive_overlap", 0)),
        "output_mode": pytestconfig.getoption("--output-mode") or cfg.get("output_mode", "terminal"),
        "gen_model": (
            pytestconfig.getoption("--gen-model")
            or pytestconfig.getoption("--model-path")
            or cfg.get("gen_model")
            or cfg.get("model_path", "models/generators/qwen2.5-1.5b-instruct-q5_k_m.gguf")
        ),
        "embed_model": (
            pytestconfig.getoption("--embed-model")
            or cfg.get("embed_model")
            or os.path.join(Path(__file__).parent.parent, "models", "embedders", "Qwen3-Embedding-4B-Q5_K_M.gguf")
        ),
        "system_prompt_mode": pytestconfig.getoption("--system-prompt") or cfg.get("system_prompt_mode", "baseline"),
        "max_gen_tokens": cfg.get("max_gen_tokens", 400),
        "artifacts_dir": pytestconfig.getoption("--artifacts_dir") or "index/tokens-200",
        "index_prefix": pytestconfig.getoption("--index-prefix") or cfg.get("index_prefix", "textbook_index"),
        "metrics": pytestconfig.getoption("metrics_list") or cfg.get("metrics", ["all"]),
        "threshold_override": pytestconfig.getoption("--threshold") or cfg.get("threshold_override", None),
        "use_hyde": cfg.get("use_hyde", False),
        "hyde_max_tokens": cfg.get("hyde_max_tokens", 300),
    }

    disable_chunks_cli = pytestconfig.getoption("--disable-chunks")
    if disable_chunks_cli:
        merged_config["disable_chunks"] = True
    else:
        merged_config["disable_chunks"] = cfg.get("disable_chunks", False)

    use_golden = pytestconfig.getoption("--use-golden-chunks")
    if use_golden is not None:
        merged_config["use_golden_chunks"] = use_golden
    else:
        merged_config["use_golden_chunks"] = cfg.get("use_golden_chunks", False)

    return merged_config


@pytest.fixture(scope="session")
def benchmarks(pytestconfig, config):
    """
    Load benchmark questions from YAML file.

    Optionally filters by benchmark IDs if specified.
    """
    benchmark_file = Path(__file__).parent / "benchmarks.yaml"
    with open(benchmark_file, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    all_benchmarks = data["benchmarks"]
    selected_ids = pytestconfig.getoption("--benchmark-ids")
    if selected_ids:
        id_set = set(item.strip() for item in selected_ids.split(","))
        filtered = [benchmark for benchmark in all_benchmarks if benchmark["id"] in id_set]
        print(f"\nRunning {len(filtered)} selected benchmarks: {', '.join(id_set)}")
        return filtered

    print(f"\nRunning all {len(all_benchmarks)} benchmarks")
    return all_benchmarks


@pytest.fixture(scope="session")
def results_dir():
    """Create and return the results directory."""
    results_path = Path(__file__).parent / "results"
    results_path.mkdir(exist_ok=True)
    return results_path


@pytest.fixture(scope="session", autouse=True)
def setup_results_file(results_dir):
    """Initialize results file (clean previous results)."""
    results_file = results_dir / "benchmark_results.json"
    if results_file.exists():
        results_file.unlink()
    return results_file


def pytest_sessionstart(session):
    """Handle session start - check for list-metrics flag."""
    if session.config.getoption("--list-metrics"):
        from tests.metrics import MetricRegistry

        registry = MetricRegistry()
        available = registry.list_metric_names()
        print(f"\nAvailable metrics: {', '.join(available)}\n")
        pytest.exit("Metric listing complete", returncode=0)


def pytest_sessionfinish(session, exitstatus):
    """Generate report after all tests complete."""
    config = session.config
    config_path = Path(config.getoption("--config"))
    output_mode = config.getoption("--output-mode")

    if not output_mode and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        output_mode = cfg.get("testing", {}).get("output_mode", "html")

    _wait_for_async_grading()

    if output_mode == "html":
        from tests.utils import generate_summary_report

        results_dir = Path(__file__).parent / "results"
        generate_summary_report(results_dir)
    else:
        print("\nTest session complete (terminal output mode)")


def _wait_for_async_grading():
    """Wait for async LLM grading threads to complete."""
    try:
        from tests.metrics.async_llm_judge import wait_for_grading, get_results, save_results

        print("\n" + "=" * 60)
        print("Waiting for async LLM grading to complete...")
        print("=" * 60)

        wait_for_grading(timeout=300)

        results = get_results()
        if results:
            logs_dir = Path(__file__).parent.parent / "logs"
            subdirs = [directory for directory in logs_dir.iterdir() if directory.is_dir()]
            if subdirs:
                log_dir = max(subdirs, key=lambda directory: directory.stat().st_mtime)
                results_file = log_dir / "async_llm_results.json"
                save_results(results_file)
                print(f"Async LLM grading complete: {len(results)} answers graded")
                print(f"Results saved to: {results_file}\n")

    except ImportError:
        pass
    except Exception as exc:
        print(f"Async LLM grading failed: {exc}")
