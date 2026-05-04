import argparse
import json
import pickle
import shutil
import uuid
from pathlib import Path

import numpy as np

from src.config import RAGConfig
from src.index_builder import build_index
from src.main import apply_runtime_overrides


class StubChunker:
    def chunk(self, text):
        return [text]


class StubChunkConfig:
    def to_string(self):
        return "stub"


class StubEmbedder:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, show_progress_bar=False):
        rows = len(texts)
        data = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
        return data


def test_apply_runtime_overrides_updates_gen_model():
    cfg = RAGConfig(gen_model="models/generators/original.gguf")
    args = argparse.Namespace(gen_model="models/generators/override.gguf")

    updated = apply_runtime_overrides(args, cfg)

    assert updated.gen_model == "models/generators/override.gguf"


def test_build_index_supports_multiple_markdown_files(monkeypatch):
    tmp_path = Path("tests") / f"tmp_build_index_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        markdown_files = [
            tmp_path / "silberschatz-textbook.md",
            tmp_path / "study-notes.md",
        ]
        for path in markdown_files:
            path.write_text("# placeholder", encoding="utf-8")

        def fake_extract_sections(markdown_file, exclusion_keywords=None):
            if "silberschatz" in markdown_file:
                return [{
                    "heading": "Recovery",
                    "level": 1,
                    "chapter": 19,
                    "content": "Textbook recovery explanation. --- Page 1 --- More textbook detail.",
                }]
            return [{
                "heading": "Notes",
                "level": 1,
                "chapter": 1,
                "content": "Lecture note summary for recovery.",
            }]

        monkeypatch.setattr("src.index_builder.extract_sections_from_markdown", fake_extract_sections)
        monkeypatch.setattr("src.index_builder.SentenceTransformer", StubEmbedder)

        build_index(
            markdown_files=[str(path) for path in markdown_files],
            chunker=StubChunker(),
            chunk_config=StubChunkConfig(),
            embedding_model_path="models/embedders/fake.gguf",
            embedding_model_context_window=128,
            artifacts_dir=tmp_path,
            index_prefix="multi",
        )

        with open(tmp_path / "multi_sources.pkl", "rb") as handle:
            sources = pickle.load(handle)
        with open(tmp_path / "multi_meta.pkl", "rb") as handle:
            metadata = pickle.load(handle)
        with open(tmp_path / "multi_page_to_chunk_map.json", "r", encoding="utf-8") as handle:
            page_map = json.load(handle)

        assert len(sources) == 2
        assert metadata[0]["source_space"] == "textbook"
        assert metadata[1]["source_space"] == "notes"
        assert page_map == {"1": [0], "2": [0]}
    finally:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
