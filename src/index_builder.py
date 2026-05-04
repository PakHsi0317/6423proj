#!/usr/bin/env python3
"""
index_builder.py
PDF -> markdown text -> chunks -> embeddings -> BM25 + FAISS + metadata
"""

import os
import pickle
import pathlib
import re
import json
from typing import List, Dict, Optional, Sequence

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from src.embedder import SentenceTransformer
from src.retrieval_policy import infer_source_space

from src.preprocessing.chunking import DocumentChunker, ChunkConfig, print_chunk_stats
from src.preprocessing.extraction import extract_sections_from_markdown

# ----- runtime parallelism knobs (avoid oversubscription) -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

DEFAULT_EXCLUSION_KEYWORDS = ['questions', 'exercises', 'summary', 'references']


def _normalize_markdown_inputs(
    markdown_file: Optional[str],
    markdown_files: Optional[Sequence[str]],
) -> List[str]:
    inputs: List[str] = []
    if markdown_files:
        inputs.extend(str(path) for path in markdown_files if path)
    if markdown_file:
        inputs.append(str(markdown_file))

    # Preserve order but remove duplicates.
    seen = set()
    normalized: List[str] = []
    for path in inputs:
        if path in seen:
            continue
        seen.add(path)
        normalized.append(path)

    if not normalized:
        raise ValueError("At least one markdown file is required to build the index.")
    return normalized


def build_index(
    markdown_file: Optional[str] = None,
    *,
    markdown_files: Optional[Sequence[str]] = None,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    embedding_model_context_window: int,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    use_multiprocessing: bool = False,
    use_headings: bool = False,
) -> None:
    """
    Extract sections, chunk, embed, and build both FAISS and BM25 indexes.

    Persists:
        - {prefix}.faiss
        - {prefix}_bm25.pkl
        - {prefix}_chunks.pkl
        - {prefix}_sources.pkl
        - {prefix}_meta.pkl
        - {prefix}_page_to_chunk_map.json
    """
    markdown_inputs = _normalize_markdown_inputs(markdown_file, markdown_files)

    all_chunks: List[str] = []
    sources: List[str] = []
    metadata: List[Dict] = []
    page_to_chunk_ids: Dict[int, set[int]] = {}
    page_pattern = re.compile(r'--- Page (\d+) ---')

    for markdown_path in markdown_inputs:
        source_space = infer_source_space(markdown_path)
        sections = extract_sections_from_markdown(
            markdown_path,
            exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS
        )

        current_page = 1
        heading_stack = []

        # Step 1: Chunk
        for c in sections:
            current_level = c.get('level', 1)
            chapter_num = c.get('chapter', 0)

            while heading_stack and heading_stack[-1][0] >= current_level:
                heading_stack.pop()

            if c['heading'] != "Introduction":
                heading_stack.append((current_level, c['heading']))

            path_list = [h[1] for h in heading_stack]
            full_section_path = " ".join(path_list)
            full_section_path = f"Chapter {chapter_num} " + full_section_path

            sub_chunks = chunker.chunk(c['content'])

            for sub_chunk in sub_chunks:
                chunk_pages = set()
                fragments = page_pattern.split(sub_chunk)

                if fragments[0].strip():
                    chunk_pages.add(current_page)

                for idx in range(1, len(fragments), 2):
                    try:
                        new_page = int(fragments[idx]) + 1
                        if fragments[idx + 1].strip():
                            chunk_pages.add(new_page)
                        current_page = new_page
                    except (IndexError, ValueError):
                        continue

                clean_chunk = re.sub(page_pattern, '', sub_chunk).strip()

                if c["heading"] == "Introduction" or not clean_chunk:
                    continue

                chunk_id = len(all_chunks)
                if source_space == "textbook":
                    for page_no in chunk_pages:
                        page_to_chunk_ids.setdefault(page_no, set()).add(chunk_id)

                meta = {
                    "filename": markdown_path,
                    "document_id": pathlib.Path(markdown_path).stem,
                    "mode": chunk_config.to_string(),
                    "char_len": len(clean_chunk),
                    "word_len": len(clean_chunk.split()),
                    "estimated_tokens": max(1, len(clean_chunk) // 4),
                    "section": c['heading'],
                    "section_path": full_section_path,
                    "text_preview": clean_chunk[:100],
                    "page_numbers": sorted(list(chunk_pages)),
                    "chunk_id": chunk_id,
                    "source_space": source_space,
                }

                chunk_prefix = (
                    f"Description: {full_section_path} Content: "
                    if use_headings else ""
                )

                all_chunks.append(chunk_prefix + clean_chunk)
                sources.append(markdown_path)
                metadata.append(meta)

    # Save page-to-chunk map
    final_map = {page: sorted(list(ids)) for page, ids in page_to_chunk_ids.items()}
    output_file = artifacts_dir / f"{index_prefix}_page_to_chunk_map.json"
    with open(output_file, "w") as f:
        json.dump(final_map, f, indent=2)
    print(f"Saved page to chunk ID map: {output_file}")

    # Print chunk stats before embedding - TODO: wrap in some verbose cfg param
    # print_chunk_stats(all_chunks, chunk_size_in_chars=chunk_config.recursive_chunk_size)

    # Step 2: Load embedder
    print(f"Loading embedding model (n_ctx={embedding_model_context_window})...")
    embedder = SentenceTransformer(
        embedding_model_path,
        n_ctx=embedding_model_context_window,
    )
    print(f"Embedding {len(all_chunks):,} chunks sequentially...")

    if use_multiprocessing:
        print("Starting multi-process pool for embeddings...")
        pool = embedder.start_multi_process_pool(workers=4)
        try:
            embeddings = embedder.encode_multi_process(
                all_chunks,
                pool,
                batch_size=4,
            )
        finally:
            embedder.stop_multi_process_pool(pool)
    else:
        embeddings = embedder.encode(
            all_chunks,
            show_progress_bar=True,
        )

    # Step 3: Build FAISS index
    print(f"Building FAISS index for {len(all_chunks):,} chunks...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(artifacts_dir / f"{index_prefix}.faiss"))
    print(f"FAISS index built: {index_prefix}.faiss")

    # Step 4: Build BM25 index
    print(f"Building BM25 index for {len(all_chunks):,} chunks...")
    tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    with open(artifacts_dir / f"{index_prefix}_bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"BM25 index built: {index_prefix}_bm25.pkl")

    # Step 5: Persist remaining artifacts
    with open(artifacts_dir / f"{index_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(artifacts_dir / f"{index_prefix}_sources.pkl", "wb") as f:
        pickle.dump(sources, f)
    with open(artifacts_dir / f"{index_prefix}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved all index artifacts with prefix: {index_prefix}")


def preprocess_for_bm25(text: str) -> list[str]:
    """Lowercase and tokenize text for BM25 indexing."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9_'#+-]", " ", text)
    return text.split()
