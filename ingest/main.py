#!/usr/bin/env python3
"""
Only embeds new or changed files. Uses LlamaIndex (StorageContext) + Ollama embeddings.
"""
import os
import json
import hashlib
from pathlib import Path
import sys

# LlamaIndex imports
try:
    from llama_index.legacy import (
        VectorStoreIndex,
        StorageContext,
        ServiceContext,
        SimpleDirectoryReader,
        load_index_from_storage,
    )
    from llama_index.legacy.embeddings import OllamaEmbedding
    from llama_index.legacy.llms.ollama import Ollama
except Exception as e:
    print("Error importing LlamaIndex. Make sure package is installed.", e)
    sys.exit(1)

DATA_DIR = Path("recipes")
STORAGE_DIR = Path("storage")
MANIFEST = Path("ingested_files.json")

# Recommended Pi embedding model
EMBEDDING_MODEL = "nomic-embed-text"
MODEL_NAME = "phi3:mini" 


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(manifest: dict):
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def smart_ingest():
    DATA_DIR.mkdir(exist_ok=True)
    STORAGE_DIR.mkdir(exist_ok=True)

    manifest = load_manifest()
    new_files = []

    for p in DATA_DIR.iterdir():
        if not p.is_file():
            continue
        h = file_hash(p)
        if p.name not in manifest or manifest[p.name] != h:
            new_files.append(p)
            manifest[p.name] = h

    if not new_files:
        print("No new or updated files detected.")
        return

    print(f"Found {len(new_files)} new/updated files:")
    for f in new_files:
        print(" -", f.name)

    # Initialize embedding model wrapper for LlamaIndex
    embedding = OllamaEmbedding(model_name=EMBEDDING_MODEL)
    llm = Ollama(model=MODEL_NAME, request_timeout=120.0)
    service_ctx = ServiceContext.from_defaults(llm=llm, embed_model=embedding)
    # Load or create index
    if any(STORAGE_DIR.iterdir()):
        print("Loading existing index from storage...")
        storage_ctx = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
        index = load_index_from_storage(storage_ctx, service_context=service_ctx)
    else:
        print("Creating new index from all files...")
        docs = SimpleDirectoryReader(str(DATA_DIR)).load_data()
        index = VectorStoreIndex.from_documents(docs, service_context=service_ctx)
        index.storage_context.persist()
        save_manifest(manifest)
        print("Index created and persisted.")
        return

    # Ingest only new files
    docs = SimpleDirectoryReader(input_files=new_files).load_data()
    index.insert_nodes(docs)
    index.storage_context.persist()

    save_manifest(manifest)
    print("Index updated with new files.")


if __name__ == '__main__':
    smart_ingest()

