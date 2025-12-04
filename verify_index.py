#!/usr/bin/env python3
"""
Verify and inspect the stored index to ensure it was created correctly.
"""
import json
from pathlib import Path

try:
    from llama_index.legacy import (
        StorageContext,
        ServiceContext,
        load_index_from_storage,
    )
    from llama_index.legacy.embeddings import OllamaEmbedding
    from llama_index.legacy.llms.ollama import Ollama
except Exception as e:
    print("Error importing LlamaIndex:", e)
    exit(1)

STORAGE_DIR = Path("storage")
MANIFEST = Path("ingested_files.json")
EMBEDDING_MODEL = "nomic-embed-text"
MODEL_NAME = "phi3"


def verify_index():
    if not STORAGE_DIR.exists():
        print("❌ Storage directory does not exist!")
        return
    
    if not list(STORAGE_DIR.iterdir()):
        print("❌ Storage directory is empty!")
        return
    
    print("✓ Storage directory exists and contains files")
    print(f"  Files in storage: {list(STORAGE_DIR.iterdir())}\n")
    
    # Load manifest
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        print(f"✓ Manifest found with {len(manifest)} files:")
        for fname, fhash in manifest.items():
            print(f"  - {fname}: {fhash[:16]}...")
    else:
        print("❌ Manifest file not found!")
        return
    
    # Try loading the index
    print("\nLoading index from storage...")
    try:
        embedding = OllamaEmbedding(model_name=EMBEDDING_MODEL)
        llm = Ollama(model=MODEL_NAME, request_timeout=120.0)
        service_ctx = ServiceContext.from_defaults(llm=llm, embed_model=embedding)
        
        storage_ctx = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
        index = load_index_from_storage(storage_ctx, service_context=service_ctx)
        
        print(f"✓ Index loaded successfully!")
        print(f"  Index type: {type(index).__name__}")
        print(f"  Number of nodes: {len(index.docstore.docs)}")
        
        # Show docstore info
        print(f"\n✓ Documents in index:")
        for doc_id, doc in list(index.docstore.docs.items())[:5]:
            text_preview = doc.text[:80].replace('\n', ' ') if hasattr(doc, 'text') else "N/A"
            print(f"  - {doc_id}: {text_preview}...")
        
        if len(index.docstore.docs) > 5:
            print(f"  ... and {len(index.docstore.docs) - 5} more documents")
        
        print("\n✅ Index verification passed!")
        
    except Exception as e:
        print(f"❌ Failed to load index: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    verify_index()
