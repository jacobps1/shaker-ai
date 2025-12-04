#!/usr/bin/env python3
import streamlit as st
from pathlib import Path
import sys

try:
    from llama_index.legacy import StorageContext, load_index_from_storage, ServiceContext
    from llama_index.legacy.llms.ollama import Ollama
    from llama_index.legacy.embeddings import OllamaEmbedding
except Exception as e:
    st.error("Error importing LlamaIndex or Ollama bindings. Check your environment.")
    raise

STORAGE_DIR = Path(__file__).parent.parent / "storage"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "phi3:mini"


def main():
    st.set_page_config(page_title="Shaker AI", layout="wide")
    st.title("📚 Shaker AI Assistant")

    STORAGE_DIR.mkdir(exist_ok=True)

    query_engine = None
    llm = None

    # Check if index exists
    if not (STORAGE_DIR / "docstore.json").exists():
        st.error("❌ Index not found. Please run the ingest service first to create the index.")
        st.info("Run: `./run_ingest_service.sh` from the project root directory.")
    else:
        # Load persisted index (fast) - this should NOT rebuild embeddings
        with st.spinner("Loading index from storage..."):
            try:
                embedding = OllamaEmbedding(model_name=EMBEDDING_MODEL)
                llm = Ollama(model=LLM_MODEL, request_timeout=120.0)
                service_ctx = ServiceContext.from_defaults(llm=llm, embed_model=embedding)
                
                storage_ctx = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
                index = load_index_from_storage(storage_ctx, service_context=service_ctx)
                
                # Create query engine with the same LLM
                query_engine = index.as_query_engine(llm=llm, streaming=True)
            except Exception as e:
                if "404" in str(e) or "localhost:11434" in str(e):
                    st.error("❌ Cannot connect to Ollama service")
                    st.info("Please start Ollama first. Run: `ollama serve`")
                else:
                    st.error(f"❌ Failed to load index: {str(e)}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask a question…")

    if user_input:
        if query_engine is None:
            st.error("Cannot process query: Index not loaded. Please check the error message above.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)

            with st.chat_message("assistant"):
                response = query_engine.query(user_input)
                full_text = ""
                placeholder = st.empty()
                for tok in response.response_gen:
                    full_text += tok
                    placeholder.write(full_text)

                st.session_state.messages.append({"role": "assistant", "content": full_text})


if __name__ == '__main__':
    main()