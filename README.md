# shaker-ai
Repository for Shaker AI, an assistant that uses Retrieval-Augmented Generation (RAG) to answer queries from a knowledge base of custom recipes.

## Installation
```
# 1. Create a Python venv (optional but recommended)
python3 -m venv venv
source venv/bin/activate


# 2. Install Python dependencies
pip install -r requirements.txt


# 3. Install Ollama (if not installed)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini
ollama pull nomic-embed-text

# 4. Run an initial ingest to build the index
chmod u+x ./run_ingest_service.sh
./run_ingest_service.sh
```


## Running Shaker AI UI
```
chmod u+x ./run_ui.sh
./run_ui.sh
```