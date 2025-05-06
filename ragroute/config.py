import os

SERVER_ROUTER_PORT = 5555
ROUTER_SERVER_PORT = 5556
SERVER_CLIENT_BASE_PORT = 6000
CLIENT_SERVER_BASE_PORT = 7500

# HTTP server settings
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8000

# Router queue settings
MAX_QUEUE_SIZE = 100

# For loading the models and data
USR_DIR = "/Users/martijndevos"
MEDRAG_DIR = os.path.join(USR_DIR, "MedRAG", "corpus")

# Dataset information
DATA_SOURCES = {
    "medrag": ["pubmed", "statpearls", "textbooks", "wikipedia"]
}
K = 32  # Number of documents to retrieve from each data source

SUPPORTED_MODELS = ["llama3.1-8B-instruct", "qwen3-8B"]
MODELS = {
    "llama3.1-8B-instruct": {
        "docs_context_length": 128000,
        "max_tokens": 131072,
        "hf_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "ollama_name": "llama3.1:8b-instruct-q4_K_M",
    },
    "qwen3-8B": {
        "docs_context_length": 38000,
        "max_tokens": 40960,
        "hf_name": "Qwen/Qwen3-8B",
        "ollama_name": "qwen3:8b",
    }
}
