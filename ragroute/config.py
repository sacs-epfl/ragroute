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
USR_DIR = "/mnt/nfs/home/dpetresc"
MEDRAG_DIR = os.path.join(USR_DIR, "MedRAG", "corpus")
FEB4RAG_DIR = os.path.join(USR_DIR, "FeB4RAG")

# Dataset information
DATA_SOURCES = {
    "medrag": ["pubmed", "statpearls", "textbooks", "wikipedia"],
    "feb4rag": ["msmarco"],  # TODO order and extend
}
EMBEDDING_MODELS_PER_DATA_SOURCE = {
    "medrag": {
        "pubmed": ("ncbi/MedCPT-Query-Encoder", None),
        "statpearls": ("ncbi/MedCPT-Query-Encoder", None),
        "textbooks": ("ncbi/MedCPT-Query-Encoder", None),
        "wikipedia": ("ncbi/MedCPT-Query-Encoder", None),
    },
    "feb4rag": {
        "msmarco": ("e5-large", "custom"),
        "trec-covid": ("SGPT-5.8B-weightedmean-msmarco-specb-bitfit", "custom"),
        "nfcorpus": ("UAE-Large-V1", "custom"),
        "scidocs": ("all-mpnet-base-v2", "beir"),
        "nq": ("multilingual-e5-large", "custom"),
        "hotpotqa": ("ember-v1", "beir"),
        "fiqa": ("all-mpnet-base-v2", "beir"),
        "arguana": ("UAE-Large-V1", "custom"),
        "webis-touche2020": ("e5-base", "custom"),
        "dbpedia-entity": ("UAE-Large-V1", "custom"),
        "fever": ("UAE-Large-V1", "custom"),
        "climate-fever": ("UAE-Large-V1", "custom"),
        "scifact": ("gte-base", "beir"),
    }
}
K = 32  # Number of documents to retrieve from each data source

SUPPORTED_MODELS = ["llama3.1-8B-instruct", "qwen3-8B", "qwen3-0.6B"]
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
    },
    "qwen3-0.6B": {
        "docs_context_length": 38000,
        "max_tokens": 40960,
        "hf_name": "Qwen/Qwen3-0.6B",
        "ollama_name": "qwen3:0.6b",
    }
}
