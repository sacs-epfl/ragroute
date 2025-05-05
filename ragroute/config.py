"""Configuration settings for the federated search system."""

# ZMQ Communication ports
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
ONLINE = True  # Whether we compute everything from scratch or use the precomputed information

# Dataset information
DATA_SOURCES = {
    "medrag": ["pubmed", "statpearls", "textbooks", "wikipedia"]
}
K = 32
CONTEXT_LENGTH = 128000
MAX_TOKENS = 131072
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OLLAMA_MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"
