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

# For loading the models
MODELS_USR_DIR = "/mnt/nfs/home/dpetresc"
MODELS_MEDRAG_DIR = os.path.join(MODELS_USR_DIR, "MedRAG", "corpus")
MODELS_FEB4RAG_DIR = os.path.join(MODELS_USR_DIR, "FeB4RAG")

# For loading the data
USR_DIR = "/mnt/nfs/home/dpetresc"
MEDRAG_DIR = os.path.join(USR_DIR, "MedRAG", "corpus")
FEB4RAG_DIR = os.path.join(USR_DIR, "FeB4RAG")
WIKIPEDIA_DIR = os.path.join(USR_DIR, "wiki_dataset", "dpr_wiki_index")

# If we're in simulation mode, these are the delays for each component (in seconds)
ROUTER_DELAY = 1
DATA_SOURCE_DELAY = 2
LLM_DELAY = 1

# Dataset information
DATA_SOURCES = {
    "medrag": ["pubmed", "statpearls", "textbooks", "wikipedia"],
    "feb4rag": ["msmarco", "trec-covid", "nfcorpus", "scidocs", "nq", "hotpotqa", "fiqa", "arguana", "webis-touche2020", "dbpedia-entity", "fever", "climate-fever", "scifact"],
    "wikipedia": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
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
    },
    "wikipedia": {
        "0": ("facebook/dpr-question_encoder-single-nq-base", None),
        "1": ("facebook/dpr-question_encoder-single-nq-base", None),
        "2": ("facebook/dpr-question_encoder-single-nq-base", None),
        "3": ("facebook/dpr-question_encoder-single-nq-base", None),
        "4": ("facebook/dpr-question_encoder-single-nq-base", None),
        "5": ("facebook/dpr-question_encoder-single-nq-base", None),
        "6": ("facebook/dpr-question_encoder-single-nq-base", None),
        "7": ("facebook/dpr-question_encoder-single-nq-base", None),
        "8": ("facebook/dpr-question_encoder-single-nq-base", None),
        "9": ("facebook/dpr-question_encoder-single-nq-base", None),
    }
}
FEB4RAG_SOURCE_TO_ID = {
  "arguana": 0,
  "climate-fever": 1,
  "dbpedia-entity": 2,
  "fever": 3,
  "fiqa": 4,
  "hotpotqa": 5,
  "msmarco": 6,
  "nfcorpus": 7,
  "nq": 8,
  "scidocs": 9,
  "scifact": 10,
  "trec-covid": 11,
  "webis-touche2020": 12
}
EMBEDDING_MAX_LENGTH = {
    "medrag": 768,
    "feb4rag": 4096,
    "wikipedia": 768,
}
K = 32  # Number of documents to retrieve from each data source

SYSTEM_PROMPTS = {
    "medrag": """You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents.
Please first think step-by-step and then choose the answer from the provided options.
Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}.
Your responses will be used for research purposes only, so please have a definite answer.""",
    "feb4rag": """You are a helpful assistant helping to answer user requests based on the provided search result.
Your responses should directly address the user's request and must be based on the information obtained from the provided search results.
You are forbidden to create new information that is not supported by these results.
You must attribute your response to the source from the search results by including citations, for example, [1]."""
}
USER_PROMPT_TEMPLATES = {
    "medrag": """Here are the relevant documents:
{{context}}

Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}:""",
    "feb4rag": """Here are the search results:
{{context}}

Here is the question:
{{question}}"""
}

SUPPORTED_MODELS = ["llama3.1-8B-instruct", "qwen3-8B", "qwen3-0.6B"]
MODELS = {
    "llama3.1-8B-instruct": {
        "docs_context_length": 128000,
        "max_tokens": 131072,
        "hf_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        #"ollama_name": "llama3.1:8b-instruct-q4_K_M",
        "ollama_name": "llama3.1_extended",
        #"ollama_name": "llama3.1:8b-instruct-q8_0",
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
