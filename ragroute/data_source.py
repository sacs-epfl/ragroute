"""
Contains the implementation of a RAGRoute data source.
"""

import asyncio
import json
import logging
import os
import pickle
import time

import faiss

import numpy as np
import zmq
import zmq.asyncio

from ragroute.config import DATA_SOURCE_DELAY, EMBEDDING_MODELS_PER_DATA_SOURCE, FEB4RAG_DIR, K, MEDRAG_DIR, SERVER_CLIENT_BASE_PORT, CLIENT_SERVER_BASE_PORT, WIKIPEDIA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("client")

class DataSource:
    
    def __init__(self, client_id: int, dataset: str, name: str, simulate: bool = False):
        self.client_id: int = client_id
        self.dataset: str = dataset
        self.simulate: bool = simulate
        
        if dataset == "medrag":
            self.dataset_dir = MEDRAG_DIR
        elif dataset == "feb4rag":
            self.dataset_dir = FEB4RAG_DIR
        elif dataset == "wikipedia":
            self.dataset_dir = WIKIPEDIA_DIR
        else:
            raise ValueError(f"Unknown dataset when starting data source {name}: {dataset}")

        self.name: str = name
        self.recv_port: int = SERVER_CLIENT_BASE_PORT + client_id
        self.send_port: int = CLIENT_SERVER_BASE_PORT + client_id
        self.running: bool = False
        self.context = zmq.asyncio.Context()

        if dataset == "medrag":
            self.index_dir: str = os.path.join(self.dataset_dir, self.name, "index", "ncbi/MedCPT-Article-Encoder")
            self.index_path: str = os.path.join(self.index_dir, "faiss.index")
            self.doc_ids_path: str = os.path.join(self.index_dir, "metadatas.jsonl")
        elif dataset == "feb4rag":
            self.index_dir: str = os.path.join(self.dataset_dir, "dataset_creation", "2_search", "embeddings", name)
            model_name: str = EMBEDDING_MODELS_PER_DATA_SOURCE[dataset][name][0]
            self.index_path: str = os.path.join(self.index_dir, f"{name}_{model_name}.faiss")
            self.doc_ids_path: str = os.path.join(self.index_dir, f"{name}_{model_name}.docids.json")
        elif dataset == "wikipedia":
            self.index_dir: str = os.path.join(self.dataset_dir, "faiss_clusters", "normalized_indexes")
            self.index_path: str = os.path.join(self.index_dir, f"faiss_index_{name}_normalized.index")

            # Load auxiliary data for Wikipedia
            with open(os.path.join(self.dataset_dir, "faiss_clusters", "split_texts_titles", f"titles_{self.name}.txt"), "r", encoding="utf-8") as f:
                self.mmlu_titles = f.read().splitlines()
            with open(os.path.join(self.dataset_dir, "faiss_clusters", "split_texts_titles", f"texts_{self.name}.txt"), "r", encoding="utf-8") as f:
                self.mmlu_texts = f.read().splitlines()

            logger.info(f"Initialized Wikipedia data source {name} with index path {self.index_path}")
        
        self.faiss_indexes = None
        self.cache_jsonl = {}

    def load_faiss_index(self):
        logger.info(f"Loading FAISS index for {self.name}")
        index = faiss.read_index(self.index_path)
        if self.dataset == "medrag":
            metadatas = [json.loads(line) for line in open(self.doc_ids_path).read().strip().split('\n')]
        elif self.dataset == "feb4rag":
            with open(self.doc_ids_path, "r") as f:
                metadatas = json.load(f)
        elif self.dataset == "wikipedia":
            metadatas = []  # TODO not sure what to do here
        self.faiss_indexes = index, metadatas
        logger.info(f"FAISS index for {self.name} loaded successfully")
        
    async def start(self):
        """Start the client and listen for queries."""
        logger.info(f"Starting client {self.client_id}")
        self.running = True
        
        # Socket to receive queries from server
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.bind(f"tcp://*:{self.recv_port}")
        
        # Socket to send results back to server
        self.sender = self.context.socket(zmq.PUSH)
        self.sender.connect(f"tcp://localhost:{self.send_port}")

        if not self.simulate:
            self.load_faiss_index()
        
        try:
            while self.running:
                try:
                    # Wait for queries with a timeout to allow for clean shutdown
                    query_data = await self.receiver.recv_json()
                    logger.info(f"Data source {self.client_id} received query: {query_data['id']}")
                    start_time = time.time()

                    if self.simulate:
                        # Simulate passing of time (async)
                        ids = ["doc1", "doc2", "doc3"]
                        docs = ["Document 1 content", "Document 2 content", "Document 3 content"]
                        scores = [0.9, 0.85, 0.8]
                        await asyncio.sleep(DATA_SOURCE_DELAY)
                    else:
                        embedding = query_data["embedding"]
                        embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
                        if self.dataset == "medrag":
                            ids, docs, scores = self.retrieve_docs_medrag(embedding, K[self.dataset])
                        elif self.dataset == "feb4rag":
                            ids, docs, scores = self.retrieve_docs_fed4rag(embedding, K[self.dataset])
                        elif self.dataset == "wikipedia":
                            ids, docs, scores = self.retrieve_docs_wikipedia(embedding, K[self.dataset])
                    
                    # Prepare and send response
                    response = {
                        "query_id": query_data["id"],
                        "client_id": self.client_id,
                        "name": self.name,
                        "indices": ids,
                        "docs": docs,
                        "scores": scores,
                        "duration": time.time() - start_time,
                    }
                    await self.sender.send_json(response)
                    logger.info(f"Client {self.client_id} sent response for query: {query_data['id']}")
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error when fetching documents from data source {self.name} (query data: {query_data}): {e}")

        except asyncio.CancelledError:
            logger.info(f"Client {self.client_id} shutdown requested")

    def retrieve_docs_fed4rag(self, query_embed, k):
        def idx2txt(ids):
            if self.name not in self.cache_jsonl:
                corpus_path = os.path.join(self.dataset_dir, "dataset_creation/original_dataset", self.name, self.name, "corpus.jsonl")
                corpus = {}
                with open(corpus_path, "r") as file:
                    for line in file:
                        entry = json.loads(line)
                        corpus[entry["_id"]] = entry
                self.cache_jsonl[self.name] = corpus

            corpus_data = self.cache_jsonl[self.name]
            return [corpus_data.get(doc_id, None) for doc_id in ids]
        
        index, docids = self.faiss_indexes
        res_ = index.search(query_embed, k)
        ids = [docids[i] for i in res_[1][0]]

        docs = idx2txt(ids)

        return ids, docs, []  # We don't have scores for FeB4RAG, so return empty list

    def retrieve_docs_medrag(self, query_embed, k):
        def idx2txt(indices):
            results = []
            for i in indices:
                source = i["source"]
                index = i["index"]

                # added by me to go faster...
                # Checks if the file's lines are already cached
                if source not in self.cache_jsonl:
                    file_path = os.path.join(self.dataset_dir, self.name, "chunk", f"{source}.jsonl")
                    with open(file_path, "r") as file:
                        # Cache raw lines as strings instead of fully parsed JSON
                        self.cache_jsonl[source] = file.read().strip().split("\n")

                # Parse the specific line at the requested index
                line = self.cache_jsonl[source][index]
                results.append(json.loads(line))  # Parse only when needed
            return results
        
        index, metadatas = self.faiss_indexes
        res_ = index.search(query_embed, k=k)
        scores = res_[0][0].tolist()

        # from faiss idx to corresponding source and index
        indices = [metadatas[i] for i in res_[1][0]]
        # get the corresponding documents
        docs = idx2txt(indices)

        return indices, docs, scores

    def retrieve_docs_wikipedia(self, query_embed, k):
        # Normalize the query
        query_vec = query_embed.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vec)

        index, metadatas = self.faiss_indexes
        res_ = index.search(query_vec, k)
        scores = res_[0][0].tolist()
        local_indices = res_[1]

        docs = []
        for i, local_idx in enumerate(local_indices[0]):
            title = self.mmlu_titles[local_idx]
            text = self.mmlu_texts[local_idx]
            docs.append((title, text))

        return local_indices[0].tolist(), docs, scores
            
    def stop(self):
        logger.info(f"Stopping client {self.client_id}")
        self.running = False
        self.receiver.close()
        self.sender.close()
        self.context.term()

async def run_data_source(client_id: int, dataset: int, name: str, simulate: bool = False):
    data_source = DataSource(client_id, dataset, name, simulate=simulate)
    await data_source.start()
