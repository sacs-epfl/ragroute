"""
Contains the implementation of a RAGRoute data source.
"""

import asyncio
import json
import logging
import os
import time

import faiss

import numpy as np
import zmq
import zmq.asyncio

from ragroute.config import FEB4RAG_DIR, K, MEDRAG_DIR, SERVER_CLIENT_BASE_PORT, CLIENT_SERVER_BASE_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("client")

class DataSource:
    
    def __init__(self, client_id: int, dataset: str, name: str):
        self.client_id: int = client_id
        self.dataset: str = dataset
        
        if dataset == "medrag":
            self.dataset_dir = MEDRAG_DIR
        elif dataset == "feb4rag":
            self.dataset_dir = FEB4RAG_DIR
        else:
            raise ValueError(f"Unknown dataset when starting data source {name}: {dataset}")

        self.name: str = name
        self.recv_port: int = SERVER_CLIENT_BASE_PORT + client_id
        self.send_port: int = CLIENT_SERVER_BASE_PORT + client_id
        self.running: bool = False
        self.context = zmq.asyncio.Context()

        self.index_dir: str = os.path.join(self.dataset_dir, self.name, "index", "ncbi/MedCPT-Article-Encoder")
        self.faiss_indexes = {}
        self.cache_jsonl = {}
        
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

        # Load the FAISS index and metadata
        logger.info(f"Loading FAISS index for {self.name}")
        index = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))
        metadatas = [json.loads(line) for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split('\n')]
        self.faiss_indexes[self.index_dir] = (index, metadatas)
        logger.info(f"FAISS index for {self.name} loaded successfully")
        
        try:
            while self.running:
                try:
                    # Wait for queries with a timeout to allow for clean shutdown
                    query_data = await asyncio.wait_for(self.receiver.recv_json(), timeout=0.5)
                    logger.debug(f"Data source {self.client_id} received query: {query_data['id']}")
                    start_time = time.time()

                    embedding = query_data["embedding"]
                    embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
                    docs, scores = self.retrieve_docs(embedding, K)
                    
                    # Prepare and send response
                    response = {
                        "query_id": query_data["id"],
                        "client_id": self.client_id,
                        "name": self.name,
                        "docs": docs,
                        "scores": scores,
                        "duration": time.time() - start_time,
                    }
                    await self.sender.send_json(response)
                    logger.debug(f"Client {self.client_id} sent response for query: {query_data['id']}")
                    
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"Client {self.client_id} shutdown requested")
        finally:
            self.stop()

    def retrieve_docs(self, query_embed, k):
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
        
        index, metadatas = self.faiss_indexes[self.index_dir]
        res_ = index.search(query_embed, k=k)
        scores = res_[0][0].tolist()

        # from faiss idx to corresponding source and index
        indices = [metadatas[i] for i in res_[1][0]]
        # get the corresponding documents
        docs = idx2txt(indices)

        return docs, scores
            
    def stop(self):
        logger.info(f"Stopping client {self.client_id}")
        self.running = False
        self.receiver.close()
        self.sender.close()
        self.context.term()

async def run_data_source(client_id: int, dataset: int, name: str):
    data_source = DataSource(client_id, dataset, name)
    await data_source.start()
