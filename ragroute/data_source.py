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

from ragroute.config import EMBEDDING_MODELS_PER_DATA_SOURCE, FEB4RAG_DIR, K, MEDRAG_DIR, SERVER_CLIENT_BASE_PORT, CLIENT_SERVER_BASE_PORT, MAX_QUEUE_SIZE, INTERNAL_CLIENT_SERVER_BASE_PORT, INTERNAL_SERVER_CLIENT_BASE_PORT
from ragroute.queue_manager import QueryQueue
from typing import List, Dict
import asyncio


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("client")

class DataSource:
    
    def __init__(self, client_id: int, dataset: str, name: str, bs: int):
        self.client_id: int = client_id
        self.dataset: str = dataset
        self.bs = bs
        
        if dataset == "medrag":
            self.dataset_dir = MEDRAG_DIR
        elif dataset == "feb4rag":
            self.dataset_dir = FEB4RAG_DIR
        else:
            raise ValueError(f"Unknown dataset when starting data source {name}: {dataset}")

        self.name: str = name
        self.recv_port: int = INTERNAL_SERVER_CLIENT_BASE_PORT + client_id
        self.send_port: int = CLIENT_SERVER_BASE_PORT + client_id
        self.running: bool = False
        self.context = zmq.asyncio.Context()
        self.queue = QueryQueue(MAX_QUEUE_SIZE)

        if dataset == "medrag":
            self.index_dir: str = os.path.join(self.dataset_dir, self.name, "index", "ncbi/MedCPT-Article-Encoder")
            self.index_path: str = os.path.join(self.index_dir, "faiss.index")
            self.doc_ids_path: str = os.path.join(self.index_dir, "metadatas.jsonl")
        elif dataset == "feb4rag":
            self.index_dir: str = os.path.join(self.dataset_dir, "dataset_creation", "2_search", "embeddings", name)
            model_name: str = EMBEDDING_MODELS_PER_DATA_SOURCE[dataset][name][0]
            self.index_path: str = os.path.join(self.index_dir, f"{name}_{model_name}.faiss")
            self.doc_ids_path: str = os.path.join(self.index_dir, f"{name}_{model_name}.docids.json")
        self.faiss_indexes = None
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
        index = faiss.read_index(self.index_path)
        if self.dataset == "medrag":
            metadatas = [json.loads(line) for line in open(self.doc_ids_path).read().strip().split('\n')]
        elif self.dataset == "feb4rag":
            with open(self.doc_ids_path, "r") as f:
                metadatas = json.load(f)
        self.faiss_indexes = index, metadatas
        logger.info(f"FAISS index for {self.name} loaded successfully")
        # Warm start
        self.faiss_indexes[0].search(np.zeros((1, self.faiss_indexes[0].d), dtype=np.float32), 1)


        # Start the worker tasks
        receive_task = asyncio.create_task(self._receive_queries())
        process_task = asyncio.create_task(self._process_queue())

        await asyncio.gather(receive_task, process_task)

    
    async def _receive_queries(self):
        """Receive queries from the server and add them to the queue."""
        try:
            while self.running:
                try:
                    # Wait for queries with a timeout to allow for clean shutdown
                    query_data = await asyncio.wait_for(self.receiver.recv_json(), timeout=0.5)
                    logger.debug(f"Router received query: {query_data['id']}")
                    await self.queue.enqueue(query_data)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.info("Receive task cancelled")
            raise
    
    async def _process_queue(self):
        """Process batches of queries from the queue."""
        try:
            while self.running:
                if self.dataset == "medrag":
                    query_batch = await self.queue.dequeue_batch(batch_size=self.bs, timeout=0.5)
                else:
                    query_batch = await self.queue.dequeue_batch(batch_size=self.bs, timeout=0.5)
                if query_batch:
                    await self._process_batch(query_batch)
                    for _ in query_batch:
                        self.queue.task_done()
                else:
                    await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            logger.info("Process task cancelled")
            raise

    async def _process_batch(self, query_batch: List[Dict]):
        try:
            start_time = time.time()
            embeddings = np.array([q["embedding"] for q in query_batch], dtype=np.float32)
            query_ids = [q["id"] for q in query_batch]
            print("PROCESSING BATCH ", self.name, " ", len(query_batch), query_ids)

            index, metadatas = self.faiss_indexes
            res_scores, res_indices = index.search(embeddings, k=K)
            print(self.name, len(query_batch), "FAISS search done")
            search_duration = time.time() - start_time
            print(self.name, " " , len(query_batch), " total duration ", search_duration)

            # For each query in the batch, get docs and send response ASAP
            for i in range(len(query_batch)):
                metadata_time = time.time()
                if self.dataset == "medrag":
                    indices = [metadatas[idx] for idx in res_indices[i]]
                    docs = self.idx2txt_medrag(indices)
                    scores = res_scores[i].tolist()
                elif self.dataset == "feb4rag":
                    indices = [metadatas[idx] for idx in res_indices[i]]
                    docs = self.idx2txt_fed4rag(indices)
                    scores = []  # No scores available for feb4rag

                response = {
                    "query_id": query_ids[i],
                    "client_id": self.client_id,
                    "name": self.name,
                    "indices": indices,
                    "docs": docs,
                    "scores": scores,
                    "duration": search_duration / len(query_batch) + (time.time()-metadata_time),
                }
                await self.sender.send_json(response)
                logger.info(f"Client {self.client_id} sent response for query {query_ids[i]} in {send_duration:.4f} seconds")

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
        
    def idx2txt_fed4rag(self, ids):
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

    def idx2txt_medrag(self, indices):
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
            
    def stop(self):
        logger.info(f"Stopping client {self.client_id}")
        self.running = False
        self.receiver.close()
        self.sender.close()
        self.context.term()

async def run_data_source(client_id: int, dataset: int, name: str, bs: int):
    data_source = DataSource(client_id, dataset, name, bs)
    try:
        await data_source.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested via keyboard interrupt.")
    finally:
        data_source.stop()

