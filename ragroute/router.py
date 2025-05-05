"""Router process for the federated search system."""

import asyncio
import json
import logging
import os
import pickle
import time
from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import zmq
import zmq.asyncio

from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer

from ragroute.config import ONLINE, SERVER_ROUTER_PORT, ROUTER_SERVER_PORT, MAX_QUEUE_SIZE, USR_DIR
from ragroute.queue_manager import QueryQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("router")


class CorpusRoutingNN(nn.Module):
    def __init__(self, input_dim):
        super(CorpusRoutingNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)


class CustomizeSentenceTransformer(SentenceTransformer): # change the default pooling "MEAN" to "CLS"
    def _load_auto_model(self, model_name_or_path, *args, **kwargs):
        cache_path = os.path.join(USR_DIR, ".cache/torch/sentence_transformers", model_name_or_path)
        transformer_model = Transformer(model_name_or_path, cache_dir=cache_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'cls')
        return [transformer_model, pooling_model]


class Router:
    """Router that processes queries and determines which clients should handle them."""
    
    def __init__(self, data_sources: List[str]):
        self.data_sources = data_sources
        self.running = False
        self.context = zmq.asyncio.Context()
        self.queue = QueryQueue(MAX_QUEUE_SIZE)

        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_function = CustomizeSentenceTransformer("ncbi/MedCPT-Query-Encoder", device=self.device)
        self.embedding_function.eval()
        
    async def start(self):
        """Start the router and process queries."""
        logger.info("Starting router process with %d data sources", len(self.data_sources))
        self.running = True
        
        # Socket to receive queries from server
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.bind(f"tcp://*:{SERVER_ROUTER_PORT}")
        
        # Socket to send routing decisions back to server
        self.sender = self.context.socket(zmq.PUSH)
        self.sender.connect(f"tcp://localhost:{ROUTER_SERVER_PORT}")
        
        # Start the worker tasks
        receive_task = asyncio.create_task(self._receive_queries())
        process_task = asyncio.create_task(self._process_queue())
        
        try:
            await asyncio.gather(receive_task, process_task)
        except asyncio.CancelledError:
            logger.info("Router shutdown requested")
            receive_task.cancel()
            process_task.cancel()
            try:
                await asyncio.gather(receive_task, process_task, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
        finally:
            self.stop()
            
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
        """Process queries from the queue."""
        try:
            while self.running:
                if not self.queue.empty():
                    query_data = await self.queue.dequeue()
                    await self._process_query(query_data)
                    self.queue.task_done()
                else:
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("Process task cancelled")
            raise

    def select_relevant_sources(self, query_embed):
        if ONLINE:
            inputs = []
            
            for corpus in self.data_sources:
                stats_file = os.path.join(USR_DIR, "MedRAG/routing/", f"{corpus}_stats.json")
                with open(stats_file, "r") as f:
                    corpus_stats = json.load(f)

                centroid = np.array(corpus_stats["centroid"], dtype=np.float32)
                features = np.concatenate([query_embed.flatten(), centroid])
                inputs.append(features)
            
            # Load scaler
            scaler_path = os.path.join(USR_DIR, "MedRAG/routing/preprocessed_data.pkl")
            with open(scaler_path, "rb") as f:
                _, _, _, scaler, _ = pickle.load(f)
            inputs = scaler.transform(inputs)

            input_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)

            model_path = os.path.join(USR_DIR, "MedRAG/routing/best_model.pth")
            model = CorpusRoutingNN(input_tensor.shape[1]).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()

            with torch.no_grad():
                outputs = model(input_tensor)
                outputs = outputs.view(-1)
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).cpu().numpy()

            sources_corpora = [corpus for prediction, corpus in zip(predictions, self.data_sources) if prediction]
        else:
            sources_corpora = self.data_sources

        return sources_corpora

    def encode_query(self, query):
        if ONLINE:
            with torch.no_grad():
                query_embed = self.embedding_function.encode([query])
        else:
            retrieval_cache_dir = os.path.join(USR_DIR, "MedRAG/retrieval_cache/") # bioasq	medmcqa  medqa	mmlu  pubmedqa
            emb_path = os.path.join(retrieval_cache_dir, dataset_name, "emb_queries", question_id+".npy")
            query_embed = np.load(emb_path)

        return query_embed
            
    async def _process_query(self, query_data):
        """Process a query and determine which clients should handle it."""
        logger.debug(f"Router processing query: {query_data['id']}")

        start_time = time.time()
        query_embed = self.encode_query(query_data["query"])
        embed_time = time.time() - start_time

        start_time = time.time()
        sources_corpora = self.select_relevant_sources(query_embed)
        select_time = time.time() - start_time
        
        response = {
            "query_id": query_data["id"],
            "data_sources": sources_corpora,
            "embedding": query_embed.tolist() if isinstance(query_embed, np.ndarray) else query_embed,
            "embedding_time": embed_time,
            "selection_time": select_time,
        }
        
        await self.sender.send_json(response)
        logger.debug(f"Router sent routing decision {response['data_sources']} to server for query: {query_data['id']}")
        
    def stop(self):
        """Stop the router."""
        logger.info("Stopping router")
        self.running = False
        self.receiver.close()
        self.sender.close()
        self.context.term()

async def run_router(data_sources: List[str]):
    """Run the router process."""
    router = Router(data_sources)
    await router.start()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python router.py <num_clients>")
        sys.exit(1)
        
    num_clients = int(sys.argv[1])
    asyncio.run(run_router(num_clients))
