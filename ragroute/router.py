"""Router process for the federated search system."""

import asyncio
import json
import logging
import os
import pickle
import time
from typing import Dict, List, Set

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import zmq
import zmq.asyncio

from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

from ragroute.config import EMBEDDING_MAX_LENGTH, EMBEDDING_MODELS_PER_DATA_SOURCE, FEB4RAG_SOURCE_TO_ID, MODELS_FEB4RAG_DIR, MODELS_USR_DIR, ROUTER_DELAY, SERVER_ROUTER_PORT, ROUTER_SERVER_PORT, MAX_QUEUE_SIZE, USR_DIR
from ragroute.models.medrag.custom_sentence_transformer import CustomizeSentenceTransformer
from ragroute.queue_manager import QueryQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("router")


# Dimension of the input features for the neural network
MEDRAG_INPUT_DIMENSION =    1536 
FEDRAG_INPUT_DIMENSION =    8205
WIKIPEDIA_INPUT_DIMENSION = 1536


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


class Router:
    """Router that processes queries and determines which clients should handle them."""
    
    def __init__(self, dataset: str, data_sources: List[str], routing_strategy: str, simulate: bool = False):
        self.dataset: str = dataset
        self.data_sources: List[str] = data_sources
        self.routing_strategy: str = routing_strategy
        self.simulate: bool = simulate
        self.running: bool = False
        self.context = zmq.asyncio.Context()
        self.queue = QueryQueue(MAX_QUEUE_SIZE)
        self.tokenizer = None  # For MMLU

        self.device="cuda" if torch.cuda.is_available() else "cpu"

        embedding_models_info: Set[str] = set()
        for data_source in self.data_sources:
            embedding_models_info.add(EMBEDDING_MODELS_PER_DATA_SOURCE[dataset][data_source])

        if self.simulate:
            # Skip the loading of embedding models in simulation mode
            logger.info("Running in simulation mode, skipping embedding model loading")
            self.embedding_models = {}
            for model_name, _ in embedding_models_info:
                self.embedding_models[model_name] = None
            return

        logger.info(f"Loading embedding models: {embedding_models_info}")
        self.embedding_models = {}
        if dataset == "medrag":
            for model_name, _ in embedding_models_info:
                model = CustomizeSentenceTransformer(model_name, device=self.device)
                model.eval()
                self.embedding_models[model_name] = model
        elif dataset == "feb4rag":
            from ragroute.models.feb4rag.model_zoo import CustomModel, BeirModels
            for model_name, model_type in embedding_models_info:
                model_loader = BeirModels(os.path.join(MODELS_FEB4RAG_DIR, "dataset_creation/2_search/models"), specific_model=model_name) if model_type == "beir" else \
                        CustomModel(model_dir=os.path.join(MODELS_FEB4RAG_DIR, "dataset_creation/2_search/models"), specific_model=model_name)
                model = model_loader.load_model(model_name, model_name_or_path=None, cuda=torch.cuda.is_available())
                self.embedding_models[model_name] = model
        elif dataset == "wikipedia":
            for model_name, _ in embedding_models_info:
                model = DPRQuestionEncoder.from_pretrained(model_name).to(self.device)
                model.eval()
                self.embedding_models[model_name] = model
                self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)

    def load_router(self):
        if self.dataset == "medrag":
            model_path = os.path.join(MODELS_USR_DIR, "MedRAG/routing/best_model.pth")
            input_dim = MEDRAG_INPUT_DIMENSION
        elif self.dataset == "feb4rag":
            model_path = os.path.join(MODELS_USR_DIR, "FeB4RAG/dataset_creation/2_search/router_best_model.pt")
            input_dim = FEDRAG_INPUT_DIMENSION
        elif self.dataset == "wikipedia":
            model_path = os.path.join(MODELS_USR_DIR, "Retrieval-QA-Benchmark_backup", "euromlsys", "new_submission", "cluster_router_output", "best_model.pth")
            input_dim = WIKIPEDIA_INPUT_DIMENSION

        self.router = CorpusRoutingNN(input_dim).to(self.device)
        self.router.load_state_dict(torch.load(model_path, map_location=self.device))
        self.router.eval()

        # Load scalar
        if self.dataset == "medrag":
            scaler_path = os.path.join(MODELS_USR_DIR, "MedRAG/routing/preprocessed_data.pkl")
            with open(scaler_path, "rb") as f:
                _, _, _, self.scaler, _ = pickle.load(f)

        # Load the centroids
        self.centroids = {}
        for corpus in self.data_sources:
            if self.dataset == "medrag":
                stats_file = os.path.join(USR_DIR, "MedRAG/routing/", f"{corpus}_stats.json")
            elif self.dataset == "feb4rag":
                stats_file = os.path.join(USR_DIR, "FeB4RAG/dataset_creation/2_search/embeddings", corpus+"_"+EMBEDDING_MODELS_PER_DATA_SOURCE[self.dataset][corpus][0]+"_stats.json")
            elif self.dataset == "wikipedia":
                stats_file = os.path.join(USR_DIR, "wiki_dataset", "dpr_wiki_index", "faiss_clusters", "cluster_stats.json")
            if self.dataset == "wikipedia":
                with open(stats_file, "r") as f:
                    cluster_num = int(corpus)
                    corpus_stats = json.load(f)[cluster_num]
            else:
                with open(stats_file, "r") as f:
                    corpus_stats = json.load(f)

            centroid = np.array(corpus_stats["centroid"], dtype=np.float32)
            centroid = np.pad(centroid, (0, EMBEDDING_MAX_LENGTH[self.dataset] - len(centroid)))  # Pad to max length
            self.centroids[corpus] = centroid
        
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

        if not self.simulate:
            self.load_router()
            #warmup
            with torch.no_grad():
                dummy_input = torch.zeros(1, self.router.fc1.in_features).to(self.device)
                _ = self.router(dummy_input)
        else:
            logger.info("Running in simulation mode, skipping router loading")
        
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
                    query_data = await self.receiver.recv_json()
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

    def select_relevant_sources(self, query_embeddings: Dict[str, torch.Tensor]) -> List[str]:
        if self.simulate:  # Return all data sources in simulation mode
            return self.data_sources

        if self.routing_strategy == "ragroute":
            return self.select_relevant_sources_ragroute(query_embeddings)
        elif self.routing_strategy == "all":
            return self.data_sources
        elif self.routing_strategy == "random":
            if self.dataset == "medrag":
                return random.sample(self.data_sources, 2)
            elif self.dataset == "feb4rag":
                return random.sample(self.data_sources, 9)
            elif self.dataset == "wikipedia":
                return random.sample(self.data_sources, 1)
        elif self.routing_strategy == "none":
            return []
        else:
            raise ValueError(f"Unknown routing strategy: {self.routing_strategy}")

    def select_relevant_sources_ragroute(self, query_embeddings: Dict[str, torch.Tensor]) -> List[str]:
        inputs = []

        # First, we pad the query embeddings to the maximum length
        padded_query_embeddings = {}
        for model_name in query_embeddings.keys():
            query_embed = query_embeddings[model_name]
            padded_q = np.pad(query_embed, (0, EMBEDDING_MAX_LENGTH[self.dataset] - len(query_embed)))
            padded_query_embeddings[model_name] = padded_q

        for corpus in self.data_sources:
            model_for_corpus: str = EMBEDDING_MODELS_PER_DATA_SOURCE[self.dataset][corpus][0]
            features = np.concatenate([padded_query_embeddings[model_for_corpus], self.centroids[corpus]])
            if self.dataset == "feb4rag":
                # We need to append the one-hot vector for the corpus when using the feb4rag dataset
                source_id = FEB4RAG_SOURCE_TO_ID[corpus]
                source_id_vec = np.eye(len(FEB4RAG_SOURCE_TO_ID))[source_id]
                features = np.concatenate([features, source_id_vec])
            inputs.append(features)
        
        if self.dataset == "medrag":
            inputs = self.scaler.transform(inputs)
        input_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.router(input_tensor)
            outputs = outputs.view(-1)
            probabilities = torch.sigmoid(outputs)
            if self.dataset == "medrag":	
                predictions = (probabilities > 0.5760).cpu().numpy()
            else:
                predictions = (probabilities > 0.5).cpu().numpy()
        
        sources_corpora = [corpus for prediction, corpus in zip(predictions, self.data_sources) if prediction]
        return sources_corpora

    def encode_query(self, query):
        if self.simulate:
            logger.info("Running in simulation mode, skipping query encoding")
            return {model_name: np.random.rand(EMBEDDING_MAX_LENGTH[self.dataset]) for model_name in self.embedding_models.keys()}  # Generate some random embeddings

        with torch.no_grad():
            embeddings: Dict[str, torch.Tensor] = {}
            for model_name, model in self.embedding_models.items():
                if self.dataset == "medrag":
                    embeddings[model_name] = model.encode([query], show_progress_bar=False)[0]
                elif self.dataset == "feb4rag":
                    emb = model.encode_queries([query], batch_size=1, convert_to_tensor=False)[0]
                    embeddings[model_name] = emb
                elif self.dataset == "wikipedia":
                    inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)
                    emb = model(**inputs).pooler_output
                    embeddings[model_name] = emb.cpu().numpy()[0]
        return embeddings

    async def _process_query(self, query_data):
        """Process a query and determine which clients should handle it."""
        logger.debug(f"Router processing query: {query_data['id']}")

        start_time = time.time()
        query_embeddings: Dict[str, torch.Tensor] = self.encode_query(query_data["query"])
        embed_time = time.time() - start_time

        start_time = time.time()
        sources_corpora = self.select_relevant_sources(query_embeddings)
        select_time = time.time() - start_time

        serialized_embeddings = {}
        for model_name, embedding in query_embeddings.items():
            serialized_embeddings[model_name] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

        if self.simulate:
            await asyncio.sleep(ROUTER_DELAY)
        
        response = {
            "query_id": query_data["id"],
            "data_sources": sources_corpora,
            "embeddings": serialized_embeddings,
            "embedding_time": embed_time,
            "selection_time": select_time,
        }
        
        await self.sender.send_json(response)
        logger.info(f"Router sent routing decision {response['data_sources']} to server for query: {query_data['id']}")
        
    def stop(self):
        """Stop the router."""
        logger.info("Stopping router")
        self.running = False
        self.receiver.close()
        self.sender.close()
        self.context.term()

async def run_router(dataset: str, data_sources: List[str], routing_strategy: str, simulate: bool = False):
    """Run the router process."""
    router = Router(dataset, data_sources, routing_strategy, simulate=simulate)
    await router.start()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python router.py <num_clients>")
        sys.exit(1)
        
    num_clients = int(sys.argv[1])
    asyncio.run(run_router(num_clients))
