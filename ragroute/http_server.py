"""
HTTP server that receives queries and coordinates with the router and clients to perform federated search.
"""
import asyncio
from asyncio import ensure_future
from collections import defaultdict
import logging
import json
import os
import time
from typing import List
import uuid
from aiohttp import web

from ollama import AsyncClient, ChatResponse
import zmq
import zmq.asyncio

from ragroute.config import (
    EMBEDDING_MODELS_PER_DATA_SOURCE, FEB4RAG_DIR, K, SERVER_ROUTER_PORT, ROUTER_SERVER_PORT,
    SERVER_CLIENT_BASE_PORT, CLIENT_SERVER_BASE_PORT,
    INTERNAL_SERVER_ROUTER_PORT, INTERNAL_ROUTER_SERVER_PORT, INTERNAL_SERVER_CLIENT_BASE_PORT, INTERNAL_CLIENT_SERVER_BASE_PORT,
    HTTP_HOST, HTTP_PORT, MODELS
)
from ragroute.llm_message import generate_llm_message
from ragroute.rerank import rerank_feb4rag, rerank_medrag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

class HTTPServer:
    """HTTP server that coordinates the federated search system."""
    
    def __init__(self, dataset: str, data_sources: List[str], routing_strategy: str, model: str, disable_llm: bool = False):
        self.dataset: str = dataset
        self.data_sources: List[str] = data_sources
        self.routing_strategy: str = routing_strategy
        self.model: str = model
        self.model_info = MODELS[model]
        self.disable_llm: bool = disable_llm
        self.num_clients = len(data_sources)
        self.app = web.Application()
        self.app.add_routes([
            web.get('/query', self.handle_query),
            web.post('/query', self.handle_query),
        ])
        self.context = zmq.asyncio.Context()
        self.active_queries = {}
        self.running = False
        self.router_process = None
        self.client_processes = []
        
    async def start(self):
        """Start the server and initialize ZMQ sockets."""
        logger.info(f"Starting server with {self.num_clients} clients")
        self.running = True
        
        # Socket to send queries to router
        self.router_sender = self.context.socket(zmq.PUSH)
        self.router_sender.connect(f"tcp://localhost:{SERVER_ROUTER_PORT}")
        
        # Socket to receive routing decisions from router
        self.router_receiver = self.context.socket(zmq.PULL)
        self.router_receiver.bind(f"tcp://*:{INTERNAL_ROUTER_SERVER_PORT}")
        
        # Sockets to send queries to clients
        self.client_senders = {}
        for client_id in range(self.num_clients):
            socket = self.context.socket(zmq.PUSH)
            socket.connect(f"tcp://localhost:{SERVER_CLIENT_BASE_PORT + client_id}")
            self.client_senders[client_id] = socket
            
        # Sockets to receive results from clients
        self.client_receivers = {}
        for client_id in range(self.num_clients):
            socket = self.context.socket(zmq.PULL)
            socket.bind(f"tcp://*:{INTERNAL_CLIENT_SERVER_BASE_PORT + client_id}")
            self.client_receivers[client_id] = socket

        # Start listener for router responses
        self.router_listener_task = asyncio.create_task(self._listen_router())

        # Start listeners for client responses
        self.client_listener_tasks = []
        for client_id in range(self.num_clients):
            task = asyncio.create_task(self._listen_client(client_id))
            self.client_listener_tasks.append(task)
        
        # Load the relevance data for the FeB4RAG dataset (needed for reranking)
        self.relevance_data = {}
        if self.dataset == "feb4rag":
            self.relevance_data = defaultdict(list)
            with open(os.path.join(FEB4RAG_DIR, "dataset/qrels/BEIR-QRELS-RM.txt"), "r") as f:
                for line in f:
                    qid, _, docid, rel = line.strip().split()
                    self.relevance_data[qid].append((docid, rel))

        # Start the HTTP server
        runner = web.AppRunner(self.app)
        await runner.setup()
        self.site = web.TCPSite(runner, HTTP_HOST, HTTP_PORT)
        await self.site.start()
        logger.info(f"HTTP server started on http://{HTTP_HOST}:{HTTP_PORT}")
        
    async def handle_query(self, request):
        """Handle search requests."""
        if request.method == "GET":
            query = request.query.get("q", "")
            choices = request.query.get("choices", "")
            question_id = request.query.get("qid", "")
        else:  # POST
            data = await request.post()
            query = data.get("q", "")
            choices = data.get("choices", "")
            question_id = data.get("qid", "")

        if not query:
            return web.Response(text="Please provide a query", status=400)
        
        if not choices:
            return web.Response(text="Please provide choices", status=400)
        
        if self.dataset == "feb4rag" and not question_id:
            return web.Response(text="For FeB4RAG, please provide a question ID (qid)", status=400)
        
        # Load the choices which is a URLEncoded JSON string
        try:
            choices = json.loads(choices)
        except json.JSONDecodeError:
            return web.Response(text="Invalid choices format", status=400)

        query_id = str(uuid.uuid4())
        logger.debug(f"Received search query: {query} (ID: {query_id})")
        
        # Create a new future to track this query
        future = asyncio.Future()
        self.active_queries[query_id] = {
            "future": future,
            "query": query,
            "choices": choices,
            "question_id": question_id,
            "client_results": {},
            "pending_data_sources": set(),
            "metadata": {},
            "query_start_time": time.time(),
        }
        
        # Send query to router
        await self.router_sender.send_json({
            "id": query_id,
            "query": query
        })
        
        # Wait for all results
        try:
            #results = await asyncio.wait_for(future, timeout=300.0)
            results = await asyncio.wait_for(future, timeout=100000.0)
            return web.json_response(results)
        except asyncio.TimeoutError:
            logger.error(f"Query timed out: {query_id}")
            if query_id in self.active_queries:
                del self.active_queries[query_id]
            return web.Response(text="Search timed out", status=504)

    async def _listen_router(self):
        """Listen for routing decisions from the router."""
        try:
            while self.running:
                try:
                    # Wait for messages with a timeout to allow for clean shutdown
                    routing_data = await asyncio.wait_for(self.router_receiver.recv_json(), timeout=5)
                    query_id = routing_data["query_id"]
                    data_sources = routing_data["data_sources"]
                    embeddings = routing_data["embeddings"]
                    embedding_time = routing_data["embedding_time"]
                    selection_time = routing_data["selection_time"]

                    # Convert data_sources to a list of client IDs
                    client_ids = [self.data_sources.index(ds) for ds in data_sources]
                    
                    logger.debug(f"Received routing for query {query_id}: {data_sources}")
                    
                    if query_id in self.active_queries:
                        # Update pending clients
                        self.active_queries[query_id]["pending_data_sources"] = set(client_ids)
                        self.active_queries[query_id]["metadata"]["data_sources"] = data_sources
                        self.active_queries[query_id]["metadata"]["embedding_time"] = embedding_time
                        self.active_queries[query_id]["metadata"]["selection_time"] = selection_time
                        self.active_queries[query_id]["metadata"]["data_sources_stats"] = {}

                        # Start time for document selection
                        start_time = time.time()
                        self.active_queries[query_id]["doc_select_start_time"] = start_time
                        
                        # Forward query to designated clients
                        query = self.active_queries[query_id]["query"]
                        for client_id in client_ids:
                            data_source_name: str = self.data_sources[client_id]
                            model_for_data_source = EMBEDDING_MODELS_PER_DATA_SOURCE[self.dataset][data_source_name][0]

                            await self.client_senders[client_id].send_json({
                                "id": query_id,
                                "query": query,
                                "embedding": embeddings[model_for_data_source]
                            })
                            
                        # If no clients were selected, complete the query immediately
                        if not data_sources:
                            self.active_queries[query_id]["metadata"]["doc_select_time"] = 0
                            ensure_future(self._complete_query(query_id))
                    else:
                        logger.warning(f"Received routing for unknown query: {query_id}")
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.exception("Unhandled exception in _listen_router")
                    raise  # Re-raise to ensure it's not silently swallowed

        except asyncio.CancelledError:
            logger.info("Router listener task cancelled")
            raise
            
    async def _listen_client(self, client_id):
        """Listen for results from a specific client."""
        try:
            while self.running:
                try:
                    # Wait for messages with a timeout to allow for clean shutdown
                    raw_message = await asyncio.wait_for(self.client_receivers[client_id].recv(), timeout=1)
                    message_size = len(raw_message)  # Get the size of the raw message in bytes
                    result_data = json.loads(raw_message.decode('utf-8'))

                    query_id = result_data["query_id"]
                    ds_name: str = result_data["name"]
                    
                    logger.debug(f"Received results from data source {ds_name} for query {query_id}")
                    
                    if query_id in self.active_queries:
                        # Store the results
                        self.active_queries[query_id]["client_results"][client_id] = (result_data["indices"], result_data["docs"], result_data["scores"])
                        self.active_queries[query_id]["metadata"]["data_sources_stats"][ds_name] = {
                            "duration": result_data["duration"],
                            "message_size": message_size
                        }
                        
                        # Update pending clients
                        if client_id in self.active_queries[query_id]["pending_data_sources"]:
                            self.active_queries[query_id]["pending_data_sources"].remove(client_id)
                            
                        # We received responses from all clients, rerank and complete the query
                        if not self.active_queries[query_id]["pending_data_sources"]:
                            self.active_queries[query_id]["metadata"]["doc_select_time"] = time.time() - self.active_queries[query_id]["doc_select_start_time"]
                            ensure_future(self._complete_query(query_id))
                    else:
                        logger.warning(f"Received result for unknown query: {query_id}")
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"Client {client_id} listener task cancelled")
            raise
            
    async def _complete_query(self, query_id):
        if query_id not in self.active_queries:
            return
        
        query_data = self.active_queries[query_id]
        print("FINISHED QUERY ", query_id)
        
        # Combine results from all clients
        response = {
            "query_id": query_id,
            "query": query_data["query"],
            "answer": "dummy"
        }

        all_indices = []
        all_docs = []
        all_scores = []
        for client_id, results in query_data["client_results"].items():
            all_indices.extend(results[0])
            all_docs.extend(results[1])
            all_scores.extend(results[2])

        if self.dataset == "medrag":
            filtered_docs, _ = rerank_medrag(all_docs, all_scores, K)
        elif self.dataset == "feb4rag":
            filtered_docs, _ = rerank_feb4rag(all_indices, all_docs, query_data["question_id"], K, self.relevance_data)
        llm_message, docs_tokens = generate_llm_message(self.dataset, query_data["query"], filtered_docs, query_data["choices"], self.model)

        if self.disable_llm:
            self.active_queries[query_id]["metadata"]["generate_time"] = 0
            response["answer"] = ""
        else:
            try:
                start_time = time.time()
                # TODO
                #response_: ChatResponse = await AsyncClient().chat(model=self.model_info["ollama_name"], messages=llm_message, options={"num_predict": self.model_info["max_tokens"]})
                try:
                    response_: ChatResponse = await asyncio.wait_for(
                        AsyncClient().chat(
                            model=self.model_info["ollama_name"],
                            messages=llm_message,
                            options={"num_predict": self.model_info["max_tokens"]}
                        ),
                        timeout=120
                    )
                    generate_time = time.time() - start_time
                    self.active_queries[query_id]["metadata"]["generate_time"] = generate_time
                    response["answer"] = response_['message']['content']

                except asyncio.TimeoutError:
                    logger.warning(f"LLM generation timed out for query {query_id}")
                    response_ = "TIMEOUT"
                    self.active_queries[query_id]["metadata"]["generate_time"] = -1
                    response["answer"] = "Error generation timed out."

            except Exception as e:
                logger.error(f"Error generating LLM message: {e}", exc_info=True)
                self.active_queries[query_id]["metadata"]["generate_time"] = -1
                response["answer"] = f"Error generating response: {str(e)}"

        response["metadata"] = query_data["metadata"]
        response["metadata"]["e2e_time"] = time.time() - query_data["query_start_time"]
        response["metadata"]["docs_tokens"] = docs_tokens
        response["metadata"]["llm"] = self.model
        if not query_data["future"].done():
            query_data["future"].set_result(response)

        del self.active_queries[query_id]
            
    async def stop(self):
        """Stop the server and clean up resources."""
        if not self.running:
            logger.info("Server already stopped")
            return
            
        logger.info("Stopping server")
        self.running = False
        
        # Cancel listener tasks
        if hasattr(self, 'router_listener_task') and self.router_listener_task:
            self.router_listener_task.cancel()
            
        if hasattr(self, 'client_listener_tasks'):
            for task in self.client_listener_tasks:
                task.cancel()
            
        try:
            tasks_to_gather = []
            if hasattr(self, 'router_listener_task') and self.router_listener_task:
                tasks_to_gather.append(self.router_listener_task)
            if hasattr(self, 'client_listener_tasks'):
                tasks_to_gather.extend(self.client_listener_tasks)
                
            if tasks_to_gather:
                await asyncio.gather(*tasks_to_gather, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error during task cancellation: {e}")
            
        # Close all ZMQ sockets
        if hasattr(self, 'router_sender'):
            self.router_sender.close()
        if hasattr(self, 'router_receiver'):
            self.router_receiver.close()
        
        if hasattr(self, 'client_senders'):
            for socket in self.client_senders.values():
                socket.close()
            
        if hasattr(self, 'client_receivers'):
            for socket in self.client_receivers.values():
                socket.close()
            
        self.context.term()
        
        # Stop the HTTP server
        if hasattr(self, 'site') and self.site:
            try:
                await self.site.stop()
                logger.info("HTTP site stopped")
            except RuntimeError as e:
                logger.warning(f"HTTP site stop issue (likely already stopped): {e}")
        
        logger.info("Server stopped")
        
async def run_server(dataset: str, data_sources: List[str], routing_strategy: str, model: str, disable_llm: bool = False) -> HTTPServer:
    server = HTTPServer(dataset, data_sources, routing_strategy, model, disable_llm=disable_llm)
    await server.start()
    return server


