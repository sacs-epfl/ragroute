"""
HTTP server that receives queries and coordinates with the router and clients to perform federated search.
"""
import asyncio
import logging
import uuid
from aiohttp import web

import zmq
import zmq.asyncio

from ragroute.config import (
    SERVER_ROUTER_PORT, ROUTER_SERVER_PORT,
    SERVER_CLIENT_BASE_PORT, CLIENT_SERVER_BASE_PORT,
    HTTP_HOST, HTTP_PORT
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

class SearchServer:
    """HTTP server that coordinates the federated search system."""
    
    def __init__(self, num_clients):
        self.num_clients = num_clients
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
        self.router_receiver.bind(f"tcp://*:{ROUTER_SERVER_PORT}")
        
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
            socket.bind(f"tcp://*:{CLIENT_SERVER_BASE_PORT + client_id}")
            self.client_receivers[client_id] = socket

        # Start listener for router responses
        self.router_listener_task = asyncio.create_task(self._listen_router())

        # Start listeners for client responses
        self.client_listener_tasks = []
        for client_id in range(self.num_clients):
            task = asyncio.create_task(self._listen_client(client_id))
            self.client_listener_tasks.append(task)

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
        else:  # POST
            data = await request.post()
            query = data.get("q", "")

        if not query:
            return web.Response(text="Please provide a query", status=400)

        query_id = str(uuid.uuid4())
        logger.info(f"Received search query: {query} (ID: {query_id})")
        
        # Create a new future to track this query
        future = asyncio.Future()
        self.active_queries[query_id] = {
            "future": future,
            "query": query,
            "client_results": {},
            "pending_clients": set()
        }
        
        # Send query to router
        await self.router_sender.send_json({
            "id": query_id,
            "query": query
        })
        
        # Wait for all results
        try:
            results = await asyncio.wait_for(future, timeout=10.0)
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
                    routing_data = await asyncio.wait_for(self.router_receiver.recv_json(), timeout=0.5)
                    query_id = routing_data["query_id"]
                    client_ids = routing_data["client_ids"]
                    
                    logger.info(f"Received routing for query {query_id}: {client_ids}")
                    
                    if query_id in self.active_queries:
                        # Update pending clients
                        self.active_queries[query_id]["pending_clients"] = set(client_ids)
                        
                        # Forward query to designated clients
                        query = self.active_queries[query_id]["query"]
                        for client_id in client_ids:
                            await self.client_senders[client_id].send_json({
                                "id": query_id,
                                "query": query
                            })
                            
                        # If no clients were selected, complete the query immediately
                        if not client_ids:
                            self._complete_query(query_id)
                    else:
                        logger.warning(f"Received routing for unknown query: {query_id}")
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            logger.info("Router listener task cancelled")
            raise
            
    async def _listen_client(self, client_id):
        """Listen for results from a specific client."""
        try:
            while self.running:
                try:
                    # Wait for messages with a timeout to allow for clean shutdown
                    result_data = await asyncio.wait_for(self.client_receivers[client_id].recv_json(), timeout=0.5)
                    query_id = result_data["query_id"]
                    
                    logger.info(f"Received results from client {client_id} for query {query_id}")
                    
                    if query_id in self.active_queries:
                        # Store the results
                        self.active_queries[query_id]["client_results"][client_id] = result_data["results"]
                        
                        # Update pending clients
                        if client_id in self.active_queries[query_id]["pending_clients"]:
                            self.active_queries[query_id]["pending_clients"].remove(client_id)
                            
                        # If all clients have responded, complete the query
                        if not self.active_queries[query_id]["pending_clients"]:
                            self._complete_query(query_id)
                    else:
                        logger.warning(f"Received result for unknown query: {query_id}")
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"Client {client_id} listener task cancelled")
            raise
            
    def _complete_query(self, query_id):
        """Complete a query by resolving its future."""
        if query_id in self.active_queries:
            query_data = self.active_queries[query_id]
            
            # Combine results from all clients
            combined_results = {
                "query_id": query_id,
                "query": query_data["query"],
                "results": []
            }
            
            # Add results from each client
            for client_id, results in query_data["client_results"].items():
                client_section = {
                    "client_id": client_id,
                    "results": results
                }
                combined_results["results"].append(client_section)
                
            # Resolve the future
            if not query_data["future"].done():
                query_data["future"].set_result(combined_results)
                
            # Clean up
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
        
async def run_server(num_clients):
    """Run the server process."""
    server = SearchServer(num_clients)
    await server.start()
    return server


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python server.py <num_clients>")
        sys.exit(1)
        
    num_clients = int(sys.argv[1])
    asyncio.run(run_server(num_clients))
