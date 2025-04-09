"""Router process for the federated search system."""

import asyncio
import logging

import zmq
import zmq.asyncio

from ragroute.config import SERVER_ROUTER_PORT, ROUTER_SERVER_PORT, MAX_QUEUE_SIZE
from ragroute.queue_manager import QueryQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("router")

class Router:
    """Router that processes queries and determines which clients should handle them."""
    
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.running = False
        self.context = zmq.asyncio.Context()
        self.queue = QueryQueue(MAX_QUEUE_SIZE)
        
    async def start(self):
        """Start the router and process queries."""
        logger.info("Starting router process")
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
                    logger.info(f"Router received query: {query_data['id']}")
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
            
    async def _process_query(self, query_data):
        """Process a query and determine which clients should handle it."""
        logger.info(f"Router processing query: {query_data['id']}")
        
        # TODO here we should decide which clients should handle the query
        # For now, we just send the query to all clients
        # In a real implementation, this would involve more complex logic
        client_ids = list(range(self.num_clients))
        
        response = {
            "query_id": query_data["id"],
            "query": query_data["query"],
            "client_ids": client_ids
        }
        
        await self.sender.send_json(response)
        logger.info(f"Router sent routing decision to server for query: {query_data['id']}")
        
    def stop(self):
        """Stop the router."""
        logger.info("Stopping router")
        self.running = False
        self.receiver.close()
        self.sender.close()
        self.context.term()

async def run_router(num_clients):
    """Run the router process."""
    router = Router(num_clients)
    await router.start()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python router.py <num_clients>")
        sys.exit(1)
        
    num_clients = int(sys.argv[1])
    asyncio.run(run_router(num_clients))
