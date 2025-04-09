"""Client process for the federated search system."""

import asyncio
import logging
import sys

import zmq
import zmq.asyncio

from ragroute.config import SERVER_CLIENT_BASE_PORT, CLIENT_SERVER_BASE_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("client")

class SearchClient:
    """Client that handles search queries."""
    
    def __init__(self, client_id):
        self.client_id = client_id
        self.recv_port = SERVER_CLIENT_BASE_PORT + client_id
        self.send_port = CLIENT_SERVER_BASE_PORT + client_id
        self.running = False
        self.context = zmq.asyncio.Context()
        
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
        
        try:
            while self.running:
                try:
                    # Wait for queries with a timeout to allow for clean shutdown
                    query_data = await asyncio.wait_for(self.receiver.recv_json(), timeout=0.5)
                    logger.info(f"Client {self.client_id} received query: {query_data['id']}")
                    
                    # Process the query (simulated delay)
                    await asyncio.sleep(0.01)
                    
                    # Prepare and send response
                    response = {
                        "query_id": query_data["id"],
                        "client_id": self.client_id,
                        "results": [f"Result from client {self.client_id}"]
                    }
                    await self.sender.send_json(response)
                    logger.info(f"Client {self.client_id} sent response for query: {query_data['id']}")
                    
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"Client {self.client_id} shutdown requested")
        finally:
            self.stop()
            
    def stop(self):
        """Stop the client."""
        logger.info(f"Stopping client {self.client_id}")
        self.running = False
        self.receiver.close()
        self.sender.close()
        self.context.term()

async def run_client(client_id):
    """Run a client process."""
    client = SearchClient(client_id)
    await client.start()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python client.py <client_id>")
        sys.exit(1)
        
    client_id = int(sys.argv[1])
    asyncio.run(run_client(client_id))
