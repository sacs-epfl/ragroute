"""Main entry point for the federated search system."""

import argparse
import asyncio
import logging
import signal
import sys
from multiprocessing import Process

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


def start_router(num_clients):
    from ragroute.router import run_router
    asyncio.run(run_router(num_clients))

def start_client(client_id):
    from ragroute.client import run_client
    asyncio.run(run_client(client_id))


class FederatedSearchSystem:
    """Main controller for the federated search system."""
    
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.processes = []
        self.server = None
        self.shutting_down = False
        self.main_task = None
        
    async def start(self):
        """Start all components of the system."""
        logger.info(f"Starting RAGRoute with {self.num_clients} clients")
        
        # Store reference to the main task for clean cancellation
        self.main_task = asyncio.current_task()
        
        # Start router process
        router_process = Process(target=start_router, args=(self.num_clients,))
        router_process.start()
        self.processes.append(router_process)
        logger.info("Router process started")
        
        # Start client processes
        for client_id in range(self.num_clients):
            client_process = Process(target=start_client, args=(client_id,))
            client_process.start()
            self.processes.append(client_process)
            logger.info(f"Client {client_id} process started")
        
        # Give the router and clients some time to initialize
        await asyncio.sleep(1)
        
        # Start the server
        from ragroute.server import run_server
        self.server = await run_server(self.num_clients)
        logger.info("Server started")
        
        # Setup signal handler for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
        
        try:
            # Keep the main process running
            while True:
                await asyncio.sleep(1)
                
                # Check if any processes have died
                if not self.shutting_down:
                    dead_processes = [p for p in self.processes if not p.is_alive()]
                    for process in dead_processes:
                        logger.error(f"Process {process.pid} died unexpectedly")
                        self.processes.remove(process)
                    
                    if not self.processes and dead_processes:
                        logger.error("All processes have died, shutting down")
                        await self.stop()
                        break
        except asyncio.CancelledError:
            logger.info("Main task cancelled")
        finally:
            # Ensure cleanup happens if the task is cancelled
            if not self.shutting_down:
                await self.stop()
            
    async def stop(self):
        """Stop all components of the system."""
        if self.shutting_down:
            return
            
        self.shutting_down = True
        logger.info("Shutting down Federated Search System")
        
        # Cancel main task if called from signal handler
        if self.main_task and asyncio.current_task() != self.main_task:
            self.main_task.cancel()
        
        try:
            # Stop the server
            if self.server:
                await self.server.stop()
                self.server = None
                
            # Terminate all processes
            for process in self.processes:
                if process.is_alive():
                    logger.info(f"Terminating process {process.pid}")
                    process.terminate()
            
            # Wait for processes to end with timeout
            for process in self.processes[:]:
                process.join(timeout=2)
                if not process.is_alive():
                    self.processes.remove(process)
                
            # Force kill any remaining processes
            for process in self.processes:
                logger.warning(f"Force killing process {process.pid}")
                process.kill()
                process.join(timeout=1)
                
            logger.info("All processes terminated")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

def main():
    parser = argparse.ArgumentParser(description="RAGRoute")
    parser.add_argument("--clients", type=int, default=3, help="Number of client processes (data sources)")
    args = parser.parse_args()
    
    controller = FederatedSearchSystem(args.clients)
    try:
        asyncio.run(controller.start())
    except KeyboardInterrupt:
        # This should be handled by the signal handler, but just in case
        pass
    except Exception as e:
        logger.error(f"Error in main process: {e}")
    
    logger.info("Exiting application")

if __name__ == "__main__":
    main()
