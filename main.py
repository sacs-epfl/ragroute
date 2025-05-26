import argparse
import asyncio
import logging
import signal
from multiprocessing import Process
from typing import Dict, List

from ragroute.config import DATA_SOURCES, SUPPORTED_MODELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


def start_router(dataset: str, data_sources: List[str], routing_strategy: str):
    from ragroute.router import run_router
    asyncio.run(run_router(dataset, data_sources, routing_strategy))

def start_data_source(index: int, dataset: str, data_source: str, bs: int):
    from ragroute.data_source import run_data_source
    asyncio.run(run_data_source(index, dataset, data_source, bs))


class FederatedSearchSystem:
    """Main controller for the federated search system."""
    
    def __init__(self, args):
        self.dataset: str = args.dataset
        self.routing_strategy: str = args.routing
        self.disable_llm: bool = args.disable_llm
        self.model: str = args.model
        self.processes = []
        self.server = None
        self.shutting_down = False
        self.main_task = None
        self.data_sources: List[str] = DATA_SOURCES[self.dataset]
        self.data_source_processes: Dict = {}
        self.bs: int = args.bs
        
    async def start(self):
        logger.info(f"Starting RAGRoute with dataset {self.dataset}")
        
        # Store reference to the main task for clean cancellation
        self.main_task = asyncio.current_task()
        
        # Start router process
        router_process = Process(target=start_router, args=(self.dataset, self.data_sources, self.routing_strategy))
        router_process.start()
        self.processes.append(router_process)
        logger.info("Router process started")
        
        # Start data source processes
        for idx, data_source in enumerate(self.data_sources):
            data_source_process = Process(target=start_data_source, args=(idx, self.dataset, data_source, self.bs))
            data_source_process.start()
            self.processes.append(data_source_process)
            self.data_source_processes[data_source] = data_source_process
            logger.info(f"Data source {data_source} process started")
        
        # Give the router and data sources some time to initialize
        await asyncio.sleep(1)
        
        # Start the server
        from ragroute.http_server import run_server
        self.server = await run_server(self.dataset, self.data_sources, self.routing_strategy, self.model, self.disable_llm)
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
    parser.add_argument("--dataset", type=str, default="feb4rag", choices=["medrag", "feb4rag"], help="The dataset being evaluated (influences the data sources)")
    parser.add_argument("--routing", type=str, default="ragroute", choices=["ragroute", "all", "random", "none"], help="The routing method to use - for random, we randomly pick n/2 of the n data sources")
    parser.add_argument("--disable-llm", action="store_true", help="Disable the LLM for testing purposes")
    parser.add_argument("--model", type=str, default=SUPPORTED_MODELS[0], choices=SUPPORTED_MODELS, help="The model to use for the LLM")
    parser.add_argument("--bs", type=int, default=1, help="The batch size to use for concurrency in the datasources")
    args = parser.parse_args()
    
    controller = FederatedSearchSystem(args)
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

