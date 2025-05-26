import asyncio
from typing import Dict, List
import signal
import logging
from multiprocessing import Process

from ragroute.config import DATA_SOURCES


def start_router(dataset: str, data_sources: List[str], routing_strategy: str, simulate: bool = False):
    from ragroute.router import run_router
    asyncio.run(run_router(dataset, data_sources, routing_strategy, simulate))

def start_data_source(index: int, dataset: str, data_source: str, simulate: bool = False):
    from ragroute.data_source import run_data_source
    asyncio.run(run_data_source(index, dataset, data_source, simulate))


class RAGRoute:
    """Main controller for the federated search system."""
    
    def __init__(self, args):
        self.dataset: str = args.dataset
        self.routing_strategy: str = args.routing
        self.disable_llm: bool = args.disable_llm
        self.model: str = args.model
        self.simulate: bool = args.simulate
        self.processes = []
        self.server = None
        self.shutting_down = False
        self.main_task = None
        self.data_sources: List[str] = DATA_SOURCES[self.dataset]
        self.data_source_processes: Dict = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    async def start(self):
        self.logger.info(f"Starting RAGRoute with dataset {self.dataset}")
        
        # Store reference to the main task for clean cancellation
        self.main_task = asyncio.current_task()
        
        # Start router process
        router_process = Process(target=start_router, args=(self.dataset, self.data_sources, self.routing_strategy, self.simulate))
        router_process.start()
        self.processes.append(router_process)
        self.logger.info("Router process started")
        
        # Start data source processes
        for idx, data_source in enumerate(self.data_sources):
            data_source_process = Process(target=start_data_source, args=(idx, self.dataset, data_source, self.simulate))
            data_source_process.start()
            self.processes.append(data_source_process)
            self.data_source_processes[data_source] = data_source_process
            self.logger.info(f"Data source {data_source} process started")
        
        # Give the router and data sources some time to initialize
        await asyncio.sleep(1)
        
        # Start the server
        from ragroute.http_server import run_server
        self.server = await run_server(self.dataset, self.data_sources, self.routing_strategy, self.model, self.disable_llm, self.simulate)
        self.logger.info("Server started")
        
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
                        self.logger.error(f"Process {process.pid} died unexpectedly")
                        self.processes.remove(process)
                    
                    if not self.processes and dead_processes:
                        self.logger.error("All processes have died, shutting down")
                        await self.stop()
                        break
        except asyncio.CancelledError:
            self.logger.info("Main task cancelled")
        finally:
            # Ensure cleanup happens if the task is cancelled
            if not self.shutting_down:
                await self.stop()
            
    async def stop(self):
        """Stop all components of the system."""
        if self.shutting_down:
            return
            
        self.shutting_down = True
        self.logger.info("Shutting down Federated Search System")
        
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
                    self.logger.info(f"Terminating process {process.pid}")
                    process.terminate()
            
            # Wait for processes to end with timeout
            for process in self.processes[:]:
                process.join(timeout=2)
                if not process.is_alive():
                    self.processes.remove(process)
                
            # Force kill any remaining processes
            for process in self.processes:
                self.logger.warning(f"Force killing process {process.pid}")
                process.kill()
                process.join(timeout=1)
                
            self.logger.info("All processes terminated")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
