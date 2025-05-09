from asyncio import Queue


class QueryQueue:
    """Queue for managing incoming queries to the router."""
    
    def __init__(self, max_size=100):
        self.queue = Queue(maxsize=max_size)
        
    async def enqueue(self, query_data):
        """Add a query to the queue."""
        await self.queue.put(query_data)
        
    async def dequeue(self):
        """Get the next query from the queue."""
        return await self.queue.get()
        
    def task_done(self):
        """Mark a task as done."""
        self.queue.task_done()
        
    async def join(self):
        """Wait for all items in the queue to be processed."""
        await self.queue.join()
        
    def empty(self):
        """Check if the queue is empty."""
        return self.queue.empty()
        
    def qsize(self):
        """Get the current size of the queue."""
        return self.queue.qsize()
