"""Stress test script for the RAGRoute federated search system."""

import asyncio
import aiohttp
import argparse
import json
import time
import statistics
import sys
from typing import Dict, List, Any

# Default server configuration
DEFAULT_URL = "http://127.0.0.1:8000/query"

async def send_query(session: aiohttp.ClientSession, url: str, query: str, query_id: int, use_post: bool = False) -> Dict[str, Any]:
    """Send a single query and return timing information."""
    start_time = time.time()
    
    try:
        if use_post:
            data = {"q": query}
            async with session.post(url, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                else:
                    # For non-200 responses, get the text instead of trying to parse JSON
                    error_text = await response.text()
                    result = {"error": error_text}
        else:
            params = {"q": query}
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                else:
                    # For non-200 responses, get the text instead of trying to parse JSON
                    error_text = await response.text()
                    result = {"error": error_text}
        
        elapsed = time.time() - start_time
        
        return {
            "query_id": query_id,
            "success": response.status == 200,
            "status": response.status,
            "time": elapsed,
            "result": result,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "query_id": query_id,
            "success": False,
            "status": None,
            "time": elapsed,
            "error": str(e),
        }

async def run_stress_test(
    total_queries: int,
    concurrency: int,
    query_template: str,
    server_url: str,
    use_post: bool = False,
    delay: float = 0,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """Run a stress test with the specified parameters."""
    connector = aiohttp.TCPConnector(limit=concurrency)
    all_results = []
    
    async with aiohttp.ClientSession(connector=connector) as session:
        for batch_start in range(0, total_queries, concurrency):
            batch_end = min(batch_start + concurrency, total_queries)
            batch_size = batch_end - batch_start
            
            if show_progress:
                progress = batch_start / total_queries
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '#' * filled_length + '-' * (bar_length - filled_length)
                sys.stdout.write(f'\rProgress: [{bar}] {batch_start}/{total_queries} ({progress:.1%})')
                sys.stdout.flush()
            else:
                print(f"Sending batch of {batch_size} queries ({batch_start+1}-{batch_end}/{total_queries})")
            
            # Create tasks for this batch
            tasks = []
            for i in range(batch_start, batch_end):
                query = query_template.format(query_id=i)
                task = send_query(session, server_url, query, i, use_post)
                tasks.append(task)
            
            # Execute the batch
            batch_results = await asyncio.gather(*tasks)
            all_results.extend(batch_results)
            
            # Process and display batch results if not showing progress bar
            if not show_progress:
                process_batch_results(batch_results)
            
            # Optional delay between batches
            if delay > 0 and batch_end < total_queries:
                await asyncio.sleep(delay)
    
    if show_progress:
        sys.stdout.write(f'\rProgress: [{"#" * bar_length}] {total_queries}/{total_queries} (100%)\n')
        sys.stdout.flush()
    
    return all_results

def process_batch_results(results: List[Dict[str, Any]]) -> None:
    """Process and display results from a batch of queries."""
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    
    success_times = [r["time"] for r in successes]
    
    print(f"Batch completed: {len(results)} queries")
    print(f"  Success: {len(successes)} ({len(successes)/len(results)*100:.1f}%)")
    if success_times:
        print(f"  Avg time: {statistics.mean(success_times):.3f}s")
        print(f"  Min time: {min(success_times):.3f}s")
        print(f"  Max time: {max(success_times):.3f}s")
    if failures:
        print(f"  Failures: {len(failures)}")
        for f in failures[:5]:  # Show first 5 failures
            if "error" in f:
                print(f"    Query {f['query_id']}: {f['error']}")
            else:
                print(f"    Query {f['query_id']}: Status {f['status']}")
        if len(failures) > 5:
            print(f"    ... and {len(failures)-5} more failures")
    print()

def summarize_results(all_results: List[Dict[str, Any]]) -> None:
    """Summarize the results of the entire stress test."""
    successes = [r for r in all_results if r["success"]]
    failures = [r for r in all_results if not r["success"]]
    
    if not all_results:
        print("No results to summarize.")
        return
    
    success_times = [r["time"] for r in successes]
    
    print("\n====== Stress Test Summary ======")
    print(f"Total queries: {len(all_results)}")
    print(f"Successful: {len(successes)} ({len(successes)/len(all_results)*100:.1f}%)")
    print(f"Failed: {len(failures)} ({len(failures)/len(all_results)*100:.1f}%)")
    
    if success_times:
        print("\nTiming statistics for successful queries:")
        print(f"  Average response time: {statistics.mean(success_times):.3f}s")
        print(f"  Median response time: {statistics.median(success_times):.3f}s")
        print(f"  Minimum response time: {min(success_times):.3f}s")
        print(f"  Maximum response time: {max(success_times):.3f}s")
        
        if len(success_times) > 1:
            print(f"  Standard deviation: {statistics.stdev(success_times):.3f}s")
        
        # Calculate percentiles
        sorted_times = sorted(success_times)
        p90 = sorted_times[int(len(sorted_times) * 0.9)]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]
        
        print(f"  90th percentile: {p90:.3f}s")
        print(f"  95th percentile: {p95:.3f}s")
        print(f"  99th percentile: {p99:.3f}s")
    
    if failures:
        error_types = {}
        for f in failures:
            if "error" in f:
                error = str(f["error"])
                error_types[error] = error_types.get(error, 0) + 1
            else:
                status = f"HTTP {f['status']}"
                error_types[status] = error_types.get(status, 0) + 1
        
        print("\nError distribution:")
        for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error}: {count} ({count/len(failures)*100:.1f}% of errors)")

def save_results_to_file(results: List[Dict[str, Any]], filename: str) -> None:
    """Save the test results to a JSON file."""
    # Create a clean version of results without the full result content to save space
    clean_results = []
    for r in results:
        clean_r = {
            "query_id": r["query_id"],
            "success": r["success"],
            "status": r["status"],
            "time": r["time"]
        }
        if "error" in r:
            clean_r["error"] = r["error"]
        clean_results.append(clean_r)
    
    # Add summary statistics
    successes = [r for r in results if r["success"]]
    success_times = [r["time"] for r in successes]
    
    summary = {
        "total_queries": len(results),
        "successful_queries": len(successes),
        "failed_queries": len(results) - len(successes),
        "success_rate": len(successes) / len(results) if results else 0
    }
    
    if success_times:
        summary["timing"] = {
            "average": statistics.mean(success_times),
            "median": statistics.median(success_times),
            "min": min(success_times),
            "max": max(success_times),
            "standard_deviation": statistics.stdev(success_times) if len(success_times) > 1 else 0,
            "percentiles": {
                "90": sorted(success_times)[int(len(success_times) * 0.9)],
                "95": sorted(success_times)[int(len(success_times) * 0.95)],
                "99": sorted(success_times)[int(len(success_times) * 0.99)]
            }
        }
    
    output = {
        "summary": summary,
        "results": clean_results
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {filename}")

async def main() -> None:
    """Parse arguments and run the stress test."""
    parser = argparse.ArgumentParser(description="Stress test the RAGRoute federated search system")
    parser.add_argument("--queries", type=int, default=100, help="Total number of queries to send")
    parser.add_argument("--concurrency", type=int, default=10, help="Maximum number of concurrent queries")
    parser.add_argument("--query", type=str, default="test query {query_id}", help="Query template (use {query_id} for ID)")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Server URL (default: http://127.0.0.1:8000/query)")
    parser.add_argument("--post", action="store_true", help="Use POST instead of GET")
    parser.add_argument("--delay", type=float, default=0, help="Delay between batches (seconds)")
    parser.add_argument("--output", type=str, help="Save results to this JSON file")
    parser.add_argument("--progress", action="store_true", help="Show progress bar instead of batch results")
    args = parser.parse_args()
    
    print(f"=== RAGRoute Stress Test ===")
    print(f"Server URL: {args.url}")
    print(f"Total queries: {args.queries}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Query template: '{args.query}'")
    print(f"HTTP method: {'POST' if args.post else 'GET'}")
    print(f"Delay between batches: {args.delay}s")
    print("Starting test...\n")
    
    start_time = time.time()
    
    # Run the stress test
    all_results = await run_stress_test(
        total_queries=args.queries,
        concurrency=args.concurrency,
        query_template=args.query,
        server_url=args.url,
        use_post=args.post,
        delay=args.delay,
        show_progress=args.progress
    )
    
    total_time = time.time() - start_time
    
    print(f"\nStress test completed in {total_time:.2f} seconds")
    summarize_results(all_results)
    
    # Save results to file if specified
    if args.output:
        save_results_to_file(all_results, args.output)

if __name__ == "__main__":
    asyncio.run(main())
