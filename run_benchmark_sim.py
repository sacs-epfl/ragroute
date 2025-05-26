import argparse
import asyncio
import json
import os
import time
import aiohttp
import random
import numpy as np
from ragroute.benchmark import Benchmark

completed_queries = 0
throughput_lock = asyncio.Lock()
MAX_CONCURRENT = 1000
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

async def fetch_answer(session, url):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=600)) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"[ERROR] Failed to fetch: {response.status} - {await response.text()}")
                return None
    except asyncio.TimeoutError:
        print(f"[TIMEOUT] Request to {url} timed out.")
        return None
    except Exception as e:
        print(f"[ERROR] Fetching answer failed: {e}")
        return None

async def log_throughput(interval, log_path, stop_event):
    global completed_queries
    start_time = time.time()
    with open(log_path, "w") as f:
        f.write("timestamp,completed,throughput_qps\n")

    while not stop_event.is_set():
        await asyncio.sleep(interval)
        async with throughput_lock:
            now = time.time()
            elapsed = now - start_time
            qps = completed_queries / elapsed if elapsed > 0 else 0.0
            with open(log_path, "a") as f:
                f.write(f"{now:.2f},{completed_queries},{qps:.2f}\n")

async def handle(session, question_id, question_data, benchmark, answer_file, benchmark_file, ds_stats_file, benchmark_name, dataset_name):
    global completed_queries
    try:
        print(f"[SEND] Query {question_id}")
        question = question_data["question"]
        options = question_data["options"]
        encoded_q = aiohttp.helpers.quote(question)
        encoded_opts = aiohttp.helpers.quote(json.dumps(options))
        url = f"http://localhost:8000/query?q={encoded_q}&choices={encoded_opts}&qid={question_id}"
        result = await fetch_answer(session, url)
        if not result:
            return
        print(f"[RECV] Answer for {question_id}")
        is_correct = benchmark.check_mirage_answer(question_data, result["answer"]) if benchmark_name == "MIRAGE" else True
        metadata = result["metadata"]
        data_sources = ":".join(metadata["data_sources"])

        with open(answer_file, "a") as f:
            f.write(json.dumps({"question_id": question_id, "answer": result["answer"]}) + "\n")
        with open(benchmark_file, "a") as f:
            f.write(f"{benchmark_name},{dataset_name},{metadata['llm']},{question_id},{int(is_correct)},{data_sources},{len(metadata['data_sources'])},"
                    f"{metadata['selection_time']},{metadata['embedding_time']},{metadata['doc_select_time']},{metadata['generate_time']},"
                    f"{metadata['e2e_time']},{metadata['docs_tokens']}\n")
        with open(ds_stats_file, "a") as f:
            for ds, stats in metadata["data_sources_stats"].items():
                f.write(f"{benchmark_name},{dataset_name},{question_id},{ds},{stats['duration']},{stats['message_size']}\n")

        async with throughput_lock:
            completed_queries += 1
    except Exception as e:
        print(f"[ERROR] Handling question {question_id}: {e}")

async def handle_with_limit(*args, **kwargs):
    async with semaphore:
        await handle(*args, **kwargs)

async def main():
    parser = argparse.ArgumentParser(description="Run a benchmark with RAGRoute.")
    parser.add_argument("--benchmark", type=str, default="FeB4RAG", choices=["MIRAGE", "FeB4RAG"])
    parser.add_argument("--benchmark-path", type=str, default="data/benchmark")
    parser.add_argument("--routing", type=str, required=True, choices=["ragroute", "all", "random", "none"])
    parser.add_argument("--questions", type=str, default=None, choices=['medqa', 'medmcqa', 'pubmedqa', 'bioasq', 'mmlu'])
    parser.add_argument("--interarrival", type=float, default=0.5, help="Mean interarrival time (Poisson rate)")
    parser.add_argument("--duration", type=int, default=60, help="Benchmark duration in seconds")
    parser.add_argument("--bs", type=int, default=1, help="bs concurrency datasources")
    # Remove default, so we can generate it dynamically later
    parser.add_argument("--query-set-path", type=str, default=None, help="Path to store or load queries+timing")
    args = parser.parse_args()

    # Use unique query set file per benchmark+routing+bs if not explicitly set
    if args.query_set_path is None:
        args.query_set_path = os.path.join("data", f"query_set_{args.benchmark}.json")


    file_suffix = f"{args.benchmark}_{args.routing}_{args.bs}"
    benchmark_file = os.path.join("data", f"benchmark_{file_suffix}.csv")
    ds_stats_file = os.path.join("data", f"ds_stats_{file_suffix}.csv")
    answer_file = os.path.join("data", f"answers_{file_suffix}.jsonl")
    throughput_file = os.path.join("data", f"throughput_{file_suffix}.csv")
    summary_file = os.path.join("data", f"summary_{file_suffix}.txt")

    os.makedirs("data", exist_ok=True)

    for path, header in [
        (benchmark_file, "benchmark,dataset,model,question_id,correct,data_sources,num_data_sources,selection_time,embedding_time,doc_select_time,generate_time,e2e_time,docs_tokens\n"),
        (ds_stats_file, "benchmark,dataset,question_id,data_source,duration,msg_size\n"),
        (throughput_file, "timestamp,completed,throughput_qps\n")
    ]:
        with open(path, "w") as f:
            f.write(header)

    benchmark = Benchmark(args.benchmark_path, args.benchmark)
    dataset_name = args.questions or "all"
    questions = benchmark.benchmark_data[args.questions] if args.questions else [q for b in benchmark.benchmark_data.values() for q in b.items()]
    question_pool = list(questions.items()) if isinstance(questions, dict) else questions

    # Load or create reproducible query set
    query_set, interarrivals = [], []
    if os.path.exists(args.query_set_path):
        print(f"[INFO] Loading query set from {args.query_set_path}")
        with open(args.query_set_path, "r") as f:
            data = json.load(f)
            query_set = [(qid, qdata) for qid, qdata in data["queries"]]
            interarrivals = data["interarrivals"]
    else:
        print(f"[INFO] Generating query set to match duration ~{args.duration}s")
        total_time = 0.0
        while total_time < args.duration:
            question_id, question_data = random.choice(question_pool)
            delay = np.random.exponential(scale=args.interarrival)
            query_set.append((question_id, question_data))
            interarrivals.append(delay)
            total_time += delay
        with open(args.query_set_path, "w") as f:
            json.dump({"queries": query_set, "interarrivals": interarrivals}, f)
        print(f"[INFO] Generated {len(query_set)} queries with total simulated interarrival time: {total_time:.2f}s")
        print(f"[DEBUG] Max interarrival: {max(interarrivals):.2f}s")

    query_iter = iter(query_set)
    interval_iter = iter(interarrivals)

    global completed_queries
    completed_queries = 0
    stop_event = asyncio.Event()
    timeout = aiohttp.ClientTimeout(total=600)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        log_task = asyncio.create_task(log_throughput(10, throughput_file, stop_event))
        tasks = []
        start_time = time.time()

        try:
            while True:
                try:
                    question_id, question_data = next(query_iter)
                    sleep_time = next(interval_iter)
                except StopIteration:
                    print(f"[INFO] SENT {len(tasks)} queries total. Done sending.")
                    break

                task = asyncio.create_task(handle_with_limit(
                    session, question_id, question_data,
                    benchmark, answer_file, benchmark_file, ds_stats_file,
                    args.benchmark, dataset_name
                ))
                tasks.append(task)
                await asyncio.sleep(sleep_time)

        finally:
            print("[INFO] Awaiting all in-flight tasks to finish...")
            await asyncio.gather(*tasks, return_exceptions=True)
            stop_event.set()
            await log_task
            print("[INFO] All queries have completed. Exiting.")

    total_time = time.time() - start_time
    throughput = completed_queries / total_time if total_time > 0 else 0.0
    print(f"\n[RESULT] Routing: {args.routing}")
    print(f"Total queries completed: {completed_queries}")
    print(f"Total time elapsed: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} QPS")

    with open(summary_file, "w") as f:
        f.write(f"[RESULT] Routing: {args.routing}\n")
        f.write(f"Total queries completed: {completed_queries}\n")
        f.write(f"Total time elapsed: {total_time:.2f}s\n")
        f.write(f"Throughput: {throughput:.2f} QPS\n")

if __name__ == "__main__":
    asyncio.run(main())

