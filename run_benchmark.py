import argparse
import asyncio
import json
import os

import aiohttp
from tqdm import tqdm

from ragroute.benchmark import Benchmark
import time


async def fetch_answer(session, url):
    async with session.get(url) as response:
        if response.status == 200:
            return await response.json()
        else:
            print(f"Failed to fetch data: {response.status} - {await response.text()}")
            return None


async def main():
    parser = argparse.ArgumentParser(description="Run a benchmark with RAGRoute.")
    parser.add_argument("--benchmark", type=str, default="FeB4RAG", choices=["MIRAGE", "FeB4RAG"], help="Benchmark name")
    parser.add_argument("--benchmark-path", type=str, default="data/benchmark", help="Path to the benchmark data")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel requests to send")
    parser.add_argument("--routing", type=str, required=True, choices=["ragroute", "all", "random", "none"], help="Routing method to use")
    parser.add_argument("--questions", type=str, default=None, choices=['medqa', 'medmcqa', 'pubmedqa', 'bioasq', 'mmlu'], help="The questions to use for the benchmark")  # TODO add questions from FeB4RAG
    args = parser.parse_args()

    file_suffix = f"{args.benchmark}_{args.routing}_p{args.parallel}"

    benchmark_file: str = os.path.join("data", f"_benchmark_{file_suffix}.csv")
    ds_stats_file: str = os.path.join("data", f"_ds_stats_{file_suffix}.csv")
    answer_file: str = os.path.join("data", f"_answers_{file_suffix}.jsonl")


    batch_log_path = os.path.join("data", f"_batch_timings_{file_suffix}.csv")
    if not os.path.exists(batch_log_path):
        with open(batch_log_path, "w") as f:
            f.write("successes,batch_time,total_sent\n")


    if not os.path.exists(benchmark_file):
        with open(benchmark_file, "w") as f:
            f.write("benchmark,dataset,model,question_id,correct,data_sources,num_data_sources,selection_time,embedding_time,doc_select_time,generate_time,e2e_time,docs_tokens\n")

    if not os.path.exists(ds_stats_file):
        with open(ds_stats_file, "w") as f:
            f.write("benchmark,dataset,question_id,data_source,duration,msg_size\n")

    # Load the benchmark file and create the set of all question_ids that are there
    existing_question_ids = set()
    with open(benchmark_file, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) > 3:
                existing_question_ids.add(parts[3])

    num_questions: int = 0
    num_correct: int = 0
    
    # Load the benchmark
    benchmark = Benchmark(args.benchmark_path, args.benchmark)
    async with aiohttp.ClientSession() as session:
        question_banks = sorted(benchmark.benchmark_data.keys())
        if args.questions is not None:
            question_banks = [args.questions]

        for question_bank in question_banks:
            questions = benchmark.benchmark_data[question_bank]
            question_items = sorted(questions.items())
            for i in tqdm(range(0, len(question_items), args.parallel)):
                batch = question_items[i:i + args.parallel]
                tasks = []

                for question_id, question_data in batch:
                    if question_id in existing_question_ids:
                        print(f"Skipping {question_id} as it is already processed.")
                        continue
                    question = question_data['question']
                    options = question_data['options']

                    encoded_question = aiohttp.helpers.quote(question)
                    encoded_options = aiohttp.helpers.quote(json.dumps(options))
                    url = f"http://localhost:8000/query?q={encoded_question}&choices={encoded_options}&qid={question_id}"

                    task = fetch_answer(session, url)
                    tasks.append(task)


                batch_start_time = time.time()
                # Run the batch concurrently
                results = await asyncio.gather(*tasks)
                # TODO
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                # Count valid responses (skip None and generation failures)
                valid_results = [r for r in results if r and r["metadata"]["generate_time"] != -1]
                question_ids = [qid for qid,_ in batch]
                if valid_results:
                    with open(batch_log_path, "a") as f:
                        f.write(f"{json.dumps(question_ids)},{len(valid_results)},{batch_duration:.4f},{len(tasks)}\n")

                for result in results:
                    if not result:
                        print("Error: No result returned from the server.")
                        continue

                    # Process the question result
                    is_correct = benchmark.check_mirage_answer(question_data, result["answer"]) if args.benchmark == "MIRAGE" else True
                    num_questions += 1
                    num_correct += int(is_correct)

                    # Record the answer
                    with open(answer_file, "a") as f:
                        f.write(json.dumps({"question_id": question_id, "answer": result["answer"]}) + "\n")

                    metadata = result["metadata"]
                    data_sources = ":".join(metadata["data_sources"])

                    with open(benchmark_file, "a") as f:
                        f.write(f"{args.benchmark},{question_bank},{metadata['llm']},{question_id},{int(is_correct)},{data_sources},{len(metadata['data_sources'])},"
                                f"{metadata['selection_time']},{metadata['embedding_time']},{metadata['doc_select_time']},"
                                f"{metadata['generate_time']},{metadata['e2e_time']},{metadata['docs_tokens']}\n")
                        
                    with open(ds_stats_file, "a") as f:
                        for data_source, stats in metadata["data_sources_stats"].items():
                            f.write(f"{args.benchmark},{question_bank},{question_id},{data_source},{stats['duration']},{stats['message_size']}\n")

                    if args.benchmark == "MIRAGE":
                        print(f"--> Score: {num_correct}/{num_questions}")


if __name__ == "__main__":
    asyncio.run(main())
