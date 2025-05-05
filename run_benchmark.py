import argparse
import asyncio
import json
import os

import aiohttp
from tqdm import tqdm

from ragroute.benchmark import Benchmark


async def fetch_answer(session, url, question_data, benchmark):
    async with session.get(url) as response:
        if response.status == 200:
            return await response.json()
        else:
            print(f"Failed to fetch data: {response.status} - {await response.text()}")
            return None


async def main():
    parser = argparse.ArgumentParser(description="Run a benchmark with RAGRoute.")
    parser.add_argument("--benchmark", type=str, default="MIRAGE", choices=["MIRAGE"], help="Benchmark name")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel requests to send")
    args = parser.parse_args()

    benchmark_file: str = os.path.join("data", "benchmark_%s.csv" % args.benchmark)
    ds_durations_file: str = os.path.join("data", "ds_durations_%s.csv" % args.benchmark)

    if not os.path.exists(benchmark_file):
        with open(benchmark_file, "w") as f:
            f.write("benchmark,dataset,question_id,correct,data_sources,num_data_sources,selection_time,embedding_time,doc_select_time,generate_time,e2e_time\n")

    if not os.path.exists(ds_durations_file):
        with open(ds_durations_file, "w") as f:
            f.write("benchmark,dataset,question_id,data_source,duration\n")

    # Load the benchmark file and create the set of all question_ids that are there
    existing_question_ids = set()
    with open(benchmark_file, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) > 2:
                existing_question_ids.add(parts[2])

    num_questions: int = 0
    num_correct: int = 0
    
    # Load the benchmark
    benchmark = Benchmark(args.benchmark)
    async with aiohttp.ClientSession() as session:
        for dataset_name, questions in benchmark.benchmark_data.items():
            question_items = list(questions.items())
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
                    url = f"http://localhost:8000/query?q={encoded_question}&choices={encoded_options}"

                    task = fetch_answer(session, url, question_data, benchmark)
                    tasks.append(task)

                # Run the batch concurrently
                results = await asyncio.gather(*tasks)

                for result in results:
                    # Process the question result
                    if "answer" in result:
                        is_correct = benchmark.check_mirage_answer(question_data, result["answer"])
                    else:
                        print(f"Error: No answer in result for question {question_id}")
                        is_correct = False

                    num_questions += 1
                    num_correct += bool(is_correct)

                    metadata = result["metadata"]
                    data_sources = ":".join(metadata["data_sources"])

                    with open(benchmark_file, "a") as f:
                        f.write(f"{args.benchmark},{dataset_name},{question_id},{int(is_correct)},{data_sources},{len(metadata['data_sources'])},"
                                f"{metadata['selection_time']},{metadata['embedding_time']},{metadata['doc_select_time']},"
                                f"{metadata['generate_time']},{metadata['e2e_time']}\n")
                        
                    with open(ds_durations_file, "a") as f:
                        for data_source, duration in metadata["time_per_data_source"].items():
                            f.write(f"{args.benchmark},{dataset_name},{question_id},{data_source},{duration}\n")

                    print(f"--> Score: {num_correct}/{num_questions}")


if __name__ == "__main__":
    asyncio.run(main())
