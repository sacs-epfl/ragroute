import argparse
import asyncio
import json

import aiohttp
from tqdm import tqdm

from ragroute.benchmark import Benchmark


async def fetch_answer(session, url, question_data, benchmark):
    async with session.get(url) as response:
        if response.status == 200:
            json_response = await response.json()
            is_correct = benchmark.check_mirage_answer(question_data, json_response["answer"])
            return is_correct
        else:
            print(f"Failed to fetch data: {response.status} - {await response.text()}")
            return False


async def main():
    parser = argparse.ArgumentParser(description="Run a benchmark with RAGRoute.")
    parser.add_argument("--benchmark", type=str, default="MIRAGE", choices=["MIRAGE"], help="Benchmark name")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel requests to send")
    args = parser.parse_args()

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
                    question = question_data['question']
                    options = question_data['options']

                    encoded_question = aiohttp.helpers.quote(question)
                    encoded_options = aiohttp.helpers.quote(json.dumps(options))
                    url = f"http://localhost:8000/query?q={encoded_question}&choices={encoded_options}"

                    task = fetch_answer(session, url, question_data, benchmark)
                    tasks.append(task)

                # Run the batch concurrently
                results = await asyncio.gather(*tasks)

                # Process results
                num_questions += len(results)
                num_correct += sum(results)

                print(f"--> Score: {num_correct}/{num_questions}")


if __name__ == "__main__":
    asyncio.run(main())
