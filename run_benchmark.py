import argparse
import asyncio
import json

import aiohttp
from tqdm import tqdm


from ragroute.benchmark import Benchmark


async def main():
    parser = argparse.ArgumentParser(description="Run a benchmark with RAGRoute.")
    parser.add_argument("--benchmark", type=str, default="MIRAGE", choices=["MIRAGE"], help="Benchmark name")
    args = parser.parse_args()

    questions: int = 0
    correct: int = 0
    
    # Load the benchmark
    benchmark = Benchmark(args.benchmark)
    async with aiohttp.ClientSession() as session:
        for dataset_name, questions in benchmark.benchmark_data.items():
            for question_id, data in tqdm(questions.items()):
                question = data['question']
                options = data['options']

                # Fire the HTTP request using asyncio

                # URL encode the question and options
                encoded_question = aiohttp.helpers.quote(question)
                encoded_options = aiohttp.helpers.quote(json.dumps(options))
                url = f"http://localhost:8000/query?q={encoded_question}&choices={encoded_options}"

                async with session.get(url) as response:
                    if response.status == 200:
                        json_response = await response.json()
                        print(json_response)
                    else:
                        print(f"Failed to fetch data for {question_id}: {response.status} - {await response.text()}")


if __name__ == "__main__":
    asyncio.run(main())
