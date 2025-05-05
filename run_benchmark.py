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

    num_questions: int = 0
    num_correct: int = 0
    
    # Load the benchmark
    benchmark = Benchmark(args.benchmark)
    async with aiohttp.ClientSession() as session:
        for dataset_name, questions in benchmark.benchmark_data.items():
            for question_id, question_data in tqdm(questions.items()):
                question = question_data['question']
                options = question_data['options']

                # Fire the HTTP request using asyncio

                # URL encode the question and options
                encoded_question = aiohttp.helpers.quote(question)
                encoded_options = aiohttp.helpers.quote(json.dumps(options))
                url = f"http://localhost:8000/query?q={encoded_question}&choices={encoded_options}"

                async with session.get(url) as response:
                    if response.status == 200:
                        json_response = await response.json()
                        print(json_response)

                        # Check if the answer is correct
                        is_correct: bool = benchmark.check_mirage_answer(question_data, json_response["answer"])
                        if is_correct:
                            num_correct += 1
                        num_questions += 1
                            
                        print("--> Score: %d/%d" % (num_correct, num_questions))
                    else:
                        print(f"Failed to fetch data for {question_id}: {response.status} - {await response.text()}")


if __name__ == "__main__":
    asyncio.run(main())
