import argparse
import asyncio
import json
import os

import aiohttp
from tqdm import tqdm

from ragroute.benchmark import Benchmark


async def fetch_answer(session, url):
    async with session.get(url) as response:
        if response.status == 200:
            return await response.json()
        else:
            print(f"Failed to fetch data: {response.status} - {await response.text()}")
            return None


async def main():
    parser = argparse.ArgumentParser(description="Run a benchmark with RAGRoute.")
    parser.add_argument("--benchmark", type=str, default="MMLU", choices=["MIRAGE", "FeB4RAG", "MMLU"], help="Benchmark name")
    parser.add_argument("--benchmark-path", type=str, default="data/benchmark", help="Path to the benchmark data")
    parser.add_argument("--save-logs-dir", dest="save_logs_dir", type=str, default="data", help="Path to the benchmark data")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel requests to send")
    parser.add_argument("--routing", type=str, required=True, choices=["ragroute", "all", "random", "none"], help="Routing method to use")
    parser.add_argument("--questions", type=str, default=None, choices=['medqa', 'medmcqa', 'pubmedqa', 'bioasq', 'high_school_microeconomics', 'international_law', 'business_ethics', 'high_school_biology', 'college_mathematics'], help="The questions to use for the benchmark")
    args = parser.parse_args()

    os.makedirs(args.save_logs_dir, exist_ok=True)
    if args.questions is not None:
        benchmark_file: str = os.path.join("%s" % args.save_logs_dir, "benchmark_%s_%s_%s.csv" % (args.benchmark, args.routing, args.questions))
        ds_stats_file: str = os.path.join("%s" % args.save_logs_dir, "ds_stats_%s_%s_%s.csv" % (args.benchmark, args.routing, args.questions))
        answer_file: str = os.path.join("%s" % args.save_logs_dir, "answers_%s_%s_%s.jsonl" % (args.benchmark, args.routing, args.questions))
        top_docs_file: str = os.path.join("%s" % args.save_logs_dir, "top_docs_%s_%s_%s.jsonl" % (args.benchmark, args.routing, args.questions))
    else:
        benchmark_file: str = os.path.join("%s" % args.save_logs_dir, "benchmark_%s_%s.csv" % (args.benchmark, args.routing))
        ds_stats_file: str = os.path.join("%s" % args.save_logs_dir, "ds_stats_%s_%s.csv" % (args.benchmark, args.routing))
        answer_file: str = os.path.join("%s" % args.save_logs_dir, "answers_%s_%s.jsonl" % (args.benchmark, args.routing))
        top_docs_file: str = os.path.join("%s" % args.save_logs_dir, "top_docs_%s_%s.jsonl" % (args.benchmark, args.routing))

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

        all_question_batches = {}

        for question_bank in question_banks:
            questions = benchmark.benchmark_data[question_bank]
            # For reproducibility 
            order_path = f"data/question_order_{args.benchmark}_{question_bank}.json"

            if os.path.exists(order_path):
                with open(order_path) as f:
                    ordered_ids = json.load(f)
                question_items = [(qid, questions[qid]) for qid in ordered_ids if qid in questions]
                print(f"Reusing saved question order from {order_path}")
            else:
                question_items = list(questions.items())
                with open(order_path, "w") as f:
                    json.dump([qid for qid, _ in question_items], f)
                print(f"Saved new question order to {order_path}")

            all_question_batches[question_bank] = question_items

            for i in tqdm(range(0, len(question_items), args.parallel)):
                tasks = []
                raw_batch = question_items[i:i + args.parallel]

                # Filter out already-processed questions
                batch = [(qid, qdata) for qid, qdata in raw_batch if qid not in existing_question_ids]
                
                if not batch:
                    continue

                for question_id, question_data in batch:
                    print(question_id)
                    #print(question_data)
                    question = question_data['question']
                    options = question_data['options']

                    encoded_question = aiohttp.helpers.quote(question)
                    encoded_options = aiohttp.helpers.quote(json.dumps(options))
                    url = f"http://localhost:8000/query?q={encoded_question}&choices={encoded_options}&qid={question_id}"

                    task = fetch_answer(session, url)
                    tasks.append(task)

                # Run the batch concurrently
                results = await asyncio.gather(*tasks)

                for (question_id, question_data), result in zip(batch, results):
                    if not result:
                        print("Error: No result returned from the server.")
                        continue

                    # Process the question result
                    if args.benchmark == "MIRAGE":
                        is_correct = benchmark.check_mirage_answer(question_data, result["answer"])
                    elif args.benchmark == "MMLU":
                        is_correct = benchmark.check_mmlu_answer(question_data, result["answer"])
                    else:
                        is_correct = True
                    num_questions += 1
                    num_correct += int(is_correct)

                    # Record the answer
                    with open(answer_file, "a") as f:
                        f.write(json.dumps({"question_id": question_id, "answer": result["answer"]}) + "\n")
                    
                    # Save top documents used in reranking
                    top_docs_record = {
                        "question_id": question_id,
                        "top_docs": result["metadata"]["top_docs"]
                    }
                    with open(top_docs_file, "a") as f:
                        f.write(json.dumps(top_docs_record) + "\n")

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
