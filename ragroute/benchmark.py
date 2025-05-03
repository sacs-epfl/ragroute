import json
import os

from ragroute.config import USR_DIR


class Benchmark:

    def __init__(self, benchmark_name: str):
        benchmark_file = os.path.join(USR_DIR, benchmark_name, "benchmark.json")
        with open(benchmark_file, 'r') as f:
            self.benchmark_data = json.load(f)
