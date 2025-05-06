import json
import os
import re
from typing import Dict

from ragroute.config import USR_DIR


class Benchmark:

    def __init__(self, benchmark_name: str):
        benchmark_file = os.path.join(USR_DIR, benchmark_name, "benchmark.json")
        with open(benchmark_file, 'r') as f:
            self.benchmark_data = json.load(f)

    def check_mirage_answer(self, data_question: Dict, llm_output: str) -> bool:
        def locate_answer(sentence: str):
            ans = re.findall(r"^\s*(A|B|C|D)$", sentence)
            if len(ans) > 0:
                return ans[0].upper()
            
            ans = re.findall(r"^\s*(A|B|C|D) or", sentence)
            if len(ans) > 0:
                return ans[0].upper()
            
            ans = re.findall(r"^\s*(A|B|C|D) and", sentence)
            if len(ans) > 0:
                return ans[0].upper()
                
            ans = re.findall(r"^\s*(A|B|C|D)/", sentence)
            if len(ans) > 0:
                return ans[0].upper()
            
            ans = re.findall(r"^\s*(A|B|C|D),", sentence)
            if len(ans) > 0:
                return ans[0].upper()
            
            ans = re.findall(r"[Oo]ption (A|B|C|D)", sentence)
            if len(ans) > 0:
                return ans[0]

            ans = re.findall(r":\s*(A|B|C|D)", sentence)
            if len(ans) > 0:
                return ans[0].upper()

            ans = re.findall(r"^\s*(A|B|C|D)\.", sentence)
            if len(ans) > 0:
                return ans[0].upper()

            ans = re.findall(r"^\s*(A|B|C|D)\"", sentence)
            if len(ans) > 0:
                return ans[0].upper()
            
            ans = re.findall(r"^\s*(A|B|C|D):", sentence)
            if len(ans) > 0:
                return ans[0].upper()
            return ""

        answer_list = ["A", "B", "C", "D"]

        ans = locate_answer(llm_output.split('"answer_choice": "')[-1].strip())

        if ans in answer_list and data_question["answer"] in answer_list:
            return ans == data_question["answer"]
        return False
