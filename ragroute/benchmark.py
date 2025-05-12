import json
import os
import re
from typing import Dict


class Benchmark:

    def __init__(self, benchmark_path: str, benchmark_name: str):
        self.benchmark_data = {}
        if benchmark_name == "MIRAGE":
            benchmark_file = os.path.join(benchmark_path, "%s.json" % benchmark_name)
            with open(benchmark_file, 'r') as f:
                self.benchmark_data = json.load(f)
        elif benchmark_name == "FeB4RAG":
            self.benchmark_data = {"FeB4RAG": {}}
            benchmark_file = os.path.join(benchmark_path, "%s.jsonl" % benchmark_name)
            with open(benchmark_file, "r") as f:
                for line in f:
                    obj = json.loads(line)
                    self.benchmark_data["FeB4RAG"][str(obj["_id"])] = {"question": obj["text"], "options": []}
        else:
            raise ValueError("Unsupported benchmark name: %s" % benchmark_name)

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
