from typing import Dict, List
from transformers import AutoTokenizer

from liquid import Template


def generate_llm_message(question: str, context, options: str) -> List[Dict[str, str]]:
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None)
    context_length = 128000
    max_tokens = 131072

    contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, context[idx]["title"], context[idx]["content"]) for idx in range(len(context))]
    if len(contexts) == 0:
        contexts = [""]
    context = tokenizer.decode(tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:context_length])

    medrag_system_prompt = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''
    medrag_prompt = Template('''
        Here are the relevant documents:
        {{context}}

        Here is the question:
        {{question}}

        Here are the potential choices:
        {{options}}

        Please think step-by-step and generate your output in json:
        ''')

    prompt_medrag = medrag_prompt.render(context=context, question=question, options=options)
    return [
                    {"role": "system", "content": medrag_system_prompt},
                    {"role": "user", "content": prompt_medrag}
            ]
