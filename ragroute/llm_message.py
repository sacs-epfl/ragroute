from typing import Dict, List
from transformers import AutoTokenizer

from liquid import Template

from ragroute.config import CONTEXT_LENGTH, MODEL_NAME


def generate_llm_message(question: str, context, options: str) -> List[Dict[str, str]]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=None)

    contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, context[idx]["title"], context[idx]["content"]) for idx in range(len(context))]
    if len(contexts) == 0:
        contexts = [""]

    encoded_docs_tokens = tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:CONTEXT_LENGTH]
    context = tokenizer.decode(encoded_docs_tokens)

    medrag_system_prompt = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''
    medrag_prompt = Template('''
        Here are the relevant documents:
        {{context}}

        Here is the question:
        {{question}}

        Here are the potential choices:
        {{options}}

        Please think step-by-step and generate your output in json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}:
        ''')

    prompt_medrag = medrag_prompt.render(context=context, question=question, options=options)
    return [
                    {"role": "system", "content": medrag_system_prompt},
                    {"role": "user", "content": prompt_medrag}
            ], len(encoded_docs_tokens)
