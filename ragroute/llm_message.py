from typing import Dict, List
from transformers import AutoTokenizer

from liquid import Template

from ragroute.config import MODELS, SYSTEM_PROMPTS, USER_PROMPT_TEMPLATES


def generate_llm_message(dataset: str, question: str, context, options: str, model: str) -> List[Dict[str, str]]:
    model_info = MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(model_info["hf_name"], cache_dir=None)

    if dataset == "medrag":
        contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, context[idx]["title"], context[idx]["content"]) for idx in range(len(context))]
    elif dataset == "feb4rag":
        contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, context[idx].get("title") or f"Doc {idx}", context[idx]["text"]) for idx in range(len(context))]
    if len(contexts) == 0:
        contexts = [""]

    encoded_docs_tokens = tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:model_info["docs_context_length"]]
    context = tokenizer.decode(encoded_docs_tokens)

    medrag_prompt = Template(USER_PROMPT_TEMPLATES[dataset])

    prompt_medrag = medrag_prompt.render(context=context, question=question, options=options)
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[dataset]},
        {"role": "user", "content": prompt_medrag}
        ], len(encoded_docs_tokens)
