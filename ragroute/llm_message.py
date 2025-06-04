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
    elif dataset == "wikipedia":
        contexts = []
        for j, (title, text) in enumerate(context):
            context.append(f"Document {j+1} [{title}]: {text}\n")
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

def generate_llm_message_wikipedia(question: str, top_docs: str, options: str, model: str) -> List[Dict[str, str]]:
    """
    Wikipedia is slightly different, as it uses a single context string.
    """
    model_info = MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(model_info["hf_name"], cache_dir=None)

    def prompt_context(ctx):
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are an assistant for answering multiple-choice questions. Below are relevant parts of documents retrieved for the question. "
            f"Use the provided context to choose the correct answer. If the context does not help, use the question and options alone.<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n\nGiven the following context, question, and four candidate answers (A, B, C, and D), choose the best answer.\n"
            f"Context:\n{ctx}\n"
            f"Question: {question}\n"
            f"A. {options[0]}\n"
            f"B. {options[1]}\n"
            f"C. {options[2]}\n"
            f"D. {options[3]}\n"
            f"Your response should end with \"The best answer is [the_answer_letter]\". Your response should be a single letter: A, B, C, or D. Only output one letter.<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\nThe best answer is"
        )
    
    docs_context = []
    for j, (title, text) in enumerate(top_docs):
        docs_context.append(f"Document {j+1} [{title}]: {text}\n")

    encoded_docs_tokens = tokenizer.encode("\n".join(docs_context), add_special_tokens=False)[:model_info["docs_context_length"]]
    context = tokenizer.decode(encoded_docs_tokens)
    
    ctx_prompt = prompt_context("".join(context))
    return [{"role": "user", "content": ctx_prompt}], len(encoded_docs_tokens)
