import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from ollama import chat
from ollama import ChatResponse
from transformers import AutoTokenizer
from liquid import Template
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer
import os
import json
import pickle
import faiss


# USE online to compute everything from scratch for the system part
online = True
k = 32
device="cuda" if torch.cuda.is_available() else "cpu"
usr_dir = "/mnt/nfs/home/dpetresc"

# query encoder for routing and retrieval
class CustomizeSentenceTransformer(SentenceTransformer): # change the default pooling "MEAN" to "CLS"
    def _load_auto_model(self, model_name_or_path, *args, **kwargs):
        cache_path = os.path.join(usr_dir, ".cache/torch/sentence_transformers", model_name_or_path)
        #transformer_model = Transformer(model_name_or_path)
        print("Sentence-transformers cache ", cache_path)
        transformer_model = Transformer(model_name_or_path, cache_dir=cache_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'cls')
        return [transformer_model, pooling_model]
embedding_function = CustomizeSentenceTransformer("ncbi/MedCPT-Query-Encoder", device=device)
embedding_function.eval()



# FOR EVALUATION / REWARD
# questions and answers file
benchmark_file = os.path.join(usr_dir, "MIRAGE/benchmark.json")
def load_benchmark(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
benchmark_data = load_benchmark(benchmark_file)

faiss_indexes = {}

# added by me to go faster...
cache_jsonl = {}

def encode_query(question, question_id, dataset_name):
    if online:
        with torch.no_grad():
                query_embed = embedding_function.encode([question])
    else:
        retrieval_cache_dir = os.path.join(usr_dir, "MedRAG/retrieval_cache/") # bioasq	medmcqa  medqa	mmlu  pubmedqa
        emb_path = os.path.join(retrieval_cache_dir, dataset_name, "emb_queries", question_id+".npy")
        query_embed = np.load(emb_path)

    return query_embed


def select_relevant_sources(query_embed):
    # ADDED here just to group the parts of code together but it should not be defined each time again
    class CorpusRoutingNN(nn.Module):
        def __init__(self, input_dim):
            super(CorpusRoutingNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 256)
            self.ln1 = nn.LayerNorm(256)
            self.dropout1 = nn.Dropout(0.4)

            self.fc2 = nn.Linear(256, 128)
            self.ln2 = nn.LayerNorm(128)
            self.dropout2 = nn.Dropout(0.4)

            self.fc3 = nn.Linear(128, 1)

        def forward(self, x):
            x = F.relu(self.ln1(self.fc1(x)))
            x = self.dropout1(x)
            x = F.relu(self.ln2(self.fc2(x)))
            x = self.dropout2(x)
            return self.fc3(x)
    
    corpus_names = ["pubmed", "textbooks", "statpearls", "wikipedia"]

    if online:
        inputs = []
        
        for corpus in corpus_names:
            stats_file = os.path.join(usr_dir, "MedRAG/routing/", f"{corpus}_stats.json")
            with open(stats_file, "r") as f:
                corpus_stats = json.load(f)

            centroid = np.array(corpus_stats["centroid"], dtype=np.float32)
            features = np.concatenate([query_embed.flatten(), centroid])
            inputs.append(features)
        
        # Load scaler
        scaler_path = os.path.join(usr_dir, "MedRAG/routing/preprocessed_data.pkl")
        with open(scaler_path, "rb") as f:
            _, _, _, scaler, _ = pickle.load(f)
        inputs = scaler.transform(inputs)

        input_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)

        model_path = os.path.join(usr_dir, "MedRAG/routing/best_model.pth")
        model = CorpusRoutingNN(input_tensor.shape[1]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            outputs = model(input_tensor).squeeze()
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).cpu().numpy()

        sources_corpora = [corpus for prediction, corpus in zip(predictions, corpus_names) if prediction]
    else:
        sources_corpora = corpus_names

    return sources_corpora

def retrieve_docs(query_embed, corpus_name, dataset_name, k):
    def idx2txt(indices):

        results = []
        for i in indices:
            source = i["source"]
            index = i["index"]

            # added by me to go faster...
            # Checks if the file's lines are already cached
            if source not in cache_jsonl:
                file_path = os.path.join(usr_dir, "MedRAG/corpus", corpus_name, "chunk", f"{source}.jsonl")
                with open(file_path, "r") as file:
                    # Cache raw lines as strings instead of fully parsed JSON
                    cache_jsonl[source] = file.read().strip().split("\n")

            # Parse the specific line at the requested index
            line = cache_jsonl[source][index]
            results.append(json.loads(line))  # Parse only when needed
        return results
    
    if online:
        index_dir = os.path.join(usr_dir, "MedRAG/corpus", corpus_name, "index", "ncbi/MedCPT-Article-Encoder")

        if index_dir not in faiss_indexes:
            index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
            metadatas = [json.loads(line) for line in open(os.path.join(index_dir, "metadatas.jsonl")).read().strip().split('\n')]
            faiss_indexes[index_dir] = (index, metadatas)
        else:
            index, metadatas = faiss_indexes[index_dir]

        res_ = index.search(query_embed, k=k)
        scores = res_[0][0].tolist()

        # from faiss idx to corresponding source and index
        indices = [metadatas[i] for i in res_[1][0]]
        # get the corresponding documents
        docs = idx2txt(indices)
    else:
        # retrieve_docs_from_file to go faster
        corpus_path = os.path.join(usr_dir, "MedRAG/retrieval_cache", dataset_name, corpus_name)
        
        # Paths to the files
        texts_file = os.path.join(corpus_path, f"top_32_{question_id}_texts.json")
        scores_file = os.path.join(corpus_path, f"top_32_{question_id}_scores.txt")

        # Read texts
        with open(texts_file, "r") as f_texts:
            texts = json.load(f_texts)
        # Read scores
        with open(scores_file, "r") as f_scores:
            scores = [float(line.strip()) for line in f_scores]

        # Retrieve only the top-k results
        docs = texts[:k]
        scores = scores[:k]

    return docs, scores

def rerank(docs, scores, k):
    # Just rerank based on scores for the moment
    sorted_indices = np.argsort(scores)[::-1]  # Sort scores descending
    merged_docs = [docs[i] for i in sorted_indices][:k]
    merged_scores = [scores[i] for i in sorted_indices][:k]

    return merged_docs, merged_scores

def generate_answer(question, context, options):
    # Please run the ollama server before by doing: OLLAMA_VERBOSE=1 /mnt/nfs/home/dpetresc/bin/ollama serve

    # TODO change for another more recent version
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
    messages=[
                    {"role": "system", "content": medrag_system_prompt},
                    {"role": "user", "content": prompt_medrag}
            ]
    response_: ChatResponse = chat(model='llama3.1_extended', messages=messages, options={"num_predict": max_tokens})
    ans = response_['message']['content']

    #print("messages ", messages)
    print("ans ", ans)
    print()
    print()

    return ans

def verify_answer(question_id, ans):
    # TODO implement
    check = 1
    return check

for dataset_name, questions in benchmark_data.items():
    for question_id, data in tqdm(questions.items()):
        question = data['question']
        options = data['options']
        print(question)

        # encode query
        query_embed = encode_query(question, question_id, dataset_name)

        # SELECTION OF SOURCES / ROUTING
        sources_corpora = select_relevant_sources(query_embed)
        print("selected sources ", sources_corpora)

        all_docs = []
        all_scores = []
        for source_corpus in sources_corpora:
            docs, scores = retrieve_docs(query_embed, source_corpus, dataset_name, k)
            all_docs.extend(docs)
            all_scores.extend(scores)
        
        print("finished retrieving")

        # MERGING AND FILTERING to keep TOPK
        filtered_docs, filtered_scores = rerank(all_docs, all_scores, k)

        # GENERATION
        ans = generate_answer(question, filtered_docs, options)
