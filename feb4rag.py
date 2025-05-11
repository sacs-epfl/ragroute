import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ollama import chat
from ollama import ChatResponse
from transformers import AutoTokenizer
from liquid import Template
import textwrap
import os
import json
from collections import defaultdict
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from ragroute.models.feb4rag.model_zoo import CustomModel, BeirModels


# USE online to compute everything from scratch for the system part
k = 32
device="cuda" if torch.cuda.is_available() else "cpu"
usr_dir = "/Users/mdevos"


# questions file
queries = {}
with open(os.path.join(usr_dir, "FeB4RAG/requests.jsonl"), "r") as f:
    for line in f:
        obj = json.loads(line)
        queries[str(obj["_id"])] = obj["text"]

# TODO
faiss_indexes = {}

# Copora and corresponding encoders for retrieval
CORPUS_MODEL_MAP = {
        "msmarco": ("e5-large", "custom"),
        "trec-covid": ("SGPT-5.8B-weightedmean-msmarco-specb-bitfit", "custom"),
        "nfcorpus": ("UAE-Large-V1", "custom"),
        "scidocs": ("all-mpnet-base-v2", "beir"),
        "nq": ("multilingual-e5-large", "custom"),
        "hotpotqa": ("ember-v1", "beir"),
        "fiqa": ("all-mpnet-base-v2", "beir"),
        "arguana": ("UAE-Large-V1", "custom"),
        "webis-touche2020": ("e5-base", "custom"),
        "dbpedia-entity": ("UAE-Large-V1", "custom"),
        "fever": ("UAE-Large-V1", "custom"),
        "climate-fever": ("UAE-Large-V1", "custom"),
        "scifact": ("gte-base", "beir"),
}
corpus_names = CORPUS_MODEL_MAP.keys()
# Group corpora by encoder for efficiency
model_to_corpora = defaultdict(list)
for corpus, (model_name, model_type) in CORPUS_MODEL_MAP.items():
    model_to_corpora[(model_name, model_type)].append(corpus)

source_to_id = {
  "arguana": 0,
  "climate-fever": 1,
  "dbpedia-entity": 2,
  "fever": 3,
  "fiqa": 4,
  "hotpotqa": 5,
  "msmarco": 6,
  "nfcorpus": 7,
  "nq": 8,
  "scidocs": 9,
  "scifact": 10,
  "trec-covid": 11,
  "webis-touche2020": 12
}

def encode_query(question):
    emb_per_corpora = {}
    for (model_name, model_type), corpora in model_to_corpora.items():
        model_loader = BeirModels(os.path.join(usr_dir, "FeB4RAG/dataset_creation/2_search/models"), specific_model=model_name) if model_type == "beir" else \
                       CustomModel(model_dir=os.path.join(usr_dir, "FeB4RAG/dataset_creation/2_search/models"), specific_model=model_name)
        model = model_loader.load_model(model_name, model_name_or_path=None, cuda=torch.cuda.is_available())

        emb = model.encode_queries(
            [question],
            batch_size=1,
            convert_to_tensor=False
        )[0]
        for corpus in corpora:
            emb_per_corpora[corpus] = emb
    
    queries_embed = []
    for corpus in corpus_names:
        queries_embed.append(emb_per_corpora[corpus])

    return queries_embed


def select_relevant_sources(queries_embed):
    # ADDED here just to group the parts of code together but it should not be defined each time again
    class CorpusRouter(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
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
            return self.fc3(x).squeeze()
        
    max_encoding_len = 4096        
    inputs = []

    for i, corpus in enumerate(corpus_names):
        padded_q = np.pad(queries_embed[i], (0, max_encoding_len - len(queries_embed[i])))

        with open(os.path.join(usr_dir, "FeB4RAG/dataset_creation/2_search/embeddings", corpus+"_"+CORPUS_MODEL_MAP[corpus][0]+"_stats.json")) as f:
                stat = json.load(f)
        centroid = np.array(stat["centroid"])
        centroid_p = np.pad(centroid, (0, max_encoding_len - len(centroid)))


        source_id = source_to_id[corpus]
        source_id_vec = np.eye(len(corpus_names))[source_id]  # one-hot
                
        features = np.concatenate([
            padded_q, centroid_p,
            source_id_vec
        ])
        inputs.append(features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)

    model_path = os.path.join(usr_dir, "FeB4RAG/dataset_creation/2_search/router_best_model.pt")
    model = CorpusRouter(input_dim=input_tensor.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor).squeeze()
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > 0.5).cpu().numpy()

    sources_corpora = [corpus for prediction, corpus in zip(predictions, corpus_names) if prediction]

    return sources_corpora

def retrieve_docs(query_embed, corpus_name, question_id, k):
    # TODO

    docs = []
    scores = []
    return docs, scores

def rerank(docs, scores, k):
    # TODO
    merged_docs, merged_scores = [], []
    return merged_docs, merged_scores

def generate_answer(question, context):
    # Please run the ollama server before by doing: OLLAMA_VERBOSE=1 /mnt/nfs/home/dpetresc/bin/ollama serve
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None)
    context_length = 128000
    max_tokens = 131072


    contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, context[idx]["title"], context[idx]["content"]) for idx in range(len(context))]
    if len(contexts) == 0:
        contexts = [""]
    context = tokenizer.decode(tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:context_length])

    system_prompt = '''You are a helpful assistant helping to answer user requests based on the provided search result. Your responses should directly address the user's request and must be based on the information obtained from the provided search results. You are forbidden to create new information that is not supported by these results. You must attribute your response to the source from the search results by including citations, for example, [1].'''
    prompt = Template(textwrap.dedent('''
        Here are the search results:
        {{context}}

        Here is the question:
        {{question}}
        '''))

    prompt = prompt.render(context=context, question=question)
    messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
            ]
    response_: ChatResponse = chat(model='llama3.1_extended', messages=messages, options={"num_predict": max_tokens})
    ans = response_['message']['content']

    #print("messages ", messages)
    print("ans ", ans)
    print()
    print()
    ans = []
    return ans

for query_id, query in queries.items():
    # encode query
    queries_embed = encode_query(query)
    #print(queries_embed)

    # SELECTION OF SOURCES / ROUTING
    sources_corpora = select_relevant_sources(queries_embed)
    print("selected sources ", sources_corpora)

    all_docs = []
    all_scores = []
    for i, source_corpus in enumerate(corpus_names):
        if source_corpus in sources_corpora:
            docs, scores = retrieve_docs(queries_embed[i], source_corpus, query_id, k)
            all_docs.extend(docs)
            all_scores.extend(scores)
        
    print("finished retrieving")

    # MERGING AND FILTERING to keep TOPK
    filtered_docs, filtered_scores = rerank(all_docs, all_scores, k)

    # GENERATIONs
    ans = generate_answer(query, filtered_docs)