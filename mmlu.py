import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from ollama import chat
import pickle 
import faiss

# === Config ===
BASE_DIR = "/mnt/nfs/home/dpetresc/Retrieval-QA-Benchmark_backup/euromlsys/new_submission"
WIKI_DIR = "/mnt/nfs/home/dpetresc/wiki_dataset/dpr_wiki_index"
CLUSTER_DIR = os.path.join(WIKI_DIR, "faiss_clusters")
RETRIEVAL_DIR = os.path.join(BASE_DIR, "top_10_results_dpr")

NUM_CLUSTERS = 10
K = 10
THRESHOLD = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

index_cache = {}


def encode_query(question):
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        embeddings = encoder(**inputs).pooler_output
    return embeddings.cpu().numpy().astype(np.float32)[0]

def select_relevant_sources(query_embed):
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
    
    CLUSTER_STATS_FILE = os.path.join(CLUSTER_DIR, "cluster_stats.json")

    router_model = CorpusRoutingNN(768 + 768).to(device)
    router_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "cluster_router_output", "best_model.pth")))
    router_model.eval()

    with open(CLUSTER_STATS_FILE) as f:
        cluster_stats = json.load(f)
    centroids = [np.array(c["centroid"], dtype=np.float32) for c in cluster_stats]
    with open(os.path.join(BASE_DIR, "cluster_router_output", "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    features = [np.concatenate([query_embed, c]) for c in centroids]
    features_scaled = scaler.transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = router_model(features_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()

    selected = [i for i, p in enumerate(probs) if p > THRESHOLD]
    return selected

def retrieve_docs(query_embed, cid):
    WIKI_DIR = "/mnt/nfs/home/dpetresc/wiki_dataset/dpr_wiki_index"
    CLUSTER_DIR = os.path.join(WIKI_DIR, "faiss_clusters")
    NORM_INDEX_DIR = os.path.join(CLUSTER_DIR, "normalized_indexes")
    TITLE_FILE = os.path.join(WIKI_DIR, "wiki_titles.txt")
    TEXT_FILE = os.path.join(WIKI_DIR, "wiki_texts.txt")

    with open(os.path.join(CLUSTER_DIR, "cluster_index_map.pkl"), "rb") as f:
        cluster_index_map = pickle.load(f)
    with open(TITLE_FILE, "r", encoding="utf-8") as f:
        all_titles = f.read().splitlines()
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        all_texts = f.read().splitlines()

    # Load normalized FAISS index for this cluster
    if cid not in index_cache:
        index_path = os.path.join(NORM_INDEX_DIR, f"faiss_index_{cid}_normalized.index")
        index_cache[cid] = faiss.read_index(index_path)
    index = index_cache[cid]
    cluster_doc_ids = cluster_index_map[cid]

    # Normalize the query
    query_vec = query_embed.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query_vec)

    scores, local_indices = index.search(query_vec, K)

    docs, doc_scores = [], []
    for i, (score, local_idx) in enumerate(zip(scores[0], local_indices[0])):
        global_idx = cluster_doc_ids[local_idx]
        title = all_titles[global_idx]
        text = all_texts[global_idx]
        docs.append((title, text))
        doc_scores.append(score)

    return docs, doc_scores

def rerank(docs, scores, k):
    # Just rerank based on scores for the moment
    sorted_indices = np.argsort(scores)[::-1]  # Sort scores descending
    merged_docs = [docs[i] for i in sorted_indices][:k]
    merged_scores = [scores[i] for i in sorted_indices][:k]

    return merged_docs, merged_scores

def generate_answer(item, top_docs):
    def prompt_context(item, ctx):
        q, c = item["question"], item["choices"]
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are an assistant for answering multiple-choice questions. Below are relevant parts of documents retrieved for the question. "
            f"Use the provided context to choose the correct answer. If the context does not help, use the question and options alone.<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n\nGiven the following context, question, and four candidate answers (A, B, C, and D), choose the best answer.\n"
            f"Context:\n{ctx}\n"
            f"Question: {q}\n"
            f"A. {c[0]}\n"
            f"B. {c[1]}\n"
            f"C. {c[2]}\n"
            f"D. {c[3]}\n"
            f"Your response should end with \"The best answer is [the_answer_letter]\". Your response should be a single letter: A, B, C, or D. Only output one letter.<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\nThe best answer is"
        )
    
    context = []
    for j, (title, text) in enumerate(top_docs):
        context.append(f"Document {j+1} [{title}]: {text}\n")
    
    ctx_prompt = prompt_context(item, "".join(context))

    response = chat(model="llama3.1_extended", messages=[{"role": "user", "content": ctx_prompt}], options={"num_predict": 131072})
    output = response["message"]["content"]
    return output.split("The best answer is")[-1].strip().replace(".", "").replace('"', "").strip(), output

def verify_answer(item, prediction):
    gold = chr(65 + item["answer"])
    return int(prediction == gold)


dataset = load_dataset("cais/mmlu", "all", split="test")
TARGET_SUBJECTS = {
    "high_school_microeconomics", "international_law", "high_school_mathematics",
    "college_mathematics", "business_ethics", "high_school_biology", "astronomy",
    "philosophy", "public_relations", "college_biology", "electrical_engineering",
    "conceptual_physics", "professional_psychology"
}

for i, item in enumerate(tqdm(dataset)):
    if item["subject"] not in TARGET_SUBJECTS:
        continue

    question_id = f"question_{i}"
    question = item["question"]
    options = item["choices"]
    formatted_q = "\n".join([question, " | ".join(options)])

    try:
        query_embed = encode_query(formatted_q)

        sources_corpora = select_relevant_sources(query_embed)

        all_docs = []
        all_scores = []
        for id in sources_corpora:
            docs, scores = retrieve_docs(query_embed, id)
            all_docs.extend(docs)
            all_scores.extend(scores)
        
        top_docs, top_scores = rerank(all_docs, all_scores, K)

        pred, raw = generate_answer(item, top_docs)

        right_answer = verify_answer(item, pred)
        print("Correct answer: ", right_answer == 1)

    except Exception as e:
        print(f"Error on {question_id}: {e}")
        continue
