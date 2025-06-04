import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
import random
from collections import defaultdict
import pickle

# === Config ===
WIKI_DIR = "/mnt/nfs/home/dpetresc/wiki_dataset/dpr_wiki_index"
CLUSTER_DIR = os.path.join(WIKI_DIR, "faiss_clusters")
CLUSTER_STATS_FILE = os.path.join(CLUSTER_DIR, "cluster_stats.json")
RETRIEVAL_DIR = "/mnt/nfs/home/dpetresc/Retrieval-QA-Benchmark_backup/euromlsys/new_submission/top_10_results_dpr"
QUESTIONS_FILE = os.path.join(RETRIEVAL_DIR, "questions.json")
NUM_CLUSTERS = 10
SEED = 42
BATCH_SIZE = 128
EPOCHS = 150
THRESHOLD = 0.5
OUTPUT_DIR = "./cluster_router_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Target subjects (with >3% improvement) ===
TARGET_SUBJECTS = {
    "high_school_microeconomics", "international_law", "high_school_mathematics",
    "college_mathematics", "business_ethics", "high_school_biology", "astronomy",
    "philosophy", "public_relations", "college_biology", "electrical_engineering",
    "conceptual_physics", "professional_psychology"
}

# === Reproducibility ===
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# === Load DPR ===
tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)

def encode_query_dpr(question: str) -> np.ndarray:
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        embeddings = encoder(**inputs).pooler_output
    return embeddings.cpu().numpy().astype(np.float32)[0]

# === Load MMLU and questions.json ===
dataset = load_dataset("cais/mmlu", "all", split="test")
with open(QUESTIONS_FILE, "r") as f:
    formatted_questions = json.load(f)

def preproc(d): return "\n".join([d["question"], " | ".join(d["choices"])])

filtered = [x for x in dataset]
formatted_to_item = {preproc(item): item for item in filtered}

# === Load Cluster Stats ===
with open(CLUSTER_STATS_FILE) as f:
    cluster_stats = json.load(f)
centroids = [np.array(c["centroid"], dtype=np.float32) for c in cluster_stats]

# === Build Samples ===
samples = []
meta_test_info = []  # for evaluation

for i, formatted_q in enumerate(tqdm(formatted_questions)):
    if formatted_q not in formatted_to_item:
        continue
    item = formatted_to_item[formatted_q]
    if item["subject"] not in TARGET_SUBJECTS:
        continue

    cluster_file = os.path.join(RETRIEVAL_DIR, f"question_{i}_cluster_ids.txt")
    if not os.path.exists(cluster_file):
        continue

    q_emb = encode_query_dpr(formatted_q)

    with open(cluster_file) as f:
        retrieved_clusters = set(int(line.strip()) for line in f if line.strip().isdigit())

    for cid in range(NUM_CLUSTERS):
        features = np.concatenate([q_emb, centroids[cid]])
        label = 1 if cid in retrieved_clusters else 0
        samples.append((features, label))
        meta_test_info.append((i, cid, label))

print(f"Total samples: {len(samples)}")

# === Dataset Class ===
class ClusterDataset(Dataset):
    def __init__(self, data):
        self.features = [torch.tensor(x, dtype=torch.float32) for x, _ in data]
        self.labels = [torch.tensor(y, dtype=torch.float32) for _, y in data]
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

# === Model ===
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

# === Split per query ===
query_to_samples = defaultdict(list)
query_indices = set()

for (i, cid, label), (features, _) in zip(meta_test_info, samples):
    query_to_samples[i].append((features, label))
    query_indices.add(i)

query_indices = sorted(list(query_indices))
train_qs, test_qs = train_test_split(query_indices, test_size=0.6, random_state=SEED)
train_qs, val_qs = train_test_split(train_qs, test_size=0.1, random_state=SEED)

print(f"Train questions: {len(train_qs)}, Val questions: {len(val_qs)}, Test questions: {len(test_qs)}")

def flatten_sample_set(qs):
    return [sample for q in qs for sample in query_to_samples[q]]

train_data_raw = flatten_sample_set(train_qs)
val_data_raw = flatten_sample_set(val_qs)
test_data_raw = flatten_sample_set(test_qs)

# === Normalize ===
scaler = StandardScaler()
train_feats = scaler.fit_transform([x for x, _ in train_data_raw])
train_labels = [y for _, y in train_data_raw]
train_data = list(zip(train_feats, train_labels))

val_feats = scaler.transform([x for x, _ in val_data_raw])
val_labels = [y for _, y in val_data_raw]
val_data = list(zip(val_feats, val_labels))

test_feats = scaler.transform([x for x, _ in test_data_raw])
test_labels = [y for _, y in test_data_raw]
test_data = list(zip(test_feats, test_labels))

with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# === DataLoaders ===
train_loader = DataLoader(ClusterDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ClusterDataset(val_data), batch_size=BATCH_SIZE)
test_loader = DataLoader(ClusterDataset(test_data), batch_size=BATCH_SIZE)

# === Model Init ===
model = CorpusRoutingNN(input_dim=train_feats[0].shape[0]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()

scheduler_cyclic = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=1e-3, max_lr=5e-3, step_size_up=10,
    mode="triangular2", cycle_momentum=False
)
scheduler_step = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.05)

# === Training Loop with Validation ===
best_val_f1 = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch < 115:
            scheduler_cyclic.step()
        else:
            scheduler_step.step()

        preds = (torch.sigmoid(outputs) > THRESHOLD).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

    acc = correct / total
    print(f"Train Loss: {total_loss:.4f} | Accuracy: {acc:.2%}")

    # === Validation ===
    model.eval()
    val_y_true, val_y_pred = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.sigmoid(outputs)
            preds = (probs > THRESHOLD).float().cpu().numpy().tolist()
            val_y_pred.extend(preds)
            val_y_true.extend(labels.numpy().tolist())

    val_f1 = f1_score(val_y_true, val_y_pred)
    print(f"Validation F1: {val_f1:.4f}")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
        print("Saved best model.")

# === Final Evaluation ===
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        outputs = model(features)
        probs = torch.sigmoid(outputs)
        preds = (probs > THRESHOLD).float().cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.numpy().tolist())

print("\n=== Final Test Results ===")
print(classification_report(y_true, y_pred, digits=4))

## === Save outputs ===
#np.save(os.path.join(OUTPUT_DIR, "train_indices.npy"), train_data)
#np.save(os.path.join(OUTPUT_DIR, "test_indices.npy"), test_data)
#np.save(os.path.join(OUTPUT_DIR, "val_indices.npy"), val_data)
#
#with open(os.path.join(OUTPUT_DIR, "query_splits.json"), "w") as f:
#    json.dump({"train": train_qs, "val": val_qs, "test": test_qs}, f, indent=2)
#
#with open(os.path.join(OUTPUT_DIR, "test_predictions.json"), "w") as f:
#    json.dump({"true": y_true, "pred": y_pred}, f, indent=2)
#
