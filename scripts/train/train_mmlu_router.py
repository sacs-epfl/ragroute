# === top of the file ===
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
import random
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, roc_curve, accuracy_score
)
from sklearn.metrics import precision_recall_curve
from collections import defaultdict
import pickle
import csv

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
DEFAULT_THRESHOLD = 0.5
OUTPUT_DIR = "./cluster_router_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_SUBJECTS = {
"high_school_microeconomics", "international_law", "college_biology", "college_physics", "miscellaneous", "prehistory", "philosophy", "professional_psychology", "high_school_mathematics"
}

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

# === Load Cluster Stats ===
with open(CLUSTER_STATS_FILE) as f:
    cluster_stats = json.load(f)
centroids = [np.array(c["centroid"], dtype=np.float32) for c in cluster_stats]

# === Build Samples ===
samples = []
meta_test_info = []

for i, item in enumerate(tqdm(dataset)):
    if item["subject"] not in TARGET_SUBJECTS:
        continue

    cluster_file = os.path.join(RETRIEVAL_DIR, f"question_{i}_cluster_ids.txt")
    if not os.path.exists(cluster_file):
        continue

    q_id = f"question_{i}"
    question = item["question"]
    options = item["choices"]
    formatted_q = "\n".join([question, " | ".join(options)])

    q_emb = encode_query_dpr(formatted_q)

    with open(cluster_file) as f:
        retrieved_clusters = set(int(line.strip()) for line in f if line.strip().isdigit())

    for cid in range(NUM_CLUSTERS):
        features = np.concatenate([q_emb, centroids[cid], np.eye(NUM_CLUSTERS)[cid]])
        #print(len(features))
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

with open(os.path.join(OUTPUT_DIR, "test_question_ids.json"), "w") as f:
    json.dump(test_qs, f, indent=2)

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

# === Pos weight for imbalance ===
num_pos = sum(train_labels)
num_neg = len(train_labels) - num_pos
pos_weight = torch.tensor([num_neg / (num_pos + 1e-6)], dtype=torch.float32).to(device)
print(f"Using pos_weight = {pos_weight.item():.4f} (Pos: {num_pos}, Neg: {num_neg})")

criterion = nn.BCEWithLogitsLoss(pos_weight=5*pos_weight)
#criterion = nn.BCEWithLogitsLoss()

scheduler_cyclic = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=1e-3, max_lr=5e-3, step_size_up=10,
    mode="triangular2", cycle_momentum=False
)
scheduler_step = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.05)

# === Training Loop ===
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

        outputs = model(features).squeeze()  # (batch_size,)
        loss = criterion(outputs, labels)

        probs = torch.sigmoid(outputs)
        preds = (probs > DEFAULT_THRESHOLD).float()

        labels = labels.squeeze()

        # Now both preds and labels are 1D tensors
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
    acc = correct / total
    print(f"Train Loss: {total_loss:.4f} | Accuracy: {acc}")

    # === Validation ===
    model.eval()
    val_y_true, val_y_pred = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.sigmoid(outputs)
            preds = (probs > DEFAULT_THRESHOLD).float().cpu().numpy().tolist()
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
y_true, y_pred, y_probs = [], [], []

with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        outputs = model(features)
        probs = torch.sigmoid(outputs).squeeze()
        y_probs.extend(probs.cpu().numpy())
        y_true.extend(labels.numpy())

# Apply optimal threshold
y_pred = [1 if p > 0.5 else 0 for p in y_probs]

# === Compute metrics ===
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
auc = roc_auc_score(y_true, y_probs) if len(set(y_true)) > 1 else 0.0
acc = (tp + tn) / (tp + tn + fp + fn)

print("\nFinal Test Results")
print(f"Accuracy: {acc:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1:.2%}")
print(f"AUC: {auc:.4f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
