import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

BASE_DIR = "/mnt/sacs/scratch/home/dpetresc/FeB4RAG/dataset_creation/2_search/embeddings"

# === 1. Load dataset and metadata ===
with open(os.path.join(BASE_DIR, "routing_grouped_by_query.pkl"), "rb") as f:
    query_to_data = pickle.load(f)

encoder_dims = {
  "e5-large": 1024,
  "SGPT-5.8B-weightedmean-msmarco-specb-bitfit": 4096,
  "UAE-Large-V1": 1024,
  "all-mpnet-base-v2": 768,
  "multilingual-e5-large": 1024,
  "ember-v1": 1024,
  "e5-base": 768,
  "gte-base": 768
}

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

# === 2. Split queries ===
query_ids = list(query_to_data.keys())
#train_q, temp_q = train_test_split(query_ids, test_size=0.5, random_state=42)
#val_q, test_q = train_test_split(temp_q, test_size=0.9, random_state=42)
train_q, rest_q = train_test_split(query_ids, test_size=0.7, random_state=42)  # 30% train
val_q, test_q = train_test_split(rest_q, test_size=6/7, random_state=42)       # 10% val, 60% test

# === 3. Flatten and unpack ===
def flatten(qids): return [ex for qid in qids for ex in query_to_data[qid]]
def unpack(data):
    X = np.array([x for x, _ in data])
    y = np.array([y for _, y in data])
    return X, y

X_train, y_train = unpack(flatten(train_q))
X_val, y_val = unpack(flatten(val_q))
X_test, y_test = unpack(flatten(test_q))

# === 4. Torch Dataset ===
class RoutingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(RoutingDataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader   = DataLoader(RoutingDataset(X_val, y_val), batch_size=128)
test_loader  = DataLoader(RoutingDataset(X_test, y_test), batch_size=128)

# === 5. Model ===
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CorpusRouter(input_dim=X_train.shape[1]).to(device)

# === 6. Imbalance-aware loss ===
pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=1e-3,
    max_lr=5e-3,
    step_size_up=10,
    mode="triangular2",
    cycle_momentum=False,
)
scheduler_fixed = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.05)

# === 7. Evaluation ===
def evaluate_with_metrics(loader, threshold=0.5):
    print("THRESHOLD ", threshold)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    TP = FP = FN = TN = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            TP += ((preds == 1) & (y == 1)).sum().item()
            TN += ((preds == 0) & (y == 0)).sum().item()
            FP += ((preds == 1) & (y == 0)).sum().item()
            FN += ((preds == 0) & (y == 1)).sum().item()

    acc = accuracy_score(all_labels, all_preds)
    prec, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0

    num_total = len(all_labels)
    num_true_positives = sum(all_labels)
    num_pred_positives = sum(all_preds)
    reduction_pred = 1 - (num_pred_positives / num_total)
    reduction_true = 1 - (num_true_positives / num_total)
    
    print(f"Total test instances: {num_total}")
    print(f"Ground-truth positives: {int(num_true_positives)}, reduction {reduction_true*100}")
    print(f"Predicted positives:   {int(num_pred_positives)}, reduction {reduction_pred*100}")
    print(f"TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}")
    print(f"Acc: {acc:.2%} | Prec: {prec:.2%} | Recall: {recall:.2%} | F1: {f1:.2%} | AUC: {auc:.2f}")
    return acc, prec, recall, f1, auc

# === 8. Training Loop ===
print("\nStarting training...")
best_f1 = 0
best_model_path = "router_best_model.pt"

num_epochs = 150
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch < 115:
            scheduler.step()
        else:
            scheduler_fixed.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch}: Train Loss = {avg_loss:.4f}")
    print("Validation Set Metrics:")
    acc, _, _, f1, _ = evaluate_with_metrics(val_loader)

    # Save best model by F1 score
    if acc > best_f1:
        best_f1 = acc
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with F1 = {f1:.4f}")

model.load_state_dict(torch.load(best_model_path))
print("\nLoaded best model from disk.")
# === 9. Final Evaluation ===
print("\nFinal Test Set Evaluation:")
evaluate_with_metrics(test_loader)