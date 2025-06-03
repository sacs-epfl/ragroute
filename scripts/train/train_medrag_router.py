import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import random
from sklearn.metrics import roc_auc_score
from torch.utils.data import random_split
import argparse

# Parse command line arguments to track router experiments
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_id", type=int, default=0, help="ID of the experiment (e.g., 0, 1, 2...)")
args = parser.parse_args()


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED=42
set_seed(SEED)

# **Paths and Configuration**
BASE_DIR = "/mnt/nfs/home/dpetresc/MedRAG/retrieval_cache/"
ROUTING_DIR = "/mnt/nfs/home/dpetresc/MedRAG/routing/"
RELEVANT_DIR = "./relevant/"
EXPERIMENT_ID = args.experiment_id
EXPERIMENT_DIR = os.path.join(ROUTING_DIR, f"experiment_{EXPERIMENT_ID}")
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

CORPORA = ["pubmed", "statpearls", "textbooks", "wikipedia"]
TRAIN_TEST_SPLIT_RATIO = 0.4

source_to_id = {src: i for i, src in enumerate(CORPORA)}
num_sources = len(source_to_id)

# **Define Device**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# **Dataset Class**
class RoutingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label = self.data[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

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

# **Load Corpus Statistics**
def load_corpus_stats():
    corpus_stats = {}
    for corpus in CORPORA:
        stats_file = os.path.join(ROUTING_DIR, f"{corpus}_stats.json")
        if not os.path.exists(stats_file):
            print(f"Skipping corpus {corpus}: Stats file not found.")
            continue
        with open(stats_file, "r") as f:
            corpus_stats[corpus] = json.load(f)
    return corpus_stats

# **Load Data and Prepare Features**
def load_data():
    print("LOAD DATA")
    corpus_stats = load_corpus_stats()
    query_to_data = {}
    benchmark_to_questions = {}

    for benchmark in os.listdir(BASE_DIR):
        benchmark_path = os.path.join(BASE_DIR, benchmark)
        emb_queries_path = os.path.join(benchmark_path, "emb_queries")
        relevant_file = os.path.join(RELEVANT_DIR, f"{benchmark}_relevant_top_32.json")

        if not os.path.exists(emb_queries_path) or not os.path.exists(relevant_file):
            print(f"Skipping {benchmark}: Missing embeddings or relevance data.")
            continue

        with open(relevant_file, "r") as f:
            relevant_corpora = json.load(f)

        for emb_file in tqdm(os.listdir(emb_queries_path), desc=f"Processing {benchmark}"):
            if not emb_file.endswith(".npy"):
                continue

            question_id = emb_file.replace(".npy", "")
            emb_path = os.path.join(emb_queries_path, emb_file)
            query_embedding = np.load(emb_path).flatten()

            if question_id not in relevant_corpora:
                print("QUESTION ID MISSING")
                continue

            query_data = []
            for corpus in CORPORA:
                if corpus not in corpus_stats:
                    print("CORPUS MISSING")
                    continue

                centroid = np.array(corpus_stats[corpus]["centroid"], dtype=np.float32)
                num_documents = corpus_stats[corpus]["num_documents"]
                density = corpus_stats[corpus]["density"]
                centroid_similarity = np.dot(query_embedding, centroid)
                source_id = source_to_id[corpus]
                source_id_vec = np.eye(num_sources)[source_id]  # one-hot
                
                # To experiment
                #features = np.concatenate([query_embedding, centroid, [centroid_similarity, num_documents, density]])
                #features = np.array([centroid_similarity], dtype=np.float32)  # Ensure it's a NumPy array
                features = np.concatenate([query_embedding, centroid])
                #features = np.concatenate([query_embedding, centroid, [num_documents, density]])
                label = 1 if corpus in relevant_corpora[question_id] else 0
                query_data.append((features, label))

            query_to_data[question_id] = query_data
            benchmark_to_questions.setdefault(benchmark, []).append(question_id)

    return query_to_data, benchmark_to_questions

from sklearn.metrics import roc_curve

def find_optimal_threshold(model, val_loader):
    """
    Finds the optimal threshold for classification based on validation data.
    The goal is to minimize False Negatives while keeping False Positives low.
    """
    model.eval()
    all_labels, all_outputs = [], []

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            output = model(features).squeeze()

            # Convert logits to probabilities for thresholding
            probabilities = torch.sigmoid(output)

            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(probabilities.cpu().numpy())  # Use probabilities, not logits

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)

    # Find best threshold: maximize TPR - FPR (true positives - false positives)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print(f"Optimal threshold found: {optimal_threshold:.4f}")
    return optimal_threshold  # This will now be in probability space


def evaluate_model_with_metrics(model, loader, threshold=0.5):
    """
    Evaluate the trained model and compute accuracy, precision, recall, F1-score, and AUC.
    """
    model.eval()
    correct, total = 0, 0
    true_positives, false_positives = 0, 0
    false_negatives, true_negatives = 0, 0
    all_labels, all_outputs = [], []

    with torch.no_grad():
        for features, label in loader:
            features, label = features.to(device), label.to(device)
            output = model(features).squeeze()

            # Apply sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(output)

            predictions = (probabilities > threshold).float()

            all_labels.extend(label.cpu().numpy())
            all_outputs.extend(probabilities.cpu().numpy())  # Save probabilities for AUC

            # Compute confusion matrix values
            true_positives += ((predictions == 1) & (label == 1)).sum().item()
            false_positives += ((predictions == 1) & (label == 0)).sum().item()
            false_negatives += ((predictions == 0) & (label == 1)).sum().item()
            true_negatives += ((predictions == 0) & (label == 0)).sum().item()

            correct += (predictions == label).sum().item()
            total += label.size(0)

    # Compute metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Compute AUC-ROC using probabilities
    auc_score = roc_auc_score(all_labels, all_outputs) if len(set(all_labels)) > 1 else 0

    # Print values as percentage
    print()
    print(f"TOTAL: {total}")
    print(f"TP: {true_positives} ({(true_positives / total) * 100:.2f}%)")
    print(f"TN: {true_negatives} ({(true_negatives / total) * 100:.2f}%)")
    print(f"FP: {false_positives} ({(false_positives / total) * 100:.2f}%)")
    print(f"FN: {false_negatives} ({(false_negatives / total) * 100:.2f}%)")

    return accuracy, precision, recall, f1_score, auc_score, true_positives, true_negatives, false_positives, false_negatives

import pickle

def save_preprocessed_data(train_data, val_data, test_datasets, scaler, val_qs):
    """ Save train/val/test data and scaler to disk to avoid recomputation. """
    save_path = os.path.join(EXPERIMENT_DIR, "preprocessed_data.pkl")
    with open(save_path, "wb") as f:
        pickle.dump((train_data, val_data, test_datasets, scaler, val_qs), f)
    print(f"Preprocessed data saved to {save_path}")


def load_preprocessed_data():
    """ Load saved preprocessed train/val/test data if available. """
    load_path = os.path.join(EXPERIMENT_DIR, "preprocessed_data.pkl")
    if os.path.exists(load_path):
        with open(load_path, "rb") as f:
            train_data, val_data, test_datasets, scaler, val_qs = pickle.load(f)
        print(f"Loaded preprocessed data from {load_path}")
        return train_data, val_data, test_datasets, scaler, val_qs
    return None, None, None, None, None

def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for features, label in loader:
            features, label = features.to(device), label.to(device)
            output = model(features).squeeze()

            # Compute loss using raw logits (BCEWithLogitsLoss handles sigmoid internally)
            loss = criterion(output, label)
            total_loss += loss.item()

            # Apply sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(output)

            # Use probability threshold (default: 0.5)
            predictions = (probabilities > 0.5).float()

            correct += (predictions == label).sum().item()
            total += label.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy

def train_and_evaluate():
    # Load saved data if available
    train_data, val_data, test_datasets, scaler, val_qs = load_preprocessed_data()

    # For feature experiment
    train_data = None

    if train_data is None or test_datasets is None:
        query_to_data, benchmark_to_questions = load_data()

        # Load saved train-test split - same for all experiments
        split_file = os.path.join(ROUTING_DIR, "train_test_split_per_benchmark.json")
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                benchmark_splits = json.load(f)
            print(f"Loaded train-test split from {split_file}")
        else:
            print("Train-test split file missing. Generating a new split...")
            benchmark_splits = {}

            for benchmark, question_ids in benchmark_to_questions.items():
                if len(question_ids) < 10:
                    print(f"Skipping {benchmark}: Not enough questions for split.")
                    continue

                train_qs, test_qs = train_test_split(
                    question_ids, test_size= 1 - TRAIN_TEST_SPLIT_RATIO, random_state=SEED
                )
                benchmark_splits[benchmark] = {"train": train_qs, "test": test_qs}

            with open(split_file, "w") as f:
                json.dump(benchmark_splits, f, indent=4)
            print(f"Train-test split saved to {split_file}")

        # Extract training and validation questions
        train_questions = []
        test_questions = {}

        for benchmark, split in benchmark_splits.items():
            train_questions.extend(split["train"])
            test_questions[benchmark] = split["test"]

        # Create train/val datasets
        if val_qs is None:
            VALIDATION_SPLIT = 0.1
            train_qs, val_qs = train_test_split(train_questions, test_size=VALIDATION_SPLIT, random_state=SEED)
        else:
            train_qs = [q for q in train_questions if q not in val_qs]

        print(f"Total Train Questions: {len(train_questions)} | Train: {len(train_qs)}, Validation: {len(val_qs)}")

        train_data = [sample for q_id in train_qs if q_id in query_to_data for sample in query_to_data[q_id]]
        val_data = [sample for q_id in val_qs if q_id in query_to_data for sample in query_to_data[q_id]]

        print(f"Training Samples: {len(train_data)}, Validation Samples: {len(val_data)}")

        # Generate Test Datasets
        test_datasets = {
            benchmark: [
                sample for question in test_qs if question in query_to_data for sample in query_to_data[question]
            ]
            for benchmark, test_qs in test_questions.items()
        }

        # Scale features or not
        scaler = StandardScaler()
        train_features = scaler.fit_transform([features for features, _ in train_data])
#        train_features = [features for features, _ in train_data]
        train_labels = [label for _, label in train_data]
        train_data = list(zip(train_features, train_labels))

        val_features = scaler.transform([features for features, _ in val_data])
#        val_features = [features for features, _ in val_data]
        val_labels = [label for _, label in val_data]
        val_data = list(zip(val_features, val_labels))

        test_datasets = {
            benchmark: list(zip(
                scaler.transform([features for features, _ in data]),
                [label for _, label in data]
            ))
            for benchmark, data in test_datasets.items()
        }
#        test_datasets = {
#	    benchmark: list(zip(
#		[features for features, _ in data],
#		[label for _, label in data]
#	    ))
#	    for benchmark, data in test_datasets.items()
#	}


        # Save processed data for future runs
        save_preprocessed_data(train_data, val_data, test_datasets, scaler, val_qs)

    # Ensure `test_datasets` is properly initialized
    if test_datasets is None or len(test_datasets) == 0:
        raise ValueError("Test datasets are empty! Check `query_to_data` and `benchmark_splits`.")

    print("Train, Validation, and Test data loaded successfully.")

    train_loader = DataLoader(RoutingDataset(train_data), batch_size=128, shuffle=True)
    val_loader = DataLoader(RoutingDataset(val_data), batch_size=128, shuffle=False)

    
    input_dim = train_loader.dataset[0][0].shape[0]  # Get input dimension dynamically
    print("INPUT DIM ", input_dim)
    model = CorpusRoutingNN(input_dim).to(device)

    # Define loss function & optimizer
    fn_weight = 1
    pos_weight = torch.tensor([len(train_data) / sum([label for _, label in train_data])* fn_weight]).to(device)

    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    num_epochs = 150

    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-3,   # Minimum learning rate
        max_lr=5e-3,    # Maximum learning rate
        step_size_up=10,  # Number of batches before reaching max_lr
        mode="triangular2",  # Learning rate follows a triangular pattern
        cycle_momentum=False  # Adam does not use momentum
    )

    scheduler_fixed = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.05)
    
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_val_loss = float("inf")
    best_val_acc = 0
    best_model_state = None
    MODEL_PATH = os.path.join(EXPERIMENT_DIR, "best_model.pth")

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0
        tp, fn, fp, tn = 0, 0, 0, 0
        for features, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            features, label = features.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(features).squeeze()
            loss = criterion(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if epoch < 115:  
                scheduler.step()  # Use CyclicLR for first 50 epochs
            else:
                scheduler_fixed.step()  # Then switch to StepLR for stability

            total_loss += loss.item()

            # Track training accuracy
            predictions = (output > 0.5).float()
            correct += (predictions == label).sum().item()
            total += label.size(0)
            tp += ((predictions == 1) & (label == 1)).sum().item()
            fn += ((predictions == 0) & (label == 1)).sum().item()
            fp += ((predictions == 1) & (label == 0)).sum().item()
            tn += ((predictions == 0) & (label == 0)).sum().item()

        avg_loss = total_loss / len(train_loader)
        #scheduler.step(avg_loss)
        train_accuracy = correct / total if total > 0 else 0
        
        # Evaluate on Validation Set
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)

        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2%}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, MODEL_PATH)
            print(f"Best model saved with Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

    # Ensure we use the best saved model
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    print(f"\n Best model loaded from {MODEL_PATH} for final evaluation.")

    # Find optimal threshold using validation set
    optimal_threshold = find_optimal_threshold(model, val_loader)

    # Evaluate on test sets with optimal threshold
    for benchmark, test_data in test_datasets.items():
        test_loader = DataLoader(RoutingDataset(test_data), batch_size=128, shuffle=False)
        accuracy, precision, recall, f1, auc, tp, tn, fp, fn = evaluate_model_with_metrics(model, test_loader, threshold=optimal_threshold)

        print(f"\n**Test Results for {benchmark}**")
        print(f"Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}, F1-Score: {f1:.2%}, AUC: {auc:.2%}")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    model.eval()
    results = {}
    with torch.no_grad():
        for benchmark, question_ids in test_questions.items():
            for question_id in tqdm(question_ids, desc=f"Predicting for {benchmark}"):
                if question_id not in query_to_data:
                    continue
                predicted_relevant = []
                for idx, (features, _) in enumerate(query_to_data[question_id]):
                    features_tensor = torch.tensor(scaler.transform([features]), dtype=torch.float32).to(device)
                    output = model(features_tensor).squeeze().item()
                    prob = torch.sigmoid(torch.tensor(output)).item()
                    if prob > optimal_threshold:
                        predicted_relevant.append(CORPORA[idx])

                results[question_id] = predicted_relevant

    # Save predictions for this benchmark
    output_path = os.path.join(EXPERIMENT_DIR, f"question_predictions.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    train_and_evaluate()
