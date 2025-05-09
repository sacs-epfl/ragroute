import numpy as np

def rerank(docs, scores, k):
    # Just rerank based on scores for the moment
    sorted_indices = np.argsort(scores)[::-1]  # Sort scores descending
    merged_docs = [docs[i] for i in sorted_indices][:k]
    merged_scores = [scores[i] for i in sorted_indices][:k]

    return merged_docs, merged_scores
