import numpy as np

def rerank_medrag(docs, scores, k):
    # Just rerank based on scores for the moment
    sorted_indices = np.argsort(scores)[::-1]  # Sort scores descending
    merged_docs = [docs[i] for i in sorted_indices][:k]
    merged_scores = [scores[i] for i in sorted_indices][:k]

    return merged_docs, merged_scores


def rerank_feb4rag(ids, docs, query_id, k, relevance_data):
    # Sort at reranking time
    rel_docs_with_scores = relevance_data.get(query_id, [])
    rel_doc_order = [docid for docid, _ in sorted(rel_docs_with_scores, key=lambda x: -int(x[1]))]

    # Rank: if doc is in rel_doc_order, use its index; otherwise, push it to the end
    sort_key = {docid: i for i, docid in enumerate(rel_doc_order)}

    # Sort the input docs by relevance (those not in qrels are pushed to end)
    sorted_data = sorted(zip(ids, docs), key=lambda x: sort_key.get(x[0], float('inf')))

    sorted_ids, sorted_docs = zip(*sorted_data) if sorted_data else ([], [])

    return list(sorted_docs[:k]), list(sorted_ids[:k])
