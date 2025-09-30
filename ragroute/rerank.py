import numpy as np

def _as_text(doc):
    """Normalize doc -> single 'title\\ntext' string."""
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        title = doc.get("title") or doc.get("article_title") or ""
        body  = doc.get("text")  or doc.get("content") or doc.get("body") or ""
        return f"{title}\n{body}".strip()
    if isinstance(doc, (list, tuple)):
        if len(doc) >= 2:
            return f"{doc[0]}\n{doc[1]}".strip()
        if len(doc) == 1:
            return str(doc[0]).strip()
    return str(doc).strip()

def rerank_medrag(docs, scores, k, reranker, q_rerank):
    # Just rerank based on scores for the moment
#    print(docs)
#    print(len(docs))
#    print(len(docs[0]))
    if reranker is None:
        sorted_indices = np.argsort(scores)[::-1]  # Sort scores descending
        merged_docs = [docs[i] for i in sorted_indices][:k]
        merged_scores = [scores[i] for i in sorted_indices][:k]

        return merged_docs, merged_scores
    else:
        n = len(docs)
        if n == 0 or k <= 0:
            return [], []

        pairs = [(q_rerank, _as_text(doc)) for doc in docs]

        ce_scores = []
        for i in range(0, n, 32): # bs = 32
            batch = pairs[i:i+32]
            s = reranker.predict(batch, convert_to_numpy=True).tolist()
            ce_scores.extend(s)

        order = np.argsort(ce_scores)[::-1][:k]
        top_docs = [docs[i] for i in order]
        top_scores = [float(ce_scores[i]) for i in order]
        return top_docs, top_scores


def rerank_feb4rag(ids, docs, query_id, k, relevance_data, reranker, q_rerank):
    if reranker is None:
        # Sort at reranking time
        rel_docs_with_scores = relevance_data.get(query_id, [])
        rel_doc_order = [docid for docid, _ in sorted(rel_docs_with_scores, key=lambda x: -int(x[1]))]

        # Rank: if doc is in rel_doc_order, use its index; otherwise, push it to the end
        sort_key = {docid: i for i, docid in enumerate(rel_doc_order)}

        # Sort the input docs by relevance (those not in qrels are pushed to end)
        sorted_data = sorted(zip(ids, docs), key=lambda x: sort_key.get(x[0], float('inf')))

        sorted_ids, sorted_docs = zip(*sorted_data) if sorted_data else ([], [])

        return list(sorted_docs[:k]), list(sorted_ids[:k])
    else:
        n = len(docs)
        if n == 0 or k <= 0:
            return [], []

        pairs = [(q_rerank, _as_text(doc))  for doc in docs]

        ce_scores = []
        for i in range(0, n, 32): # bs = 32
            batch = pairs[i:i+32]
            s = reranker.predict(batch, convert_to_numpy=True).tolist()
            ce_scores.extend(s)

        order = np.argsort(ce_scores)[::-1][:k]
        top_docs = [docs[i] for i in order]
        top_scores = [float(ce_scores[i]) for i in order]
        return top_docs, top_scores


def rerank_wikipedia(docs, scores, k, reranker, q_rerank):
#    print(docs)
#    print(len(docs))
#    print(len(docs[0]))
    if reranker is None:
        sorted_indices = np.argsort(scores)[::-1] # TODO check...
        merged_docs = [docs[i] for i in sorted_indices][:k]
        merged_scores = [scores[i] for i in sorted_indices][:k]

        return merged_docs, merged_scores
    else:
        n = len(docs)
        if n == 0 or k <= 0:
            return [], []
        
        pairs = [(q_rerank, _as_text(doc)) for doc in docs]

        ce_scores = []
        for i in range(0, n, 32): # bs = 32
            batch = pairs[i:i+32]
            s = reranker.predict(batch, convert_to_numpy=True).tolist()
            ce_scores.extend(s)

        order = np.argsort(ce_scores)[::-1][:k]
        top_docs = [docs[i] for i in order]
        top_scores = [float(ce_scores[i]) for i in order]
        return top_docs, top_scores
