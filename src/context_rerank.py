"""Context re-rankers.

Input: the RRF pool (pool_k chunks of {"title", "text"}). Return all of them,
reordered best first; pipeline.py keeps the top top_k_contexts. Extra keys
(e.g. "score") are fine. Load each model once, not per question.
"""

from __future__ import annotations
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
from functools import cache

@cache
def get_bi_encoder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=str(DEVICE))
@cache
def get_cross_encoder(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name, device=str(DEVICE))

#ranked = bi_encoder_rerank(question, pool, config["bi_encoder"]["model_name"])
def bi_encoder_rerank(question: str, contexts: list[dict], model_name: str) -> list[dict]:
    """Sort by cosine similarity of question and chunk embeddings."""
    model = get_bi_encoder(model_name)
    question_embedding = model.encode(question, convert_to_tensor=True, normalize_embeddings=True, device=DEVICE) #(D,)
    context_embeddings = model.encode([chunk['text'] for chunk in contexts], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)#(N,D)
    scores = (context_embeddings @ question_embedding.T).cpu().numpy() #(N,)
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    return [dict(contexts[i],score = float(score)) for i, score in ranked]


def cross_encoder_rerank(question: str, contexts: list[dict], model_name: str) -> list[dict]:
    """Sort by cross-encoder relevance score of each (question, chunk) pair."""
    model = get_cross_encoder(model_name)
    pairs = [(question, chunk['text']) for chunk in contexts]
    scores = model.predict(pairs) #(N,)
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    return [dict(contexts[i],score = float(score)) for i, score in ranked]
    # raise NotImplementedError
