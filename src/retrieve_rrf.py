"""RRF retrieval (reuse your HW2 retrievers). Don't change main()'s flags;
never read gold title/context/answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch as torch
from rank_bm25 import BM25Okapi
import torch
import gc
from pathlib import Path
from sentence_transformers import SentenceTransformer
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def top_k(scores, k):
    ranked_ids = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    top_ids = ranked_ids[:k]
    top_scores = [scores[i] for i in top_ids]
    return top_ids,top_scores


def ranked_retriever_def(question:str,config:dict, chunk_set, _bm_250,dense_encoder, context_vector,out_path):
    candidate_k = config["rrf"]["candidate_k"]
    question_ids = dense_encoder.encode(question,convert_to_tensor=True, normalize_embeddings=True, device=DEVICE) #(384,)
    sparse_scores = _bm_250.get_scores(question.split())
    sparse_ids, sparse_scores = top_k(sparse_scores,candidate_k)
    dense_scores = (context_vector @ question_ids.T).cpu().numpy()
    dense_ids, dense_scores = top_k(dense_scores,candidate_k)
    rrf_scores, ranked = rrf_rank(config, sparse_ids, dense_ids, config["rrf"]["rrf_k"])
    chunk_ranked = []
    for i in range(len(ranked)):
        chunk_ranked.append({
            "chunk_id": str(ranked[i]),
            "title": chunk_set[ranked[i]]["title"],
            "text": chunk_set[ranked[i]]["text"],
            "score": rrf_scores[ranked[i]]
        })
    return chunk_ranked

def rrf_rank(config, sparse_ids, dense_ids, k=60):
    rrf_scores = {}
    k = config["rrf"]["rrf_k"]
    for rank, chunk_id in enumerate(sparse_ids, start=1):
        chunk_id = int(chunk_id)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1 / (k + rank)

    for rank, chunk_id in enumerate(dense_ids, start=1):
        chunk_id = int(chunk_id)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1 / (k + rank)

    ranked = sorted(
        rrf_scores.items(),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return rrf_scores,[chunk_id for chunk_id, score in ranked]


def chunking(step: int, text: str)-> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i<len(words):
        chunks.append(" ".join(words[i:i+step]))
        if i+step>=len(words):
            break
        i+=step
    return chunks

def build_retriever(articles: list[dict], config: dict):
    """Build the sparse (BM25) and dense indexes ONCE and return them in any
    object you like (class instance, dict, tuple). run() calls this once,
    before the question loop; it is the only place indexes are built.

    Chunk the articles into ~config["chunk_words"] words each. Give every
    chunk a stable id, e.g. f"{pageid}:{chunk_index}", and keep its article
    title. The dense encoder is config["rrf"]["dense_model_name"].
    """
    chunk_set = []
    # for article in corpus:
    for article in articles:
        for chunk in chunking(int(config["chunk_words"]),article["text"]):
            chunk_set.append({
                "title": article['title'],
                "text": chunk,
            })
        corpus_chunk = [chunk["text"] for chunk in chunk_set]
    dense_encoder = SentenceTransformer(config["rrf"]["dense_model_name"], device=str(DEVICE))
    context_vector = dense_encoder.encode(corpus_chunk,convert_to_tensor=True, normalize_embeddings=True,device=DEVICE) #(N,384)
    tokenized_chunks = [chunks.split() for chunks in corpus_chunk]
    _bm_250 = BM25Okapi(tokenized_chunks)
    return chunk_set, _bm_250, dense_encoder, context_vector


def rrf_retrieve(question: str, retriever, config: dict) -> list[dict]:
    """Return chunks best first as {"chunk_id", "title", "text", "score"}, using the
    retriever from build_retriever. Never build or load indexes here: this
    runs once per question (4,681 times).

    Take each retriever's top config["rrf"]["candidate_k"] chunks and fuse
    them with RRF(d) = sum over retrievers of 1 / (rrf_k + rank(d)), where
    rank is 1-based (the best chunk has rank 1). Fuse on chunk_id, never on
    title or text: an article's chunks share a title.
    """
    chunk_set, _bm_250, dense_encoder, context_vector = retriever
    return ranked_retriever_def(question,config,chunk_set,_bm_250, dense_encoder, context_vector, Path("runs/rrf.jsonl"))
    


def run(questions_path: Path, articles_path: Path, config_path: Path, out_path: Path) -> None:
    """Write runs/rrf.jsonl: per question, the top pool_k chunks as
    {"id", "question", "contexts": [{"chunk_id", "title", "text"}, ...]}."""
    questions = load_jsonl(questions_path)
    articles = load_jsonl(articles_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pool_k = config["rrf"]["pool_k"]
    retriever = build_retriever(articles, config)

    with out_path.open("w", encoding="utf-8") as handle:
        for row in questions:
            ranked = rrf_retrieve(row["question"], retriever, config)
            contexts = [
                {key: item[key] for key in ("chunk_id", "title", "text") if key in item} for item in ranked[:pool_k]
            ]
            handle.write(json.dumps({"id": row["id"], "question": row["question"], "contexts": contexts}) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RRF retrieval and write runs/rrf.jsonl.")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--articles", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    run(args.questions, args.articles, args.config, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
