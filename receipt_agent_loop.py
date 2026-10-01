import gc
import json
from typing import List, Dict, Any

import requests
from qdrant_client import QdrantClient

MODEL_NAME = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/chat"
QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)


def search_receipts(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search Qdrant for relevant receipt data using semantic vector similarity."""
    print("[SEARCH] Loading BGE-M3 to encode the query...", flush=True)
    import torch
    from FlagEmbedding import BGEM3FlagModel

    torch.set_num_threads(1)
    embed_model = BGEM3FlagModel("BAAI/bge-m3")
    print("[SEARCH] Encoding query...", flush=True)
    try:
        vector = embed_model.encode([query])['dense_vecs'][0]
    finally:
        del embed_model
        gc.collect()

    results = QDRANT_CLIENT.query_points(
        collection_name="receipts",
        query=vector.tolist(),
        limit=limit,
    )

    matches = []
    for hit in results.points:
        matches.append({
            "score": float(hit.score),
            "payload": hit.payload,
        })
    return matches


def call_ollama(messages: List[Dict[str, str]]):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "keep_alive": 0,
        "options": {"num_thread": 2, "num_ctx": 2048},
    }

    print("[OLLAMA] Waiting for model response...", flush=True)
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


def final_answer(question: str, results: List[Dict[str, Any]]) -> str:
    data = json.dumps(results, ensure_ascii=False)
    prompt = (
        "Answer the user's question using ONLY the retrieved receipt data below. "
        "Do not invent numbers or facts. If the data does not support the answer, say so clearly.\n"
        f"Question: {question}\n"
        f"Receipt data: {data}"
    )
    response = call_ollama([{"role": "user", "content": prompt}])
    answer = response.get("message", {}).get("content", "I could not answer from the retrieved data.")
    print(f"\n[ANSWER] {answer}")
    return answer


def run_agent(question: str) -> str:
    print(f"\n[SEARCH] Searching Qdrant for: {question}")
    results = search_receipts(question, limit=5)
    print(f"[SEARCH] Retrieved {len(results)} result(s)")
    if not results:
        return "No receipts were found in Qdrant."
    return final_answer(question, results)


if __name__ == "__main__":
    print("Receipt assistant ready. Type 'exit' to quit.")
    while True:
        question = input("\nAsk about your receipts: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            run_agent(question)