import gc
import json
from typing import List, Dict, Any

import requests
from qdrant_client import QdrantClient

MODEL_NAME = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/chat"
QDRANT_CLIENT = None


def search_receipts(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search Qdrant for relevant receipt data using semantic vector similarity."""
    global QDRANT_CLIENT
    if QDRANT_CLIENT is None:
        QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)

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


def call_ollama(messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "keep_alive": 0,
        "options": {"num_thread": 2, "num_ctx": 2048},
    }
    if tools:
        payload["tools"] = tools

    print("[OLLAMA] Waiting for model response...", flush=True)
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


RECEIPT_SEARCH_TOOL = [{
    "type": "function",
    "function": {
        "name": "search_receipts",
        "description": (
            "Search the user's saved receipt records for their purchases, items, "
            "merchants, dates, totals, or personal spending. Use this when the "
            "question asks about something the user bought or paid for."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise semantic search query for the receipt records."
                }
            },
            "required": ["query"]
        }
    }
}]


def run_agent(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Answer general questions directly. For questions about the user's "
                "own purchases, receipts, merchants, or spending, call the "
                "search_receipts tool before answering. After receiving tool results, "
                "answer only from those results; do not invent purchase details."
            )
        },
        {"role": "user", "content": question},
    ]
    response = call_ollama(messages, tools=RECEIPT_SEARCH_TOOL)
    tool_calls = response.get("message", {}).get("tool_calls", [])

    if not tool_calls:
        answer = response.get("message", {}).get("content", "I could not generate an answer.")
        print(f"\n[ANSWER] {answer}")
        return answer

    tool_call = tool_calls[0]
    function = tool_call.get("function", {})
    if function.get("name") != "search_receipts":
        raise ValueError(f"Unknown tool requested: {function.get('name')}")

    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    query = arguments.get("query", question).strip() or question
    print(f"[TOOL] Qwen chose receipt search: {query}", flush=True)
    results = search_receipts(query, limit=5)
    print(f"[TOOL] Retrieved {len(results)} result(s)", flush=True)

    messages.append(response["message"])
    messages.append({
        "role": "tool",
        "content": json.dumps(results, ensure_ascii=False),
    })
    final_response = call_ollama(messages)
    answer = final_response.get("message", {}).get("content", "I could not answer from the retrieved receipts.")
    print(f"\n[ANSWER] {answer}")
    return answer


if __name__ == "__main__":
    print("Receipt assistant ready. Type 'exit' to quit.")
    while True:
        question = input("\nAsk your query: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            run_agent(question)