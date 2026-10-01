import json
from typing import List, Dict, Any

import requests
from qdrant_client import QdrantClient

MODEL_NAME = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/chat"
QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)
EMBED_MODEL = None


def search_receipts(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search Qdrant for relevant receipt data using semantic vector similarity."""
    global EMBED_MODEL
    if EMBED_MODEL is None:
        print("[SEARCH] Loading BGE-M3 embedding model...", flush=True)
        import torch
        from FlagEmbedding import BGEM3FlagModel

        torch.set_num_threads(2)
        EMBED_MODEL = BGEM3FlagModel("BAAI/bge-m3")

    print("[SEARCH] Encoding query...", flush=True)
    vector = EMBED_MODEL.encode([query])['dense_vecs'][0]
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


def call_ollama(messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None, *, json_mode: bool = False):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"num_thread": 4},
    }
    if tools:
        payload["tools"] = tools
    if json_mode:
        payload["format"] = "json"

    print("[OLLAMA] Waiting for model response...", flush=True)
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


def build_tools_schema() -> List[Dict[str, Any]]:
    return [{
        "type": "function",
        "function": {
            "name": "search_receipts",
            "description": "Search stored receipts for an item, total, vendor, or spending question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The textual question or phrase to search the receipt database for."
                    }
                },
                "required": ["query"]
            }
        }
    }]


def decide_and_search(question: str):
    messages = [
        {
            "role": "system",
            "content": "You are a receipt assistant. If the user asks about a receipt or spending, use the search_receipts tool before answering."
        },
        {"role": "user", "content": question}
    ]
    result = call_ollama(messages, tools=build_tools_schema())

    tool_calls = result.get("message", {}).get("tool_calls", [])
    if not tool_calls:
        raise ValueError("Model did not call the receipt search tool.")

    first_call = tool_calls[0]
    arguments = first_call.get("function", {}).get("arguments", {})
    query = arguments.get("query")
    if not query:
        raise ValueError("Tool call did not include a valid query string.")

    print(f"\n[STEP 1] Tool decided to search with query: {query}")
    hits = search_receipts(query, limit=5)
    print(f"[STEP 2] Retrieved {len(hits)} result(s) from Qdrant")
    return query, hits


def evaluate_sufficiency(question: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    data = json.dumps(results, ensure_ascii=False)
    prompt = (
        "You are assessing whether retrieved receipt data is enough to answer the user's question.\n"
        f"Question: {question}\n"
        f"Retrieved data: {data}\n"
        "Return valid JSON only with keys: 'sufficient' (boolean), 'reason' (string)."
    )
    response = call_ollama([{"role": "user", "content": prompt}], json_mode=True)
    content = response.get("message", {}).get("content", "{}")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"sufficient": False, "reason": "Could not parse model evaluation response."}
    print(f"[STEP 3] Sufficiency check: {parsed}")
    return parsed


def reformulate_query(question: str, previous_query: str) -> str:
    prompt = (
        f"The original question is: {question}\n"
        f"The previous search query was: {previous_query}\n"
        "Rewrite a better search query for receipt retrieval. Return only the final query text."
    )
    response = call_ollama([{"role": "user", "content": prompt}])
    return response.get("message", {}).get("content", previous_query).strip()


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
    print(f"\n[STEP 5] Final answer: {answer}")
    return answer


def run_agent(question: str, max_retries: int = 2):
    current_query = question
    for attempt in range(max_retries + 1):
        print(f"\n=== Attempt {attempt + 1} ===")
        query, results = decide_and_search(current_query)
        evaluation = evaluate_sufficiency(question, results)

        if evaluation.get("sufficient") is True:
            return final_answer(question, results)

        if attempt == max_retries:
            print("[STEP 4] Max retries reached; answering with the strongest available data.")
            return final_answer(question, results)

        current_query = reformulate_query(question, query)
        print(f"[STEP 4] Retry with better query: {current_query}")

    return final_answer(question, [])


if __name__ == "__main__":
    example_question = "How much did I spend on pizza?"
    print("Running receipt-agent loop for:", example_question)
    result = run_agent(example_question)
    print("\nResult:")
    print(result)