from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from FlagEmbedding import BGEM3FlagModel
from receipt_to_text import receipt_to_text

client = QdrantClient(host="localhost", port=6333)
model = BGEM3FlagModel('BAAI/bge-m3')

def store_receipt(parsed_json, point_id):
    text = receipt_to_text(parsed_json)
    embedding = model.encode([text])['dense_vecs'][0]

    client.upsert(
        collection_name="receipts",
        points=[
            PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=parsed_json
            )
        ]
    )
    print(f"Stored receipt {point_id}: {text}")


if __name__ == "__main__":
    sample_json = {
        "menu": [
            {"nm": "BBQ Chicken", "cnt": "1", "price": "41,000"}
        ],
        "total": {"total_price": "41,000"}
    }
    store_receipt(sample_json, point_id=2)