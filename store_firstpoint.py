from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from FlagEmbedding import BGEM3FlagModel

client = QdrantClient(host="localhost", port=6333)
model = BGEM3FlagModel('BAAI/bge-m3')

receipt_text = "REAL GANACHE, EGG TART, PIZZA TOAST, total 45500"
embedding = model.encode([receipt_text])['dense_vecs'][0]

client.upsert(
    collection_name="receipts",
    points=[
        PointStruct(
            id=1,
            vector=embedding.tolist(),
            payload={
                "items": ["REAL GANACHE", "EGG TART", "PIZZA TOAST"],
                "total": "45,500",
                "vendor": "example_vendor"
            }
        )
    ]
)

print("Stored successfully")
print(client.get_collection("receipts"))