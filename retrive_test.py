from qdrant_client import QdrantClient
from FlagEmbedding import BGEM3FlagModel

client = QdrantClient(host="localhost", port=6333)
model = BGEM3FlagModel('BAAI/bge-m3')

query = "how much did I spend on rice item"
query_vector = model.encode([query])['dense_vecs'][0]

results = client.query_points(
    collection_name="receipts",
    query=query_vector.tolist(),
    limit=3
)

for hit in results.points:
    print(f"Score: {hit.score}")
    print(f"Payload: {hit.payload}")
    print()