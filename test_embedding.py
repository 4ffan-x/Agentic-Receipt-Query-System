from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3')

text = "Coffee, total 25000"
embedding = model.encode([text])['dense_vecs'][0]

print(embedding.shape)
print(embedding[:5])