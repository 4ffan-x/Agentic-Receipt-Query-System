from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from FlagEmbedding import BGEM3FlagModel
from receipt_to_text import receipt_to_text

processor = DonutProcessor.from_pretrained("./model/donut_cord_final_v2")
donut_model = VisionEncoderDecoderModel.from_pretrained("./model/donut_cord_final_v2")

qdrant_client = QdrantClient(host="localhost", port=6333)
embed_model = BGEM3FlagModel('BAAI/bge-m3')

def extract_receipt(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    outputs = donut_model.generate(
        pixel_values,
        decoder_input_ids=torch.tensor([[donut_model.config.decoder_start_token_id]]),
        max_length=512,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    result = processor.batch_decode(outputs)[0]
    cleaned = result.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    return processor.token2json(cleaned)

def store_receipt(image, point_id):

    """Full pipeline: image -> Donut -> embed -> Qdrant."""

    parsed_json = extract_receipt(image)
    text = receipt_to_text(parsed_json)
    embedding = embed_model.encode([text])['dense_vecs'][0]

    qdrant_client.upsert(
        collection_name="receipts",
        points=[PointStruct(id=point_id, vector=embedding.tolist(), payload=parsed_json)]
    )
    print(f"Stored: {text}")