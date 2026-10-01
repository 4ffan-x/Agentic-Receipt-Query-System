from PIL import Image
from pipeline import store_receipt

image = Image.open("assets/receipt_example.jpg")
store_receipt(image, point_id=6)
# docker run -p 6333:6333 -p 6334:6334 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant