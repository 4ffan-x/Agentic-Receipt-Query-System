from PIL import Image
from pipeline import store_receipt

image = Image.open("assets/receipt_example.jpg")
store_receipt(image, point_id=5)