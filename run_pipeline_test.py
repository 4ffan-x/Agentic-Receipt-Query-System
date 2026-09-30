from PIL import Image
from pipeline import store_receipt

image = Image.open("some_receipt.jpg")
store_receipt(image, point_id=100)