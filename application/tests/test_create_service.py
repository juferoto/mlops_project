from io import BytesIO

import requests
from PIL import Image

""" Prueba para validar el despliegue local del servicio luego de realizar nuevos cambios. """
def test_create_service():
    API_URL = "http://127.0.0.1:8000/predict"
    image_path = "plaga.png"

    image = Image.open(image_path)
    img_byte_array = BytesIO()
    image.save(img_byte_array, format="PNG")
    img_byte_array = img_byte_array.getvalue()

    response = requests.post(API_URL, files={"file": img_byte_array})
    prediction = response.json()

    assert prediction[0] in ["0", "1"]
