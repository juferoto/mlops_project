from io import BytesIO
import os

import requests
from PIL import Image

""" Prueba para validar el despliegue local del servicio luego de realizar nuevos cambios. """
def set_output(name, value):
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'{name}={value}', file=fh)

def test_create_service():
    API_URL = "http://127.0.0.1:8000/predict"
    image_path = os.path.join(os.path.dirname(__file__), "plaga.png")

    image = Image.open(image_path)
    img_byte_array = BytesIO()
    image.save(img_byte_array, format="png")
    img_byte_array = img_byte_array.getvalue()

    response = requests.post(API_URL, files={"file": img_byte_array})
    result = response.json()
    condition = result['prediction'][0] in [0, 1]

    try:
        assert condition
    finally:
        env_file = os.getenv('GITHUB_OUTPUT')
        if env_file is not None:
            # Escribe las variables a exportar
            set_output("RESULT_APP", condition)