from io import BytesIO
import os

import requests
import streamlit as st
from PIL import Image

# En este metodo agregar todos los entradas que va a tener la aplicacion (texto o imagenes)
def get_inputs():
    st.title("Predecir plaga")

    data = st.file_uploader(
        "Seleccione la imagen a validar", type=["jpg", "jpeg", "png"]
    )
    return data

# En este metodo realizar la peticion en donde esta alojada el API del modelo ML
def write_predictions(data: object):
    if data is not None:
        # Colocar la ruta en donde esta alojada la aplicacion local o remota
        API_URL = f'{os.getenv("PLAGUE_API")}/predict'

        # Cargar imagen a validar (Opcional)
        image = Image.open(data)
        st.image(image, caption="Cargando imagen.", use_column_width=True)

        if st.button("Validar"):

            try:
                # Obtiene la imagen a validar
                img_byte_array = BytesIO()
                image_format = data.type.split("/")[1]
                image.save(img_byte_array, format=image_format)
                img_byte_array = img_byte_array.getvalue()

                # Envia la información a la API
                response = requests.post(API_URL, files={"file": img_byte_array})
                result = response.json()

                # Mostrar el resultado
                if result['prediction'][0] == 0:
                    st.write("Esta imagen no tiene plaga.")
                else:
                    st.write("Esta imagen tiene plaga.")

            except Exception as e:
                st.error(f"Error: {str(e)}")

def main():
    data = get_inputs()
    write_predictions(data)

if __name__ == "__main__":
    main()
