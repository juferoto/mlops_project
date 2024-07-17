import io
import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from hydra import compose, initialize
import numpy as np
from rembg import remove
import mlflow
import uvicorn

app = FastAPI()

""" Metodo para cargar un modelo en Produccion desde MLFlow despues de haber sido evaluado por los cientificos """
with initialize(version_base=None, config_path="."):
    config = compose(config_name="main")
    # Se agrega la direccion en donde esta alojado MLFlow sea remoto o local
    mlflow.set_tracking_uri(config.mlflow.tracking_ui)
    os.environ['MLFLOW_TRACKING_USERNAME'] = config.mlflow.username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = config.mlflow.password

    # Obtiene el modelo que fue asignado al entorno de Produccion
    stage = "Production"
    model_uri=f"models:/{config.model.name}/{stage}"
    MODEL = mlflow.pyfunc.load_model(model_uri)

""" Se agregan los metodos HTTP para agregar al servicio """
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Cargar contenido requerido
        content = await file.read()

        # Remover fondo
        output_image = remove(io.BytesIO(content)).convert('RGB')

        # Preprocesar datos
        dimensions = (640, 640)
        scaled_image = output_image.resize(dimensions)
        data = np.asarray(scaled_image)
        data = data.reshape((1, 640*640*3))

        # Realizar la predicción con el modelo
        prediction = MODEL.predict(data)

        # Devolver el resultado como un array de NumPy en formato JSON
        result = {"prediction": prediction.tolist()}
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)