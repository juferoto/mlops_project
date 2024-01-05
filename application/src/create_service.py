import io
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import uvicorn
import mlflow

""" Metodo para cargar un modelo desde MLFlow despues de haber sido evaluado por los cientificos """
def load_model():
    # Se agrega la direccion en donde esta alojado MLFlow sea remoto o local
    mlflow.set_tracking_uri("https://dagshub.com/juferoto/mlops_project.mlflow")
    with mlflow.start_run() as run:
        model_name = "model_regression"
        stage = "Production"
        model_uri=f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)
    return model

app = FastAPI()


""" Se agregan los metodos HTTP para agregar al servicio """
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Cargar contenido requerido
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')

        # Preprocesar datos
        dimensions = (640, 640)
        scaled_image = image.resize(dimensions)
        data = np.asarray(scaled_image)
        data = data.reshape((1, 640*640*3))

        # Cargar el modelo
        model = load_model()

        # Realizar la predicción con el modelo
        prediction = model.predict(data)

        # Devolver el resultado como un array de NumPy en formato JSON
        result = {"prediction": prediction.tolist()}
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)