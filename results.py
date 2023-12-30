import argparse

import joblib

from pipeline import create_pipeline
from train_pipeline import get_metrics, run_training

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Ejecutar el programa con parámetros.")
#     parser.add_argument("--input_dir", type=str, help="Directorio de entrada de imágenes.")
#     parser.add_argument("--categories", type=str, help="Categorías separadas por coma.")
#     parser.add_argument("--num_evaluaciones", type=int, help="Número de evaluaciones.")
#     parser.add_argument("--tamano_sampling", type=int, help="Tamaño de muestreo.")
#     parser.add_argument("--tamano_pruebas", type=float, help="Tamaño de pruebas.")

#     args = parser.parse_args()
#     categories = args.categories.split(",") if args.categories else []
#     show_results(args.input_dir, categories, args.num_evaluaciones, args.tamano_sampling, args.tamano_pruebas)


# def show_results(input_dir="D:\proyectoGrado\DatosAguacate", categories=['sin_plaga', 'plaga'], num_evaluaciones=10, tamano_sampling=50, tamano_pruebas=0.2):
def show_results():
    input_dir = "D:\proyectoGrado\DatosAguacate"
    categories = ["sin_plaga", "plaga"]
    num_evaluaciones = 10
    tamano_sampling = 50
    tamano_pruebas = 0.2

    # Crea el pipeline
    pipeline = create_pipeline(input_dir, categories)
    matrices_confusion = run_training(
        pipeline, num_evaluaciones, tamano_sampling, tamano_pruebas
    )

    joblib.dump(pipeline, "model_regression.joblib")

    # Calcula las metricas apartir de todas las matrices de confusión
    results = get_metrics(matrices_confusion)

    # Calcular y mostrar las métricas finales
    print(f'Accuracy Final: {results["accuracy"] * 100}')
    print(f'Precision Final: {results["precision"] * 100}')
    print(f'Recall Final: {results["recall"] * 100}')


if __name__ == "__main__":
    show_results()
