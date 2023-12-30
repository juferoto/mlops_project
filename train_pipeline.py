from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def run_training(pipeline, num_evaluaciones, tamano_sampling, tamano_pruebas):
    matrices_confusion = []

    data_sin_plagas, labels_sin_plagas, data_plagas, labels_plagas = pipeline.named_steps['normalize'].image_normalize()

    # Repetir el proceso de entrenamiento y evaluación n veces (10 por defecto)
    for _ in range(num_evaluaciones):

        # Realizar subsampling en la clase mayoritaria
        indices_resample = np.random.choice(len(data_plagas), size=tamano_sampling, replace=True)
        
        data_plagas_resample = np.asarray(data_plagas)[indices_resample]
        labels_plagas_resample = np.asarray(labels_plagas)[indices_resample]

        # Combinar datos de 'sin_plaga' y datos resampleados de 'plaga'
        data_resample = np.concatenate([data_sin_plagas, data_plagas_resample])
        labels_resample = np.concatenate([labels_sin_plagas, labels_plagas_resample])

        # Dividir los datos en conjuntos de entrenamiento y prueba
        x_train, x_test, y_train, y_test = train_test_split(
            data_resample,
            labels_resample,
            test_size=tamano_pruebas,
            shuffle=True,
            random_state=42
        )

        # Entrenar el modelo y obtener las métricas
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        # Calcular métricas adicionales
        matrices_confusion.append(confusion_matrix(y_test, y_pred))
    return matrices_confusion

def get_metrics(matrices_confusion):
    matrices_confusion_promedio = np.mean(matrices_confusion, axis=0)

    # disp = ConfusionMatrixDisplay(confusion_matrix=matrices_confusion_promedio, display_labels=['sin_plaga', 'plaga'])
    # disp.plot(cmap='Blues', values_format='.2f')
    # plt.title('Matriz de Confusión Promedio')
    # plt.show()

    recall = matrices_confusion_promedio[1, 1] / (matrices_confusion_promedio[1, 1] + matrices_confusion_promedio[1, 0])
    accuracy = (matrices_confusion_promedio[0, 0] + matrices_confusion_promedio[1, 1]) / np.sum(matrices_confusion_promedio)
    precision = matrices_confusion_promedio[1, 1] / (matrices_confusion_promedio[1, 1] + matrices_confusion_promedio[0, 1])

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }