# Predicción de plagas con regresión logística

## Razón
En un equipo de científico de datos, es común que se intente continuamente encontrar el mejor modelo existente en producción, pero para lograrlo hay que verificar que el servicio no falle cuando el nuevo modelo sea desplegado.

Este proyecto demuestra el uso de DagsHub, MLFlow y GitHub Actions para:

- Realizar pruebas automáticas cuando un miembro del equipo realice un pull request.
- Mezclar(merge) un pull request cuando todas las pruebas han pasado.
- Desplegar el modelo ML a un servicio existente.

![](https://i.ibb.co/jTspJCk/flujo-Ml-Ops.png)

Este es el resumen del flujo de trabajo:

## Experimentación con DagsHub y MLFlow
Después de probar diferentes parámetros usando [DagsHub](https://towardsdatascience.com/dagshub-a-github-supplement-for-data-scientists-and-ml-engineers-9ecaf49cc505) y [MLFlow](https://mlflow.org/docs/latest/python_api/index.html), se seleccionan una combinación de parámetros que nos dé un mejor desempeño que el modelo existente en producción, y se envía el codigo por Git.

![](https://i.ibb.co/6JZ59M8/experimentos.png)

## Usar GitHub Actions para probar el modelo y aplicación localmente
El primer flujo de trabajo [validate_model.yaml](https://dagshub.com/juferoto/mlops_project/src/master/.github/workflows/validate_model.yaml) automáticamente prueba un nuevo pull request, el cual únicamente puede ser mezclado (merged) cuando todas las pruebas hayan pasado. En caso de falla se envía una notificación vía correo electrónico al científico de datos designado para el proyecto para valide porque el modelo no fue desplegado correctamente.

![](https://i.ibb.co/gD7qmvz/flujo-Prueba-Codigo.png)

## Usar GitHub Actions que lanza tarea automática para probar el modelo y aplicación
El segundo flujo de trabajo [validate_model_automatically.yaml](https://dagshub.com/juferoto/mlops_project/src/master/.github/workflows/validate_model_automatically.yml) cada día automáticamente a las 4 a.m. realiza una validación al modelo y la aplicación. En caso de falla se envía una notificación vía correo electrónico al científico de datos designado para el proyecto para valide porque el modelo no fue desplegado correctamente.

![](https://i.ibb.co/hgK6wm8/flujo-Validacion-Automatica.png)

## Usar GitHub Actions para desplegar el modelo después de mezclar
El tercer flujo de trabajo [validate_deploy_app.yaml](https://dagshub.com/juferoto/mlops_project/src/master/.github/workflows/validate_deploy_app.yaml) automáticamente despliega un nuevo modelo al servicio existente después de que el pull request es mezclado o se realiza un push a la rama principal (master).

![](https://i.ibb.co/QfMDbLr/flujo-Despliega-App.png)

