# Machine Learning Challenge

## Descripción del Proyecto

Este proyecto abarca la creación de un modelo para pronosticar la demanda de Cementos Argos y un modelo de clasificación para predecir las clases 'Alpha' o 'Betha' basándose en un conjunto de características dado. El proyecto incluye tres componentes principales:

1. **Pronóstico de Demanda**: Implementación en un Jupyter notebook para pronosticar la demanda y generar un archivo con los datos y pronósticos, junto con un gráfico que muestre el rendimiento del modelo.
2. **Modelo de Clasificación**: Implementación en un Jupyter notebook para entrenar un modelo de clasificación, generar un archivo con el modelo entrenado y un archivo de texto con las métricas.
3. **API de Clasificación**: Implementación de una API usando FastAPI para recibir datos en formato JSON, realizar la clasificación y devolver los resultados. Esto usando el modelo de Clasificación anterior.

Además, se incluye un punto opcional para contenedores Docker y otros entregables.

## Estructura del Repositorio

```
/project_root
│
├── /challenge
│ │
│ ├── /output
│ │ ├── clasificator_model.pkl
│ │ ├── output_clasificator.txt
│ │ ├── output_dataset_demand_acumulate.csv
│ │
│ ├── /encoders
│ │ ├── class_label_encoder.pkl 
│ │ ├── features_scaler.pkl
│ │ ├── feature1_label_encoder.pkl
│ │ ├── feature2_label_encoder.pkl
│ │
│ ├── 1_demand_forecasting.ipynb
│ ├── 2_alpha_betha_classificator.ipynb
│ ├── api.py
│ ├── model.py
│
├── /data
│ ├── to_predict.csv
│ ├── dataset_demand_acumulate.csv 
│ ├── dataset_alpha_betha.csv
│
├── Dockerfile
├── docker-compose.yml
├── README.md
├── requirements.txt
└── teoria_.pdf
```


## Instrucciones

### 1. Pronóstico de Demanda

1. **Descripción**:
   - Se ha implementado un modelo de pronóstico de demanda utilizando ARIMA y XGBoost.
   - Se entrenaron y evaluaron ambos modelos, y se eligió XGBoost por su mejor rendimiento.

2. **Notebook**:
   - **Ubicación**: `/challenge/1_demand_forecasting.ipynb`
   - **Output**: Archivo `/output/output_dataset_demand_acumulate.csv` con los datos y pronósticos.
   - **Gráfico**: Incluye en el notebook un gráfico de entrenamiento, validación y pronóstico con métricas de evaluación.

### 2. Modelo de Clasificación

1. **Descripción**:
   - Se entrenó un modelo de clasificación usando regresión logística después de evaluar varios modelos.
   - El modelo se entrenó con 80% de los datos para entrenamiento y 20% para pruebas.

2. **Notebook**:
   - **Ubicación**: `/challenge/2_alpha_betha_classificator.ipynb`
   - **Output**: 
     - Modelo entrenado guardado como `clasificator_model.pkl` en `/challenge/output/`.
     - Métricas generadas guardadas en `output_clasificator.txt` en `/challenge/output/`.

### 3. API de Clasificación

1. **Descripción**:
   - Se ha desarrollado una API utilizando FastAPI que consume el modelo de clasificación y responde a solicitudes POST con datos JSON.
   - La API proporciona un endpoint para verificar el estado de salud y otro para realizar predicciones.

2. **Archivos**:
   - **Código de la API**: `api.py` en `/challenge/`.
   - **Requerimientos**: `requirements.txt` en `/project_root`.
   - **Petición JSON Ejemplo**:
     ```json
     {
       "data": [
         {
           "SeniorCity": 0,
           "Partner": "Yes",
           "Dependents": "Yes",
           "Service1": "Yes",
           "Service2": "No",
           "Security": "No",
           "OnlineBackup": "Yes",
           "DeviceProtection": "No",
           "TechSupport": "No",
           "Contract": "Month-to-month",
           "PaperlessBilling": "Yes",
           "PaymentMethod": "Mailed check",
           "Charges": 86.3,
           "Demand": 3266
         }
       ]
     }
     ```
   - **Respuesta JSON Ejemplo**:
     ```json
     {
       "prediction": [
         "Betha"
       ]
     }
     ```

### 4. Completar y Realizar Peticiones

1. **Descripción**:
   - Se completó el archivo `to_predict.csv` con la información de demanda pronosticada.
   - Se realizaron 3 peticiones al servicio API con datos de prueba y se entregó un archivo con los resultados obtenidos.

2. **Archivos**:
   - **Archivo CSV para predicciones**: `to_predict.csv` en `/data/`.
   - **Resultados**: Incluidos en el archivo `/challenge/output/results_to_predict.csv`.

### 5. Docker (Opcional)

1. **Descripción**:
   - Se creó un Dockerfile y un archivo docker-compose para facilitar la ejecución de la aplicación en un contenedor Docker.

2. **Archivos**:
   - **Dockerfile**: Contiene la configuración para construir el contenedor.
   - **docker-compose.yml**: Facilita el manejo del contenedor y sus servicios.

### 6. Manejo del Repositorio

1. **Descripción**:
   - El repositorio de GitHub incluye un manejo adecuado de ramas, commits, y pull requests.
   - Se subió un archivo `teoria_.pdf` con las respuestas teóricas.

2. **Archivos**:
   - **Archivo PDF**: `teoria_.pdf` en la raíz del repositorio.

## Cómo Ejecutar

1. **API de Clasificación**:
   - Navega al directorio `/challenge/`.
   - Instala las dependencias:
     ```bash
     pip install -r requirements.txt
     ```
   - Ejecuta el servidor:
     ```bash
     uvicorn api:app --reload
     ```

2. **Docker** :
   - Construye y ejecuta el contenedor Docker:
     ```bash
     docker-compose up --build
     ```
   - Para verificar que la aplicación está funcionando correctamente, puedes hacer una solicitud GET al endpoint de salud:
      http://localhost:8080/health

   - Para realizar una predicción, envía una solicitud POST al endpoint de predicción con el JSON adecuado::
      http://localhost:8080/predict

