#SGDClassifier

Este proyecto implementa desde cero un clasificador binario basado en Regresión Logística utilizando Gradiente Descendente Estocástico (SGD). Incluye procesamiento de datos categóricos y numéricos, entrenamiento personalizado y evaluación del modelo.

---

## Estructura del Código

### Variables Globales

- `lr`: Tasa de aprendizaje.
- `n_epochs`: Número de épocas de entrenamiento.
- `threshold`: Umbral para asignar clase 0 o 1.
- `fit_intercept`: Indica si se añade un sesgo (intercepto).
- `theta`: Vector de pesos del modelo.

---

## Funciones Principales

### `SuperSGDClassifier`

Define el clasificador:

- `__init__(lr, n_epochs, threshold, fit_intercept)`: Inicializa los hiperparámetros.
- `_add_intercept(X)`: Añade columna de unos para el sesgo si es necesario.
- `_sigmoid(z)`: Función sigmoide para convertir z en probabilidad.
- `_loss(h, y)`: Calcula la función de pérdida logarítmica.
- `fit(X, y)`: Entrena el modelo usando descenso estocástico.
- `predict_proba(X)`: Retorna las probabilidades predichas.
- `predict(X)`: Retorna clases binarias según el umbral.

---

### Preprocesamiento

- `ColumnTransformer`:
  - Aplica `StandardScaler` a columnas numéricas.
  - Aplica `OneHotEncoder` a columnas categóricas.
- `train_test_split`: Divide el conjunto de datos en entrenamiento y prueba.

---

## Evaluación del Modelo

- `confusion_matrix(y_test, y_pred)`: Muestra la matriz de confusión.
- `classification_report(y_test, y_pred)`: Imprime precisión, recall y F1-score.

---

## Uso

1. Ubicar los datos en un .csv dentro de la carpeta data, en caso de que no esté, crearla (importante que el nombre sea únicamente 'data').
2. Descargar requirements.txt (pip install -r requirements.txt).
3. Ejectuar desde la raiz (python main.py).

---

