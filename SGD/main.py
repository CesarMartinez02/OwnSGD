import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report

from src.SGDclass import SGDclass  # Importa desde subcarpeta src

# Cargar datos desde carpeta 'data'
df = pd.read_csv("data/datos.csv")

#Limpieza básica y corrección de caracteres especiales
df.columns = df.columns.str.strip().str.upper().str.replace(' ', '_')
df.drop_duplicates(inplace=True)
cols_corr = df.select_dtypes(include='object').columns
for col in cols_corr:
    df[col] = df[col].astype(str).str.encode('latin1', errors='ignore').str.decode('utf-8', errors='ignore')
df.dropna(inplace=True)


# Separar X (características) e y (etiqueta)
X = df.drop(columns='target')
y = df['target']

# Identificar columnas
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include='number').columns.tolist()

# Preprocesador
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Fit-transform
X_processed = preprocessor.fit_transform(X).toarray()

# División
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, stratify=y, random_state=42
)

# Convertir a arrays planos
X_train = pd.DataFrame(X_train).reset_index(drop=True).values
y_train = pd.Series(y_train).reset_index(drop=True).values
X_test = pd.DataFrame(X_test).reset_index(drop=True).values
y_test = pd.Series(y_test).reset_index(drop=True).values

# Entrenar
model = SuperSGDClassifier(lr=0.02, n_epochs=1500, threshold=0.5)
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)

# Evaluación
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
