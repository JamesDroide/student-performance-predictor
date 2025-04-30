import joblib

# Cargamos el modelo entrenado para hacer predicciones
def cargar_modelo(path):
    modelo = joblib.load(path)
    return modelo

# Función para predecir
def predecir(modelo, X):
    return modelo.predict(X)
