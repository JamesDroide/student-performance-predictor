from sklearn.ensemble import RandomForestRegressor
import joblib

def entrenar_modelo(X_train, y_train):
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

def guardar_modelo(modelo, path):
    joblib.dump(modelo, path)
