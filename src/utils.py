from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Función para evaluar el modelo entrenado usando MSE, MAE y R2
def evaluar_modelo(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    return mae, mse, r2
