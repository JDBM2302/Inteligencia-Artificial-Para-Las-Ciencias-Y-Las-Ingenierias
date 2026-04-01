import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_detect_outliers(n=100, n_features=4):
    df = pd.DataFrame(
        np.random.normal(0, 1, size=(n, n_features)),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    # Insertar outliers en posiciones aleatorias
    for _ in range(np.random.randint(2, 6)):
        row = np.random.randint(0, n)
        col = np.random.randint(0, n_features)
        df.iloc[row, col] = np.random.choice([-1, 1]) * np.random.uniform(10, 15)
    return df

df = generar_caso_de_uso_detect_outliers()
print(df.head())
# result = detect_outliers(df, threshold=3)
