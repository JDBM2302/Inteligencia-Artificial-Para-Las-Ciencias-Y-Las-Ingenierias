import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Generador de casos de uso
def create_case_1(n=100, n_features=4, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame(
        np.random.normal(0, 1, size=(n, n_features)),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    # Insertar algunos outliers
    df.iloc[5, 0] = 15
    df.iloc[20, 2] = -12
    df.iloc[50, 3] = 10
    return df

df = create_case_1()
print(df.head())
# result = detect_outliers(df, threshold=3)
