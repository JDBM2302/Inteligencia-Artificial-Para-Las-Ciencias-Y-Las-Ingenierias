import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_select_and_evaluate(n=200, n_features=6):
    X = np.random.normal(0, 1, size=(n, n_features))
    n_relevant = np.random.randint(2, 5)
    coefs = np.array([np.random.uniform(2, 6) for _ in range(n_relevant)] +
                     [0] * (n_features - n_relevant))
    y = X @ coefs + np.random.normal(0, 0.5, size=n)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    k = np.random.randint(2, n_features)
    return {"df": df, "target_col": "target", "k": k}, k  # dict primero ✅
