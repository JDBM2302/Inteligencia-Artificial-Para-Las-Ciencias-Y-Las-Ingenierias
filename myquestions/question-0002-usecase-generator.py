import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Generador de casos de uso
def create_case_2(n=200, n_features=6, seed=42):
    np.random.seed(seed)
    X = np.random.normal(0, 1, size=(n, n_features))
    # Solo las primeras 3 features son realmente relevantes
    coefs = np.array([5, 3, 2, 0, 0, 0])
    y = X @ coefs + np.random.normal(0, 0.5, size=n)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    return df

df = create_case_2()
print(df.head())
# result = select_and_evaluate(df, target_col="target", k=3)
# esperado: best_features=['feature_0','feature_1','feature_2']
