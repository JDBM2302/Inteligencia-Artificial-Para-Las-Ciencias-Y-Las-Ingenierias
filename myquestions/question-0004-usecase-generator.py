import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Generador de casos de uso
def create_case_4(n=150, seed=42):
    np.random.seed(seed)
    group1 = np.random.normal(loc=[0, 0],   scale=0.5, size=(n//3, 2))
    group2 = np.random.normal(loc=[5, 5],   scale=0.5, size=(n//3, 2))
    group3 = np.random.normal(loc=[0, 5],   scale=0.5, size=(n//3, 2))
    X = np.vstack([group1, group2, group3])
    df = pd.DataFrame(X, columns=["x", "y"])
    return df

df = create_case_4()
print(df.head())
# result = cluster_analysis(df, n_clusters=3)
