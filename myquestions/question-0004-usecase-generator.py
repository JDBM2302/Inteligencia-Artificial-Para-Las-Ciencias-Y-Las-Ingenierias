import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def generar_caso_de_uso_cluster_analysis(n_clusters=3):
    n_per_group = np.random.randint(40, 80)
    centers = np.random.uniform(-10, 10, size=(n_clusters, 2))
    groups = [
        np.random.normal(loc=centers[i], scale=np.random.uniform(0.3, 1.0),
                        size=(n_per_group, 2))
        for i in range(n_clusters)
    ]
    X = np.vstack(groups)
    np.random.shuffle(X)
    return pd.DataFrame(X, columns=["x", "y"])

df = generar_caso_de_uso_cluster_analysis()
print(df.head())
# result = cluster_analysis(df, n_clusters=3)
