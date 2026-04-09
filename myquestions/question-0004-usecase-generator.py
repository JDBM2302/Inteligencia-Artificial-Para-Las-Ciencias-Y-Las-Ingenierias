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
    df = pd.DataFrame(X, columns=["x", "y"])
    n_clusters_param = np.random.randint(2, 5)
    return {"df": df, "n_clusters": n_clusters_param}, n_clusters_param  # dict primero ✅
