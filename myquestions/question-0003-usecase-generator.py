import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Generador de casos de uso
def create_case_3(n=300, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        "age":      np.random.randint(18, 65, size=n),
        "income":   np.random.normal(50000, 15000, size=n),
        "city":     np.random.choice(["Bogota", "Medellin", "Cali"], size=n),
        "edu":      np.random.choice(["high_school", "university", "masters"], size=n),
        "target":   np.random.randint(0, 2, size=n)
    })
    return df

df = create_case_3()
print(df.head())
# result = classify(df, target_col="target")
