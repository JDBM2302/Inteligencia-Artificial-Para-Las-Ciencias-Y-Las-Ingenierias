import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def generar_caso_de_uso_classify(n=300):
    ciudades = np.random.choice(["Bogota", "Medellin", "Cali", "Barranquilla"], size=n)
    edu = np.random.choice(["high_school", "university", "masters"], size=n)
    age = np.random.randint(18, 65, size=n)
    income = np.random.normal(np.random.uniform(40000, 80000), 15000, size=n)
    target = (income > np.median(income)).astype(int)
    return pd.DataFrame({
        "age": age, "income": income,
        "city": ciudades, "edu": edu,
        "target": target
    })

df = generar_caso_de_uso_classify()
print(df.head())
# result = classify(df, target_col="target")
