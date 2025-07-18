# main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import os

# 📂 Leer CSV local
archivo = "Anexo ET_demo_round_traces_2022.csv"  # Asegúrate del nombre real
df = pd.read_csv(archivo, sep=';', encoding='latin1')
df.columns = df.columns.str.replace(';', '')

# 🧹 Limpieza de columnas
df = df.drop(columns=['Unnamed: 0', 'AbnormalMatch'], errors='ignore')

# 🩹 Correcciones manuales
df['RoundWinner'] = df['RoundWinner'].replace('False4', 'False')
df['MatchWinner'] = df['MatchWinner'].fillna('False')

# 🔁 Conversión booleana a binario
df['RoundWinner'] = df['RoundWinner'].astype(str).map({'True': 1, 'False': 0})
df['MatchWinner'] = df['MatchWinner'].astype(str).map({'True': 1, 'False': 0})
df['Survived'] = df['Survived'].astype(str).map({'True': 1, 'False': 0})

# 📊 Variables predictoras
features = ['TimeAlive', 'TravelledDistance', 'RoundKills', 'MatchKills',
            'RoundAssists', 'MatchAssists', 'RoundHeadshots']

for col in features:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=features + ['RoundWinner'])

# 🎯 División de variables
X = df[features]
y = df['RoundWinner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Entrenamiento
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 💾 Guardar modelo
joblib.dump(model, "modelo.pkl")
print("✅ Modelo entrenado y guardado como modelo.pkl")

# 📈 Curva ROC
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# 🖼️ Guardar curva
os.makedirs("static", exist_ok=True)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC para Predicción de Ganador de Ronda')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("static/roc_curve.png")
print("📸 Curva ROC guardada en static/roc_curve.png")