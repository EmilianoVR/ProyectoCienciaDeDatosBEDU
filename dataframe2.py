import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("DatasetFraude.csv")

df.drop(["nameOrig", "nameDest"], axis=1, inplace=True) # para la regresion logistica


df = pd.get_dummies(df, columns=["type"], prefix=["type"])

scaler = MinMaxScaler()
numerical_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

X = df.drop("isFraud", axis=1)
y = df["isFraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regresion logistica avr si funciona
model = LogisticRegression(max_iter=1000)  # Aumentar el número de iteraciones porque las normales no funcionaron (avr si funciona)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Prueba de Regresion logística")
print(f"Exactitud (Accuracy): {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"AUC-ROC: {roc_auc:.2f}")