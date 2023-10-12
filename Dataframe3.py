import pandas as pd
import dask.dataframe as dd
from dask_ml.preprocessing import MinMaxScaler
from dask_ml.model_selection import train_test_split
from dask_ml.naive_bayes import GaussianNB
from dask_ml.metrics import accuracy_score, recall_score, f1_score, roc_auc_score


ddf = dd.read_csv("DatasetFraude.csv")

scaler = MinMaxScaler()
numeric_features = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
ddf[numeric_features] = scaler.fit_transform(ddf[numeric_features])


X = ddf.drop("isFraud", axis=1)
y = ddf["isFraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Exactitud (Accuracy): {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"AUC-ROC: {roc_auc:.2f}")