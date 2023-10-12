import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("DatasetFraude.csv")

# Eliminar las columnas irrelevantes para la detección de fraudes
irrelevant_columns = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'nameOrig', 'nameDest']
df.drop(irrelevant_columns, axis=1, inplace=True)
#Columnas irrelevantes segun la nueva información


df = pd.get_dummies(df, columns=["type"], prefix=["type"])

X = df.drop("isFraud", axis=1)
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hiperparametros
param_grid = {
    'n_estimators': [50, 100, 150, 200],  
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4]  
}

# Random Forest
model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)

grid_search.fit(X_train, y_train)

# mejores hiperparametros?
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Imprimir los mejores hiperparametros
print(f"Mejores hiperparámetros: {best_params}")
print(f"Exactitud (Accuracy) en el conjunto de prueba: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"AUC-ROC: {roc_auc:.2f}")
