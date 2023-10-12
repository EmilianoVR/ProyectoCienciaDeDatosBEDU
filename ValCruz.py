import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

def train_evaluate_model(df, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    # Columnas irrelevantes según la nueva información
    irrelevant_columns = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'nameOrig', 'nameDest']
    df.drop(irrelevant_columns, axis=1, inplace=True)

    df = pd.get_dummies(df, columns=["type"], prefix=["type"])
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest con los hiperparámetros especificados
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )

    # validación cruzada 
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    accuracy = scores.mean()

    print(f"Exactitud (Accuracy) promedio en validación cruzada: {accuracy:.2f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Imprimir las métricas en el conjunto de prueba
    print(f"Exactitud (Accuracy) en el conjunto de prueba: {accuracy:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"AUC-ROC: {roc_auc:.2f}")


data_file = "DatasetFraude.csv"
df = pd.read_csv(data_file)

# hiperparámetros 
n_estimators = 100
max_depth = 30
min_samples_split = 2
min_samples_leaf = 2


train_evaluate_model(df, n_estimators, max_depth, min_samples_split, min_samples_leaf)
