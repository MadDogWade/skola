# MNIST ML-modell – Skoluppgift

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

print("Laddar MNIST-datasetet...")
mnist = fetch_openml("mnist_784", version=1, cache=True, as_frame=False)
X = mnist["data"]
y = mnist["target"]
print(f"Data laddad")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Träningsdata: {X_train.shape[0]} bilder")
print(f"Testdata:     {X_test.shape[0]} bilder")


print("\nTränar Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("Random Forest är REEEEDO!!")

print("\nTränar KNN (k=3)...")
knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn.fit(X_train, y_train)
print("KNN är nu REEEEDOOOOO!!!")


modeller = {"Random Forest": rf, "KNN": knn}
resultat = {}

for namn, modell in modeller.items():
    print(f" Resultat för {namn}")

    y_pred = modell.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    resultat[namn] = acc

    print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

bästa_namn = max(resultat, key=resultat.get)
bästa_modell = modeller[bästa_namn]


print(f" Bästa modellen: {bästa_namn} (accuracy: {resultat[bästa_namn]:.4f})")


filnamn = "basta_modell.joblib"
joblib.dump(bästa_modell, filnamn)
print(f"\nModellen sparad som '{filnamn}'")
