import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

class KNNDariNol:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        
        k_indices = np.argsort(distances)[:self.k]
        
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


iris = load_iris()
X = iris.data
y = iris.target

nilai_k = [1, 3, 5, 7, 10, 15]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

hasil_knn_nol = {k: [] for k in nilai_k}
hasil_knn_sklearn = {k: [] for k in nilai_k}

print("=== Memulai Proses 5-Fold Cross-Validation ===\n")

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    for k in nilai_k:
        model_nol = KNNDariNol(k=k)
        model_nol.fit(X_train, y_train)
        pred_nol = model_nol.predict(X_test)
        acc_nol = accuracy_score(y_test, pred_nol)
        hasil_knn_nol[k].append(acc_nol)
        model_sklearn = KNeighborsClassifier(n_neighbors=k)
        model_sklearn.fit(X_train, y_train)
        pred_sklearn = model_sklearn.predict(X_test)
        acc_sklearn = accuracy_score(y_test, pred_sklearn)
        hasil_knn_sklearn[k].append(acc_sklearn)

print("-" * 60)
print(f"{'Nilai K':<10} | {'Rata-rata Akurasi (Dari Nol)':<30} | {'Rata-rata Akurasi (Sklearn)':<30}")
print("-" * 60)

for k in nilai_k:
    avg_acc_nol = np.mean(hasil_knn_nol[k])
    avg_acc_sklearn = np.mean(hasil_knn_sklearn[k])
    print(f"{k:<10} | {avg_acc_nol * 100:<28.2f}% | {avg_acc_sklearn * 100:<28.2f}%")

print("-" * 60)
print("\nAnalisis Kesimpulan:")
print("1. Jika persentase akurasi KNN Dari Nol dan Sklearn bernilai SAMA, artinya implementasi rumus logika dasar kita sudah 100% benar.")
print("2. Perubahan nilai K mempengaruhi akurasi karena menentukan seberapa sensitif model terhadap noise/titik data di sekitarnya.")
