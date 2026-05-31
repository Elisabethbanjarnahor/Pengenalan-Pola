import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. IMPLEMENTASI KNN DARI NOL
class KNNManual:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # KNN tidak memiliki proses 'training' yang kompleks,
        # model hanya perlu menyimpan data latih.
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        # Lakukan prediksi untuk setiap data poin di X
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # a. Hitung jarak (Euclidean distance) antara x dengan semua data latih
        # Rumus Euclidean: akar dari jumlah kuadrat selisih
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        
        # b. Urutkan jarak dari yang terkecil dan ambil indeks 'k' terdekat
        k_indices = np.argsort(distances)[:self.k]
        
        # c. Ambil label target dari 'k' tetangga terdekat tersebut
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # d. Lakukan voting mayoritas (kelas apa yang paling banyak muncul)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# ==========================================
# 2. PERSIAPAN DATA & PENGATURAN EVALUASI
# ==========================================
# Load dataset Iris
dataset = load_iris()
X = dataset.data
y = dataset.target

# Nilai K yang diminta dalam soal
k_values = [1, 3, 5, 7, 10, 15]

# Setup 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# List untuk menyimpan rata-rata akurasi
akurasi_manual_list = []
akurasi_sklearn_list = []

# ==========================================
# 3. PROSES EVALUASI & PERBANDINGAN
# ==========================================
print("Mulai evaluasi 5-Fold Cross Validation...\n")

for k in k_values:
    fold_acc_manual = []
    fold_acc_sklearn = []
    
    # Looping untuk setiap lipatan (fold) data
    for train_index, test_index in kf.split(X):
        # Pisahkan data latih dan uji
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # --- Uji KNN Manual ---
        knn_manual = KNNManual(k=k)
        knn_manual.fit(X_train, y_train)
        y_pred_manual = knn_manual.predict(X_test)
        acc_manual = accuracy_score(y_test, y_pred_manual)
        fold_acc_manual.append(acc_manual)
        
        # --- Uji KNN Sklearn (Sebagai Pembanding) ---
        knn_sklearn = KNeighborsClassifier(n_neighbors=k)
        knn_sklearn.fit(X_train, y_train)
        y_pred_sklearn = knn_sklearn.predict(X_test)
        acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
        fold_acc_sklearn.append(acc_sklearn)
        
    # Hitung rata-rata akurasi dari 5 fold
    akurasi_manual_list.append(np.mean(fold_acc_manual))
    akurasi_sklearn_list.append(np.mean(fold_acc_sklearn))

# ==========================================
# 4. MENAMPILKAN HASIL AKHIR
# ==========================================
print(f"| {'Nilai K':<10} | {'Akurasi KNN Manual':<20} | {'Akurasi KNN Sklearn':<20} |")
print("-" * 60)
for i, k in enumerate(k_values):
    print(f"| K={k:<8} | {akurasi_manual_list[i]:<20.4f} | {akurasi_sklearn_list[i]:<20.4f} |")
