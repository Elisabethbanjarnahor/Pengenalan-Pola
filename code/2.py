import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

# Mengabaikan warning kecil agar output terminal bersih dan rapi
warnings.filterwarnings('ignore')

# ==========================================
# 1. KONFIGURASI DATASET
# ==========================================
DATASET_PATH = r"C:\mydocument\praktik_p_citra"
IMG_SIZE = (128, 128)
# 5 kelas tekstur sesuai yang ada di folder kamu
class_folders = ['banded', 'blotchy', 'braided', 'bubbly', 'bumpy']

# ==========================================
# 2. FUNGSI EKSTRAKSI FITUR (LBP, HOG, GLCM)
# ==========================================
def extract_lbp(gray_img):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    
    # PERBAIKAN DI BARIS INI:
    # Ubah n_bins = int(lbp.max() + 1) menjadi:
    n_bins = n_points + 2
    
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

def extract_hog(gray_img):
    features = hog(gray_img, orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), visualize=False)
    return features

def extract_glcm(gray_img):
    img_uint = (gray_img * 255).astype(np.uint8)
    glcm = graycomatrix(img_uint, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').ravel()
    correlation = graycoprops(glcm, 'correlation').ravel()
    energy = graycoprops(glcm, 'energy').ravel()
    homogeneity = graycoprops(glcm, 'homogeneity').ravel()
    return np.hstack([contrast, correlation, energy, homogeneity])

# ==========================================
# 3. LOADING DATA DAN PROSES EKSTRAKSI
# ==========================================
print("=" * 60)
print("Mulai memuat dataset DTD dan mengekstrak fitur...")
print("=" * 60)

features_lbp, features_hog, features_glcm, labels = [], [], [], []

for idx, class_name in enumerate(class_folders):
    class_dir = os.path.join(DATASET_PATH, class_name)
    # Ambil semua file gambar di dalam folder kelas
    img_names = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"-> Memproses kelas [{class_name}] : Ditemukan {len(img_names)} gambar. Silakan tunggu...")
    
    for img_name in img_names:
        img_path = os.path.join(class_dir, img_name)
        try:
            img = io.imread(img_path)
            if img.ndim == 3:
                gray = color.rgb2gray(img)
            else:
                gray = img / 255.0
            
            gray_resized = transform.resize(gray, IMG_SIZE)
            
            # Ekstraksi fitur
            features_lbp.append(extract_lbp(gray_resized))
            features_hog.append(extract_hog(gray_resized))
            features_glcm.append(extract_glcm(gray_resized))
            labels.append(idx)
        except Exception as e:
            continue

X_lbp = np.array(features_lbp)
X_hog = np.array(features_hog)
X_glcm = np.array(features_glcm)
y = np.array(labels)

print("\n" + "=" * 60)
print("PROSES EKSTRAKSI SELESAI!")
print(f"Total gambar yang berhasil diproses: {len(y)}")
print("=" * 60 + "\n")

# ==========================================
# 4. KLASIFIKASI & EVALUASI PERFORMA
# ==========================================
print("Mulai melatih model dan menghitung akurasi...")
feature_sets = {'LBP': X_lbp, 'HOG': X_hog, 'GLCM': X_glcm}
classifiers = {
    'SVM': SVC(kernel='linear', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for f_name, X_data in feature_sets.items():
    # Split: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y, test_size=0.2, random_state=42, stratify=y
    )
    for c_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[f"{f_name} + {c_name}"] = acc
        print(f"[Sukses] Akurasi Kombinasi [{f_name} + {c_name}]: {acc:.4f}")

# ==========================================
# 5. VISUALISASI HASIL KINERJA
# ==========================================
plt.figure(figsize=(11, 6))

combinations = list(results.keys())
accuracies = list(results.values())

# Urutkan hasil agar grafik batang berurutan
sorted_indices = np.argsort(accuracies)
combinations = [combinations[i] for i in sorted_indices]
accuracies = [accuracies[i] for i in sorted_indices]

colors = ['#E6E6FA', '#DDA0DD', '#DA70D6', '#BA55D3', '#9370DB', '#7B68EE']
bars = plt.barh(combinations, accuracies, color=colors, edgecolor='grey', height=0.6)

plt.xlabel('Akurasi (0.0 - 1.0)', fontsize=11, fontweight='bold', labelpad=10)
plt.title('Perbandingan Performa Ekstraksi Fitur Tekstur\n(5 Kelas DTD - 120 Gambar Per Kelas)', 
          fontsize=13, fontweight='bold', pad=15)
plt.xlim(0, 1.1)
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Tampilkan label nilai akurasi di setiap bar
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
             f'{width:.4f}',
             va='center', ha='left', fontsize=10, fontweight='bold', color='black')

plt.tight_layout()
plt.show()
