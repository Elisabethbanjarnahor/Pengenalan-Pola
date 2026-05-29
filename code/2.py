import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops

# ==========================================================
# 1. SIMULASI GENERATE DATASET TEKSTUR (DUMMY DATASET DTD)
# ==========================================================
def buat_dataset_tekstur_dummy(jumlah_sampel=60):
    np.random.seed(42)
    X_raw = []
    y = []
    for kelas in range(3):
        for _ in range(jumlah_sampel // 3):
            if kelas == 0:
                img = np.zeros((64, 64))
                img[:, ::4] = 255
            elif kelas == 1:
                img = np.zeros((64, 64))
                img[::4, :] = 255
                img[:, ::4] = 255
            else:
                img = np.random.randint(0, 256, (64, 64))
            img = img + np.random.normal(0, 10, (64, 64))
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            X_raw.append(img)
            y.append(kelas)
            
    return X_raw, np.array(y)

# ==========================================================
# 2. FUNGSI EKSTRAKSI FITUR (LBP, HOG, GLCM)
# ==========================================================
def ekstrak_lbp(images):
    fitur_lbp = []
    for img in images:
        lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
        fitur_lbp.append(hist)
    return np.array(fitur_lbp)

def ekstrak_hog(images):
    fitur_hog = []
    for img in images:
        hf = hog(img, orientations=8, pixels_per_cell=(16, 16), 
                 cells_per_block=(1, 1), visualize=False)
        fitur_hog.append(hf)
    return np.array(fitur_hog)

def ekstrak_glcm(images):
    fitur_glcm = []
    for img in images:
        glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        fitur_glcm.append([contrast, homogeneity])
    return np.array(fitur_glcm)

# Load data citra simulasi tekstur
X_raw, y = buat_dataset_tekstur_dummy()

# Ekstraksi ke 3 metode
X_lbp = ekstrak_lbp(X_raw)
X_hog = ekstrak_hog(X_raw)
X_glcm = ekstrak_glcm(X_raw)

daftar_fitur = {'LBP': X_lbp, 'HOG': X_hog, 'GLCM': X_glcm}
daftar_classifier = {'SVM': SVC(kernel='linear'), 'Random Forest': RandomForestClassifier(random_state=42)}
nama_fitur_list = list(daftar_fitur.keys())

# ==========================================================
# 3. PROSES TRAINING & EVALUASI (SUDAH DIPERBAIKI)
# ==========================================================
hasil_akurasi = {'SVM': [], 'Random Forest': []}

for clf_name, clf in daftar_classifier.items():
    for f_name in nama_fitur_list:
        X_fitur = daftar_fitur[f_name]
        X_train, X_test, y_train, y_test = train_test_split(X_fitur, y, test_size=0.2, random_state=42)
        
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # PERBAIKAN: Menyimpan hasil akurasi ke dalam dictionary pendukung
        hasil_akurasi[clf_name].append(acc * 100)

# ==========================================================
# 4. VISUALISASI HASIL & DIAGRAM BATANG
# ==========================================================
x = np.arange(len(nama_fitur_list))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, hasil_akurasi['SVM'], width, label='SVM', color='#3498db')
rects2 = ax.bar(x + width/2, hasil_akurasi['Random Forest'], width, label='Random Forest', color='#e67e22')

ax.set_ylabel('Akurasi (%)')
ax.set_title('Perbandingan Performa Ekstraksi Fitur Tekstur dan Classifier')
ax.set_xticks(x)
ax.set_xticklabels(nama_fitur_list)
ax.set_ylim(0, 110)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
print("Menampilkan Grafik Analisis... (Tutup jendela grafik untuk melihat rangkuman teks)")
plt.show()

# ==========================================================
# 5. PRINT RANGKUMAN TEKS KESIMPULAN
# ==========================================================
print("\n" + "="*60)
print(f"{'Metode Fitur':<15} | {'Akurasi SVM':<15} | {'Akurasi Random Forest':<15}")
print("="*60)
for idx, f_name in enumerate(nama_fitur_list):
    print(f"{f_name:<15} | {hasil_akurasi['SVM'][idx]:.2f}%{'' : <8} | {hasil_akurasi['Random Forest'][idx]:.2f}%")
print("="*60)
