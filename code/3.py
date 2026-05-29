import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ==========================================================
# 1. SIMULASI GENERATE DATASET KUSTOM (5 KELAS @100 GAMBAR)
# ==========================================================
print("=== 1. Membuat Dataset Simulasi Kustom (5 Kelas) ===")
np.random.seed(42)

# Mengikuti standar input MobileNetV2: 224x224 piksel dengan 3 channel (RGB)
X_dummy = np.random.rand(500, 224, 224, 3).astype(np.float32)
# Membuat label acak untuk 5 kelas (0, 1, 2, 3, 4)
y_dummy = np.random.randint(0, 5, 500)

# Membagi data menjadi 80% Latih (400 gambar) dan 20% Uji (100 gambar)
X_train, X_test = X_dummy[:400], X_dummy[400:]
y_train, y_test = y_dummy[:400], y_dummy[400:]

# ==========================================================
# 2. MEMBANGUN MODEL CNN DARI NOL (SCRATCH)
# ==========================================================
print("\n=== 2. Membangun Model CNN Biasa (Dari Nol) ===")
model_scratch = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(5, activation='softmax')  # 5 Output kelas
])

model_scratch.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model Scratch (Cukup 3 epoch agar cepat untuk simulasi kuliah)
print("Training Model CNN Scratch...")
history_scratch = model_scratch.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test), batch_size=32)

# ==========================================================
# 3. MEMBANGUN MODEL TRANSFER LEARNING (MOBILENETV2)
# ==========================================================
print("\n=== 3. Membangun Model Transfer Learning (MobileNetV2) ===")
# Mengambil base model MobileNetV2 yang sudah terlatih (pre-trained) di dataset ImageNet
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Membekukan bobot asli agar tidak berubah

model_tl = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(5, activation='softmax')
])

model_tl.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model Transfer Learning
print("Training Model Transfer Learning (MobileNetV2)...")
history_tl = model_tl.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test), batch_size=32)

# ==========================================================
# 4. VISUALISASI PERBANDINGAN PERFORMA & ANALISIS ERROR
# ==========================================================
acc_scratch = history_scratch.history['val_accuracy'][-1] * 100
acc_tl = history_tl.history['val_accuracy'][-1] * 100

# Plot Diagram Batang Perbandingan Akurasi
fig, ax = plt.subplots(figsize=(7, 5))
models_label = ['CNN Biasa (Scratch)', 'Transfer Learning (MobileNetV2)']
accuracies = [acc_scratch, acc_tl]

bars = ax.bar(models_label, accuracies, color=['#e74c3c', '#2ecc71'], width=0.5)
ax.set_ylabel('Akurasi Validasi (%)')
ax.set_title('Perbandingan Model: CNN Biasa vs Transfer Learning')
ax.set_ylim(0, 110)
ax.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

print("\nMenampilkan Grafik Perbandingan... (Tutup grafik untuk melihat kesimpulan teks)")
plt.show()

# Analisis Error (Identifikasi Kasus Sulit)
print("\n" + "="*60)
print("ANALISIS ERROR & IDENTIFIKASI KASUS SULIT (TUGAS 3)")
print("="*60)
# Simulasi prediksi untuk analisis error
predictions = model_tl.predict(X_test)
pred_classes = np.argmax(predictions, axis=1)
salah_prediksi = np.where(pred_classes != y_test)[0]

print(f"Total data uji: {len(y_test)} gambar")
print(f"Jumlah gambar yang salah diprediksi oleh MobileNetV2: {len(salah_prediksi)} gambar")
print("\nFaktor Penyebab Kasus Sulit (Analisis Teoretis Dataset Kustom):")
print("1. Keterbatasan Data: Jumlah data (100 per kelas) tergolong sangat kecil bagi CNN biasa untuk belajar fitur dari nol.")
print("2. Analisis Error Gambaran Fisik: Kasus sulit klasifikasi biasanya terjadi pada gambar yang memiliki latar belakang (background) yang terlalu ramai / mirip antar kelas, atau objek utama yang mengalami salah rotasi dan pencahayaan ekstrem.")
print("="*60)
