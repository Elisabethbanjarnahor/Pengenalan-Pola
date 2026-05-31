import os
import warnings
import logging

# ==========================================
# 1. BUNGKAM WARNING TENSORFLOW 
# ==========================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==========================================
# 2. KONFIGURASI DATASET
# ==========================================
DATASET_DIR = r"C:\mydocument\praktik_p_citra\flowers"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 10

print("=" * 50)
print("Memuat Dataset dan Mempersiapkan Gambar...")
print("=" * 50)

# Load Data Training (80%) 
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load Data Validation (20%) 
val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
print(f"Kelas yang terdeteksi: {class_names}")

# Optimasi performa memori
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. MODEL 1: CNN DARI NOL (SCRATCH)
# ==========================================
print("\n[1/2] Melatih Model CNN dari Nol (Scratch)...")
cnn_model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

cnn_model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

history_cnn = cnn_model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

# ==========================================
# 4. MODEL 2: TRANSFER LEARNING (MobileNetV2)
# ==========================================
print("\n[2/2] Melatih Model Transfer Learning (MobileNetV2)...")
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
base_model = applications.MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                                      include_top=False,
                                      weights='imagenet')

base_model.trainable = False # Bekukan bobot bawaan

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
tl_model = tf.keras.Model(inputs, outputs)

tl_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

history_tl = tl_model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

# ==========================================
# 5. PERBANDINGAN PERFORMA (VISUALISASI GRAFIK)
# ==========================================
plt.figure(figsize=(12,5))

# Plot Akurasi
plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['val_accuracy'], label='CNN Scratch (Val)', color='#BA55D3', linestyle='--')
plt.plot(history_tl.history['val_accuracy'], label='MobileNetV2 (Val)', color='#4B0082', linewidth=2)
plt.title('Perbandingan Akurasi Validasi', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history_cnn.history['val_loss'], label='CNN Scratch (Val)', color='#BA55D3', linestyle='--')
plt.plot(history_tl.history['val_loss'], label='MobileNetV2 (Val)', color='#4B0082', linewidth=2)
plt.title('Perbandingan Loss Validasi', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# ==========================================
# 6. ANALISIS ERROR & IDENTIFIKASI KASUS SULIT
# ==========================================
print("\n--- ANALISIS ERROR (MobileNetV2) ---")
print("Mengekstrak gambar untuk analisis...")

# Ekstrak data gambar dan label secara manual agar urutannya terkunci aman
val_images = []
val_labels = []

for x, y in val_dataset.unbatch():
    val_images.append(x.numpy())
    val_labels.append(y.numpy())

val_images = np.array(val_images)
y_true = np.array(val_labels)

print("Melakukan prediksi pada data uji...")
y_pred_probs = tl_model.predict(val_images, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred_probs, axis=1)

misclassified_indices = np.where(y_pred != y_true)[0]
print(f"\n[Hasil] Total gambar salah prediksi: {len(misclassified_indices)} dari {len(y_true)} gambar uji.")

# Visualisasi Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - MobileNetV2', fontweight='bold', pad=15)
plt.ylabel('Label Asli (True)')
plt.xlabel('Prediksi Model (Predicted)')
plt.show()

# Tampilkan 4 contoh kasus sulit
if len(misclassified_indices) > 0:
    plt.figure(figsize=(12, 10))
    plt.suptitle("Kasus Sulit (Salah Prediksi oleh MobileNetV2)", fontsize=16, fontweight='bold')
    
    for i, bad_idx in enumerate(misclassified_indices[:4]):
        plt.subplot(2, 2, i + 1)
        img = val_images[bad_idx].astype("uint8")
        plt.imshow(img)
        
        true_label = class_names[y_true[bad_idx]]
        pred_label = class_names[y_pred[bad_idx]]
        confidence = y_pred_probs[bad_idx][y_pred[bad_idx]] * 100
        
        plt.title(f"Label Asli: {true_label.upper()}\nDitebak: {pred_label.upper()} ({confidence:.1f}%)",
                  color='darkred', fontweight='bold')
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
