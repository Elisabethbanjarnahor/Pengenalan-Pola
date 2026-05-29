# Nama : Elisabeth Erni Banjarnahor

# Nim  : 312410525

# Kelas : I241E

# Matkul : Pengolahan Citra


---

## TUGAS 1: Implementasi KNN dari Nol (Fundamental)

### Output
![foto](https://github.com/Elisabethbanjarnahor/Pengenalan-Pola/blob/c0ff3ded3b1890b4f0e66c82833a389c7725f6f5/foto-hasil/Screenshot%202026-05-29%20180734.png)

### Deskripsi Tugas
Mengimplementasikan algoritma K-Nearest Neighbors (KNN) dari nol tanpa menggunakan library pihak ketiga (`sklearn`), membandingkan hasilnya dengan performa `sklearn`, serta mengevaluasi pengaruh nilai $K = (1, 3, 5, 7, 10, 15)$ menggunakan teknik **5-Fold Cross-Validation**.

### Cara Kerja Algoritma
1. **Pemisahan Data (K-Fold):** Dataset dibagi menjadi 5 bagian (*fold*). Secara bergantian, 4 bagian digunakan sebagai data latih dan 1 bagian sisanya sebagai data uji.
2. **Perhitungan Jarak:** Mengukur jarak matematis antara setiap data uji terhadap seluruh data latih menggunakan rumus **Euclidean Distance**:
   $$d = \sqrt{\sum_{i=1}^{n} (x_{uji,i} - x_{latih,i})^2}$$
3. **Voting Mayoritas:** Mengurutkan data dari jarak terkecil, mengambil sejumlah $K$ tetangga terdekat, dan menentukan kelas prediksi berdasarkan label mayoritas yang muncul.

### Hasil Pengujian dan Validasi Model
Implementasi KNN yang dibangun dari nol berhasil menghasilkan akurasi yang **identik (sama persis)** dengan library `scikit-learn` pada pengujian *5-fold cross-validation* menggunakan dataset Iris. Hasil pengujian komparatif untuk setiap nilai $K$ adalah sebagai berikut:
* **$K = 1$** : Rata-rata Akurasi **96.00%**
* **$K = 3$** : Rata-rata Akurasi **96.67%**
* **$K = 5$** : Rata-rata Akurasi **97.33%**
* **$K = 7$** : Rata-rata Akurasi **97.33%**
* **$K = 10$**: Rata-rata Akurasi **97.33%**
* **$K = 15$**: Rata-rata Akurasi **97.33%**

### Analisis dan Pembahasan
* **Validasi Keakuratan Logika:** Kesamaan metrik akurasi 100% antara model kustom dengan `scikit-learn` membuktikan secara matematis bahwa fungsi perhitungan *Euclidean Distance* serta mekanisme *voting* tetangga terdekat yang dirancang dari scratch telah berjalan dengan akurat.
* **Analisis Pengaruh Nilai K:**
  * Pada $K=1$, akurasi berada di angka 96.00%. Nilai $K$ yang terlalu kecil rentan terhadap *noise* (titik data yang menyimpang) karena keputusan klasifikasi hanya bergantung pada 1 tetangga terdekat saja.
  * Terjadi peningkatan performa menjadi 96.67% pada $K=3$ dan mencapai titik optimal sebesar 97.33% pada nilai $K=5, 7, 10,$ dan $15$.
  * Nilai $K$ yang lebih besar membuat batas keputusan (*decision boundary*) menjadi lebih halus (*smooth*). Hal ini meningkatkan kemampuan generalisasi model terhadap data uji, meskipun jika terlalu besar berisiko menyebabkan *underfitting*.

---

## TUGAS 2: Perbandingan Fitur Klasifikasi Tekstur (Menengah)

### Output

![foto](https://github.com/Elisabethbanjarnahor/Pengenalan-Pola/blob/c0ff3ded3b1890b4f0e66c82833a389c7725f6f5/foto-hasil/Screenshot%202026-05-29%20185419.png)

### Deskripsi Tugas
Membandingkan performa setidaknya 3 metode ekstraksi fitur tekstur komputer—**LBP, HOG, dan GLCM**—yang dikombinasikan dengan 2 jenis algoritma pengklasifikasi (*classifier*), yaitu **Support Vector Machine (SVM)** dan **Random Forest** pada dataset tekstur DTD (*Describable Textures Dataset*) simulasi.

### Cara Kerja Ekstraksi Fitur & Model
* **LBP (Local Binary Pattern):** Menganalisis tekstur mikro dengan membandingkan nilai biner intensitas piksel pusat terhadap 8 piksel tetangga di sekelilingnya secara melingkar.
* **HOG (Histogram of Oriented Gradients):** Menangkap struktur bentuk dan tepi objek melalui distribusi orientasi arah kecerahan gradien gambar.
* **GLCM (Gray-Level Co-occurrence Matrix):** Mengekstrak hubungan spasial piksel bertingkat abu-abu berdasarkan statistik *Contrast* dan *Homogeneity*.

### Hasil Pengujian Akurasi
Eksperimen pengujian klasifikasi tekstur menghasilkan performa berikut:
* **LBP (Local Binary Pattern):** SVM meraih akurasi **16.7%**, sedangkan Random Forest mencapai **100.0%**.
* **HOG (Histogram of Oriented Gradients):** SVM meraih akurasi **50.0%**, sedangkan Random Forest mencapai **100.0%**.
* **GLCM (Gray-Level Co-occurrence Matrix):** Baik SVM maupun Random Forest sama-sama sukses mendapatkan akurasi sempurna **100.0%**.

### Analisis Mendalam Performa Sistem
* **Keunggulan Random Forest Pada Fitur Tekstur:** Random Forest menunjukkan performa yang luar biasa stabil dengan meraih akurasi 100% pada ketiga metode ekstraksi fitur (LBP, HOG, GLCM). Hal ini terjadi karena Random Forest berbasis *Decision Trees* yang sangat kuat dalam menangkap pola non-linear dan batasan keputusan (*decision boundaries*) yang kompleks dari data tekstur buatan (garis, kotak, dan bintik), tanpa sensitif terhadap skala nilai fiturnya.
* **Kelemahan SVM Linier pada LBP dan HOG:** Model SVM yang digunakan menggunakan kernel linier. Pada fitur LBP (16.7%) and HOG (50.0%), nilai fiturnya memiliki dimensi dan sebaran statistik histogram yang membuat data kelas tekstur tersebut tidak dapat dipisahkan secara linier sempurna (*non-linearly separable*). Oleh karena itu, akurasi SVM jeblok pada kedua fitur ini.
* **GLCM Menjadi Fitur Terbaik untuk Kedua Model:** Fitur GLCM yang mengekstrak nilai statistik *Contrast* dan *Homogeneity* sukses membawa kedua model (SVM dan Random Forest) meraih akurasi 100%. Ini membuktikan bahwa hubungan spasial antar-piksel (derajat keabuan) adalah fitur yang paling diskriminatif dan paling mudah dipisahkan, bahkan oleh model linier sekalipun, untuk membedakan ketiga jenis tekstur pada dataset simulasi ini.

---

## TUGAS 3: Transfer Learning untuk Dataset Kustom (Lanjutan)

### Output

![foto](https://github.com/Elisabethbanjarnahor/Pengenalan-Pola/blob/c0ff3ded3b1890b4f0e66c82833a389c7725f6f5/foto-hasil/Screenshot%202026-05-29%20193044.png)

### Deskripsi Tugas
Menerapkan teknik **Transfer Learning** menggunakan arsitektur bawaan yang sudah terlatih (**MobileNetV2**) untuk mengklasifikasikan dataset citra kustom (5 kelas, minimal 100 gambar per kelas), kemudian membandingkan kinerjanya terhadap model **CNN biasa (Scratch)** yang dibangun dan dilatih dari nol, serta melakukan analisis error.

### Cara Kerja Arsitektur
* **CNN Scratch:** Model melatih parameter filter konvolusi secara mandiri dari awal. Memerlukan dataset masif agar konvergen dan rentan *overfitting* pada data kecil.
* **Transfer Learning (MobileNetV2):** Memanfaatkan lapisan pembawa fitur kaya (*feature extractor*) hasil pelatihan terdahulu pada jutaan citra ImageNet, kemudian membekukan bobot asli (*freeze*) dan menyesuaikan lapisan atas (*dense layer*) untuk mengenali kelas kustom baru.

### Hasil Pengujian Perbandingan
Pengujian performa klasifikasi citra pada dataset kustom menghasilkan nilai akurasi validasi akhir sebagai berikut:
* **CNN Biasa (Scratch Model):** **20.00%**
* **Transfer Learning (MobileNetV2):** **24.00%**

### Analisis dan Pembahasan
* **Interpretasi Performa:** Hasil akurasi CNN Biasa berada tepat di angka 20.00%. Nilai ini mencerminkan batas bawah performa klasifikasi 5 kelas secara acak (*baseline probability* 1/5). Hal ini membuktikan bahwa model CNN dari nol sangat kesulitan melakukan konvergensi dan mengenali pola esensial jika dihadapkan pada batasan jumlah sampel data kustom yang minimal (100 gambar per kelas) tanpa adanya optimasi arsitektur yang mendalam.
* **Keunggulan MobileNetV2:** Model MobileNetV2 berhasil mendapatkan akurasi lebih tinggi, yaitu 24.00%. Keunggulan ini didapat karena MobileNetV2 bertindak sebagai *feature extractor* yang kuat. Bobot (*weights*) model ini telah terlatih sebelumnya (*pre-trained*) menggunakan dataset raksasa ImageNet, sehingga ia memiliki modal pemahaman struktur visual (seperti bentuk tepi, pencahayaan, dan gradasi warna) yang jauh lebih baik daripada model biasa.

### Analisis Error dan Kasus Sulit (*Error Analysis*)
Melalui proses evaluasi data uji, diidentifikasi beberapa faktor utama yang memicu terjadinya kasus sulit klasifikasi (*misclassification*) pada dataset kustom:
* **Keterbatasan Kuantitas Data (*Data Scarcity*):** Sesuai instruksi, batas minimal 100 gambar per kelas merupakan jumlah yang sangat minim bagi sebuah arsitektur *Deep Learning* konvensional. Hal ini memicu terjadinya *overfitting* di mana model hanya menghafal data latih dan gagal melakukan generalisasi pada data uji.
* **Tingginya Kemiripan Visual Antar Kelas (*High Inter-class Similarity*):** Kasus salah prediksi sering terjadi pada objek antar-kelas yang memiliki karakteristik bentuk geometris atau warna dominan yang mirip, sehingga membingungkan lapisan *Dense Layer* di bagian akhir klasifikasi.
* **Gangguan Latar Belakang (*Background Noise*):** Gambar kustom sering kali diambil dengan kondisi lingkungan yang tidak steril (latar belakang ramai). Akibatnya, model rentan salah fokus dengan mengekstrak fitur *background* alih-alih objek utama yang ingin diklasifikasikan.

---

## syarat 
Pastikan dependensi berikut sudah terpasang di komputer Anda sebelum menjalankan skrip kode:
```bash
pip install numpy matplotlib scikit-learn scikit-image tensorflow
