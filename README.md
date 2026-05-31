# Nama : Elisabeth Erni Banjarnahor

# Nim  : 312410525

# Kelas : I241E

# Matkul : Pengolahan Citra


# Penjelasan Arsitektur dan Alur Kerja Model

## 1. Alur Persiapan Data (Data Pipeline)

![foto](https://github.com/Elisabethbanjarnahor/Pengenalan-Pola/blob/8ba7859882a0739993a0faff1a16c02e934db72b/foto-hasil/Screenshot%202026-05-31%20232556.png)

Sebelum model kecerdasan buatan bisa belajar, data mentah (gambar) harus diubah menjadi format matematis yang bisa diproses oleh komputer.

* **Pembuatan Dataset:** Fungsi `tf.keras.utils.image_dataset_from_directory` membaca struktur folder. Fungsi ini secara otomatis memberikan label angka (0 hingga 4) berdasarkan urutan abjad nama folder (daisy, dandelion, rose, sunflower, tulip).
* **Standarisasi Dimensi:** Semua gambar dipaksa berubah ukuran menjadi 224x224 piksel. Ini wajib dilakukan karena lapisan jaringan saraf tiruan membutuhkan ukuran matriks input yang absolut dan seragam.
* **Pemecahan Data (Splitting):** Data dibagi menjadi 80% data latih (untuk mengajar model) dan 20% data validasi (sebagai ujian tertutup untuk menguji apakah model benar-benar paham, bukan sekadar menghafal).
* **Optimasi Memori:** Fungsi `prefetch(buffer_size=AUTOTUNE)` digunakan agar proses baca data dari hard disk berjalan paralel dengan proses komputasi di CPU/GPU. Ini mencegah terjadinya bottleneck atau antrean panjang saat pemrosesan.

---

## 2. Model 1: CNN dari Nol (Scratch)

![foto](https://github.com/Elisabethbanjarnahor/Pengenalan-Pola/blob/8ba7859882a0739993a0faff1a16c02e934db72b/foto-hasil/Screenshot%202026-05-31%20192346.png)

Pada pendekatan ini, arsitektur model dibangun lapis demi lapis. Otak dari model ini dimulai dari nol; nilai bobot (weights) matematikanya dimulai dari angka acak.

* **Rescaling:** Nilai piksel warna pada gambar aslinya berada di rentang 0 hingga 255. Lapisan ini menormalkannya menjadi skala 0 hingga 1. Jaringan saraf bekerja jauh lebih optimal dan stabil dengan angka desimal kecil.
* **Conv2D (Convolutional Layer):** Lapisan ini bertugas menyapu seluruh area gambar menggunakan filter matriks (ukuran 3x3). Tujuannya adalah mengekstrak fitur visual, mulai dari fitur tingkat rendah (garis, sudut, batas warna) hingga fitur tingkat tinggi (bentuk kelopak).
* **MaxPooling2D:** Lapisan ini melakukan kompresi pada hasil ekstraksi Conv2D. Ia mengambil nilai piksel dominan pada area tertentu sehingga dimensi gambar mengecil, namun informasi pentingnya tetap dipertahankan. Ini sangat menghemat beban komputasi.
* **Flatten:** Matriks gambar dua dimensi diratakan menjadi struktur larik (array) satu dimensi agar bisa dibaca oleh lapisan pembuat keputusan.
* **Dense & Dropout:** Lapisan Dense berfungsi memetakan fitur yang telah diekstrak ke dalam 5 kategori akhir. Lapisan Dropout (0.5) dengan sengaja mematikan 50% jalur saraf secara acak pada setiap iterasi. Tujuannya adalah mencegah model mengalami overfitting (kondisi di mana model menghafal data latih dengan sempurna, tetapi gagal menebak data baru).

---

## 3. Model 2: Transfer Learning (MobileNetV2)
Ini adalah pendekatan yang lebih modern dan efisien di industri. Daripada melatih jaringan dari nol, metode ini meminjam model (MobileNetV2) yang sudah dilatih oleh Google menggunakan dataset raksasa (ImageNet) yang berisi jutaan gambar dengan ribuan kategori objek.

* **Pembekuan Basis Pengetahuan (Freeze Weights):** Baris `base_model.trainable = False` adalah kunci utama dari metode ini. Jaringan MobileNetV2 sudah memiliki matriks bobot yang sangat ahli mendeteksi tekstur, kedalaman, dan bentuk geometris yang kompleks. Dengan membekukannya, pengetahuan dasar tersebut tidak akan hancur atau tertimpa saat dilatih ulang dengan dataset bunga yang jumlahnya sangat kecil.
* **Global Average Pooling:** Lapisan ini berfungsi sebagai alternatif yang lebih canggih dari lapisan Flatten. Alih-alih meratakan semua data yang menghasilkan jutaan parameter, lapisan ini mencari nilai rata-rata dari setiap peta fitur (feature map). Hal ini membuat model jauh lebih ringan dan kebal terhadap overfitting.
* **Classifier Head:** Bagian ekor dari arsitektur MobileNetV2 asli dibuang, lalu diganti dengan lapisan Dense baru yang memiliki 5 output saja. Pada proses pelatihan (training), komputasi yang berat hanya terjadi pada bagian ujung ini saja, tugasnya murni belajar mencocokkan fitur visual tingkat tinggi milik MobileNetV2 ke 5 nama bunga tersebut.

---

## 4. Evaluasi dan Analisis Tingkat Lanjut

![foto](https://github.com/Elisabethbanjarnahor/Pengenalan-Pola/blob/8ba7859882a0739993a0faff1a16c02e934db72b/foto-hasil/Screenshot%202026-05-31%20213114.png)

![foto](https://github.com/Elisabethbanjarnahor/Pengenalan-Pola/blob/8ba7859882a0739993a0faff1a16c02e934db72b/foto-hasil/Screenshot%202026-05-31%20213224.png)

![foto](https://github.com/Elisabethbanjarnahor/Pengenalan-Pola/blob/8ba7859882a0739993a0faff1a16c02e934db72b/foto-hasil/Screenshot%202026-05-31%20213236.png)

Kode tersebut tidak berhenti hanya dengan menampilkan angka akurasi, melainkan melakukan analisis mendalam mengenai titik buta (blind spot) dari model kecerdasan buatan.

* **Grafik Perbandingan (Plotting):** Visualisasi ini akan membuktikan bahwa kurva akurasi MobileNetV2 naik jauh lebih tajam dan stabil pada iterasi awal dibandingkan CNN biasa. Grafik Loss juga akan menunjukkan kapan model mulai menghafal (biasanya saat grafik loss validasi mulai berbelok naik).
* **Confusion Matrix (Matriks Kebingungan):** Ini adalah tabel diagnostik. Sumbu Y adalah jawaban yang benar (kunci jawaban), dan Sumbu X adalah tebakan model. Angka di luar garis diagonal utama menunjukkan letak kesalahan model. Ini berguna untuk mendeteksi kelemahan spesifik. Jika matriks menunjukkan angka tinggi pada perpotongan baris "Rose" dan kolom "Tulip", artinya model secara sistematis kesulitan membedakan fitur spesifik antara Mawar dan Tulip.
* **Misclassification Analysis (Analisis Kasus Sulit):** Program menjalankan prediksi ulang menggunakan data ujian, kemudian membandingkan array jawaban asli dengan array hasil prediksi. Fungsi `np.where(y_pred != y_true)` akan menangkap secara presisi indeks data gambar mana saja yang tebakannya meleset. Gambar tersebut kemudian ditampilkan beserta nilai probabilitas kepercayaan (confidence rate). Ini memberikan perspektif logis bagi engineer untuk menilai apakah kualitas gambar aslinya memang buruk, sudut pandangnya aneh, atau apakah arsitektur modelnya yang perlu disesuaikan ulang.
