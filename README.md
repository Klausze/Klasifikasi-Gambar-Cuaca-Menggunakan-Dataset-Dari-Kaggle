# Submission: Klasifikasi Gambar Cuaca (Weather Image Classification)

## Deskripsi
Proyek ini membangun model CNN (Convolutional Neural Network) untuk mengklasifikasikan
gambar kondisi cuaca ke dalam 4 kelas: **cloudy**, **rain**, **shine**, dan **sunrise**.

## Dataset
- **Nama:** Multi-class Weather Dataset (MWD)
- **Sumber:** https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset
- **Jumlah gambar:** ±1125 gambar, 4 kelas
- Gambar asli memiliki resolusi yang tidak seragam (tanpa preprocessing awal).

## Arsitektur Model
`Sequential` CNN dengan 4 blok `Conv2D` + `BatchNormalization` + `MaxPooling2D`,
diikuti `Flatten`, `Dropout`, dan `Dense` layer dengan aktivasi `softmax` untuk 4 kelas.

## Pembagian Data
- Train: 70%
- Validation: 10%
- Test: 20%

Augmentasi data (rotasi, shift, shear, zoom, brightness, horizontal flip) hanya
diterapkan pada data training.

## Callback
Menggunakan kombinasi:
- Custom `AccuracyThresholdCallback` — menghentikan training saat akurasi train & validation mencapai target
- `EarlyStopping` — mencegah overfitting berlebihan
- `ReduceLROnPlateau` — menurunkan learning rate saat training stagnan

## Hasil
_(isi setelah menjalankan notebook)_
- Train accuracy: `__%`
- Test accuracy: `__%`

## Format Model yang Disimpan
- `saved_model/` — format TensorFlow SavedModel
- `tflite/model.tflite` + `tflite/label.txt` — format TensorFlow Lite
- `tfjs_model/` — format TensorFlow.js

## Cara Menjalankan
1. Buka `notebook.ipynb` di Google Colab.
2. Runtime > Change runtime type > pilih **T4 GPU**.
3. Runtime > Run all.
4. Saat diminta, upload `kaggle.json` (didapat dari kaggle.com/settings > Create New Token).
5. Setelah selesai, download folder `submission/` (berisi saved_model, tflite, tfjs_model,
   requirements.txt), tambahkan `notebook.ipynb` dan `README.md` ini, lalu kompres menjadi
   file `.zip` untuk dikumpulkan.

## Struktur Direktori Submission
```
submission
├───tfjs_model
│   ├───group1-shard1of1.bin
│   └───model.json
├───tflite
│   ├───model.tflite
│   └───label.txt
├───saved_model
│   ├───saved_model.pb
│   └───variables
├───notebook.ipynb
├───README.md
└───requirements.txt
```
