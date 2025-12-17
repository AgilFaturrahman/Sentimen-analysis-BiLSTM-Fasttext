# 💳 Sentiment Analysis PayLater  
### 📌 BiLSTM + FastText + Streamlit  
Proyek ini melakukan analisis sentimen pada opini pengguna terkait layanan PayLater menggunakan algoritma **BiLSTM** yang dipadukan dengan embedding **FastText Bahasa Indonesia**.  
Dilengkapi antarmuka **Streamlit** untuk memprediksi sentimen secara real-time.  

---

## 📁 Struktur Project  

streamlit/
│
├── app.py # Aplikasi Streamlit utama
│
└── modelskripsi/
├── eksperimen1b.h5 # Model BiLSTM terlatih
├── testdata1.csv # Dataset uji 1
├── testdata2.csv # Dataset uji 2
├── tokenizer1.pickle # Tokenizer utama
└── tokenizer2.pkl # Tokenizer cadangan
│
├── final_model_bilstm+fasttext.py # Script training model + FastText
└── final_model_bilstm.py # Script training BiLSTM alternatif

yaml
Salin kode

---

## 🚀 Menjalankan Streamlit  

### 1️⃣ Install dependencies  
```bash
pip install -r requirements.txt
Jika belum memiliki requirements.txt, gunakan:

nginx
Salin kode
streamlit
tensorflow
numpy
pandas
scikit-learn
fasttext
pickle-mixin
plotly
matplotlib
2️⃣ Jalankan aplikasi
bash
Salin kode
streamlit run streamlit/app.py
🔗 Aplikasi otomatis terbuka di:
👉 http://localhost:8501

📊 Contoh Output Prediksi
Input:
arduino
Salin kode
"Fitur paylaternya bener-bener membantu di saat mendesak!"
Output:
yaml
Salin kode
Sentimen : Positif ⭐⭐⭐⭐⭐
🧠 Arsitektur Model
java
Salin kode
FastText Embedding
        ↓
Bi-directional LSTM
        ↓
     Dense Layer
        ↓
 Softmax Output (3 kelas)
🎯 Kategori sentimen:

😃 Positif

😐 Netral

😠 Negatif

📦 Dataset
Dataset berasal dari opini pengguna berbahasa Indonesia yang membahas PayLater.
Dataset telah diproses melalui:
✔ pengumpulan otomatis
✔ preprocessing teks
✔ pelabelan manual sentimen
✔ pembagian train/test

📌 File Penting
File	Fungsi
app.py	UI Streamlit interaktif
eksperimen1b.h5	Model terlatih
tokenizer1.pickle	Tokenizer inference
final_model_bilstm+fasttext.py	Script training final
final_model_bilstm.py	Alternatif model
testdata1.csv	Dataset uji
testdata2.csv	Dataset uji tambahan

📈 Visualisasi Model (Konsep Akurasi)
yaml
Salin kode
Akurasi Model: ████████████████░░ 87%
Loss Model   : ████████░░░░░░░░░░ 32%
📌 (Notasi batang bersifat ilustrasi)

✨ Fitur Mendatang
🟢 Tambahkan analisis file CSV upload
🟢 Tampilkan grafik performa di Streamlit
🟢 Bandingkan hasil dengan IndoBERT
🟢 Deploy aplikasi ke HuggingFace/Render

👤 Author
Agil Faturrahman
📫 Siap berdiskusi tentang NLP, Machine Learning, dan Deep Learning

⭐ Dukung Project Ini
Jika project ini bermanfaat:
👍 Beri star ⭐ di GitHub
🔁 Share repo ini
🤝 Kolaborasi pengembangan

nginx
Salin kode
Thank you! 💛
