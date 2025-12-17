Repositori ini berisi proyek analisis sentimen terhadap opini pengguna mengenai layanan PayLater menggunakan algoritma BiLSTM (Bidirectional LSTM) dengan embedding FastText bahasa Indonesia.
Selain model dan tokenizer, repositori ini juga menyediakan aplikasi Streamlit untuk melakukan prediksi sentimen secara interaktif.

streamlit/
│
├── app.py                           # Aplikasi Streamlit utama
│
└── modelskripsi/
    ├── eksperimen1b.h5              # Model BiLSTM terlatih
    ├── testdata1.csv                # Dataset uji
    ├── testdata2.csv                # Dataset uji tambahan
    ├── tokenizer1.pickle            # Tokenizer utama
    └── tokenizer2.pkl               # Tokenizer cadangan
│
├── final_model_bilstm+fasttext.py   # Script training BiLSTM + FastText
└── final_model_bilstm.py            # Script training BiLSTM alternatif


1️⃣ Install dependencies
pip install -r requirements.txt


Jika belum membuat requirements.txt, rekomendasi:

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
streamlit run streamlit/app.py


Aplikasi akan berjalan otomatis di browser pada alamat:

http://localhost:8501

📦 Isi File Penting
File	Fungsi
app.py	Aplikasi Streamlit untuk prediksi sentimen
eksperimen1b.h5	Model BiLSTM terlatih
tokenizer1.pickle	Tokenizer untuk inference
testdata1.csv, testdata2.csv	Dataset evaluasi
final_model_bilstm+fasttext.py	Script training model final
final_model_bilstm.py	Script model alternatif
🧠 Tentang Model

Model deep learning menggunakan:

Embedding FastText (pretrained)

Bidirectional LSTM

Dense softmax

Adam optimizer

Prediksi sentimen antara:

Positif

Netral

Negatif

📊 Dataset

Dataset bersumber dari tweet pengguna Indonesia yang membahas fitur PayLater.
Dataset telah melalui proses:

crawling

preprocessing

pelabelan manual

pembagian train/test

Dataset lengkap tidak disertakan demi privasi.

✨ Rencana Pengembangan

📌 Tambahkan visualisasi performa model dalam Streamlit
📌 Tambahkan fitur upload CSV untuk analisis banyak data
📌 Bandingkan BiLSTM dengan IndoBERT untuk performa lanjutan

👤 Author

Agil Faturrahman
📩 Terbuka untuk diskusi tentang NLP, deep learning, dan sentiment analysis
