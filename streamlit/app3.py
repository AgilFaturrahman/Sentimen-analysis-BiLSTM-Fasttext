import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os

# Import plotting dan preprocessing
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- 1. KONFIGURASI HALAMAN & CSS ---
st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Menambahkan CSS kustom untuk mengimpor dan menggunakan font Poppins
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    /* Terapkan Poppins hanya pada teks biasa, judul, dan elemen input */
    html, body, h1, h2, h3, h4, h5, h6, .st-b5, .stTextInput, .stTextArea, .stForm, .stButton {
        font-family: 'Poppins', sans-serif;
    }
    
    /* General Style */
    .main-header {
        font-size: 2.5rem;
        color: #1E90FF; /* DodgerBlue */
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .stButton > button {
        width: 200px; /* Lebar spesifik untuk tombol */
        border-radius: 12px;
        font-weight: bold;
        padding: 10px 20px;
    }
    
    /* Solusi untuk memusatkan tombol 'Analisis' */
    div[data-testid="stForm"] .stButton {
        display: flex;
        justify-content: center;
        width: 100%;
        margin-top: 15px; /* Sedikit spasi di atas tombol */
        margin-bottom: 15px; /* Sedikit spasi di bawah tombol */
    }

    /* Solusi untuk memusatkan tombol 'Jalankan Evaluasi' */
    div[data-testid="stExpander"] .stButton {
        display: flex;
        justify-content: center;
        width: 100%;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
    /* Prediction Box Style */
    .prediction-box {
        background: linear-gradient(135deg, #00BFFF 0%, #1E90FF 100%); /* DeepSkyBlue to DodgerBlue */
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center; /* Ini tetap center untuk kotak hasil prediksi */
        margin-top: 1.5rem;
        border: 2px solid #FFFFFF;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .prediction-box h2 {
        font-size: 2.2rem;
        font-weight: bold;
        margin: 0;
    }
    .prediction-box h3 {
        font-size: 1.25rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    /* Style untuk area hasil dan evaluasi */
    .stExpander {
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-top: 1.5rem;
    }
    /* Style untuk merapikan form */
    .stForm {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    .stFileUploadDropzone {
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #ccc;
    }
    /* Memastikan pesan sukses/error sejajar kiri */
    div[data-testid="stAlert"] {
        text-align: left !important;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. FUNGSI PEMUATAN ASET ---

@st.cache_resource
def load_assets(model_path, tokenizer_path):
    """Memuat model BiLSTM dan tokenizer yang tersimpan."""
    try:
        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            st.error(f"Error: File tidak ditemukan. Pastikan '{model_path}' dan '{tokenizer_path}' ada.")
            return None, None
        model = load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error fatal saat memuat model atau tokenizer dari path {model_path}: {str(e)}")
        return None, None

@st.cache_resource
def load_preprocessing_assets():
    """Memuat stemmer Sastrawi dan kamus slang."""
    try:
        slang_path = 'modelskripsi/slang.csv'
        if not os.path.exists(slang_path):
            st.error(f"Error: File kamus slang tidak ditemukan di '{slang_path}'.")
            return None, None
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        slang_df = pd.read_csv(slang_path)
        slang_dict = dict(zip(slang_df['slang'], slang_df['formal']))
        return stemmer, slang_dict
    except Exception as e:
        st.error(f"Error fatal saat memuat stemmer atau kamus slang: {str(e)}")
        return None, None

# --- 3. FUNGSI PREPROCESSING & PREDIKSI ---
def preprocess_text(text, _stemmer, _slang_dict):
    """Membersihkan dan menormalisasi teks."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    normalized_words = [_slang_dict.get(word, word) for word in words]
    text = ' '.join(normalized_words)
    text = _stemmer.stem(text)
    return text

def predict_sentiment(text, _model, _tokenizer, _stemmer, _slang_dict, max_length=100):
    """Memprediksi sentimen untuk satu teks tunggal."""
    cleaned_text = preprocess_text(text, _stemmer, _slang_dict)
    sequences = _tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    prediction = _model.predict(padded)
    return prediction[0]

# --- 4. FUNGSI-FUNGSI UNTUK PLOTTING ---
def create_confusion_matrix(y_true, y_pred, size=600):
    """Membuat confusion matrix dengan ukuran tetap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    labels = ['Negatif', 'Netral', 'Positif']
    fig = px.imshow(cm, text_auto=True, aspect="auto", color_continuous_scale='Blues',
                      title="<b>Confusion Matrix</b>", labels=dict(x="Predicted Label", y="True Label"),
                      height=size, width=size) # Ukuran plot tetap
    fig.update_layout(
        title_x=0.5,
        xaxis_title="<b>Predicted Label</b>", yaxis_title="<b>True Label</b>",
        xaxis=dict(tickmode='array', tickvals=[0, 1, 2], ticktext=labels),
        yaxis=dict(tickmode='array', tickvals=[0, 1, 2], ticktext=labels)
    )
    return fig

def create_metrics_plot(precision, recall, f1_score):
    labels = ['Negatif', 'Netral', 'Positif']
    fig = go.Figure(data=[
        go.Bar(name='Precision', x=labels, y=precision, marker_color='#636EFA'),
        go.Bar(name='Recall', x=labels, y=recall, marker_color='#EF553B'),
        go.Bar(name='F1-Score', x=labels, y=f1_score, marker_color='#00CC96')
    ])
    fig.update_layout(
        title='<b>Metrics per Class</b>', title_x=0.5,
        xaxis_title='<b>Classes</b>', yaxis_title='<b>Score</b>',
        barmode='group', yaxis=dict(range=[0, 1.05]),
        legend_title_text='Metrics'
    )
    return fig

def create_probability_chart(probabilities):
    labels = ['Negatif', 'Netral', 'Positif']
    colors = ['#EF553B', '#FECB52', '#00CC96']
    fig = go.Figure(data=[go.Bar(
        x=labels, y=probabilities, marker_color=colors,
        text=[f'{p:.3f}' for p in probabilities], textposition='auto',
        textfont=dict(color='white', size=14, family="Arial, sans-serif")
    )])
    fig.update_layout(
        title='<b>Probabilitas Prediksi Sentimen</b>', title_x=0.5,
        xaxis_title='<b>Kelas Sentimen</b>',
        yaxis_title='<b>Probabilitas</b>', yaxis=dict(range=[0, 1])
    )
    return fig

# --- FUNGSI EVALUASI I ---
def run_evaluation_and_analysis(model, tokenizer, test_data_path, model_name):
    """
    Fungsi terpusat untuk menjalankan evaluasi model dan menampilkan visualisasi.
    """
    # Bagian Evaluasi
    with st.expander(f"📊 Evaluasi Performa Model {model_name} pada Data Uji", expanded=True):
        st.button(f"Jalankan Evaluasi", type="primary", key=f"eval_btn_{model_name}")

        # Kode evaluasi tetap di sini, akan dijalankan setelah tombol diklik
        if st.session_state.get(f"eval_btn_{model_name}"): # Memeriksa status tombol
            test_data = pd.DataFrame()
            try:
                test_data = pd.read_csv(test_data_path)
                label_map = {'negatif': 0, 'netral': 1, 'positif': 2}
                test_data['label_numeric'] = test_data['label'].str.lower().map(label_map)
                test_data.dropna(subset=['label_numeric', 'cleaned_text'], inplace=True)
                test_data['label_numeric'] = test_data['label_numeric'].astype(int)
            except FileNotFoundError:
                st.error(f"File data uji tidak ditemukan di `{test_data_path}`.")
                return
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memuat data uji: {e}")
                return
            
            if not test_data.empty:
                with st.spinner("Mengevaluasi model... Ini mungkin memakan waktu beberapa saat."):
                    texts_for_tokenization = test_data['cleaned_text'].astype(str)
                    true_labels = test_data['label_numeric'].values
                    sequences = tokenizer.texts_to_sequences(texts_for_tokenization)
                    padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
                    
                    all_probabilities = model.predict(padded, batch_size=32)
                    predictions = np.argmax(all_probabilities, axis=1)
                    
                    st.success(f"Evaluasi Selesai! Ditemukan {len(test_data)} baris data uji yang valid.")
                    st.divider()

                    st.subheader("Ringkasan Metrik")
                    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average=None, labels=[0, 1, 2], zero_division=0)
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy_score(true_labels, predictions):.4f}")
                    col2.metric("Avg Precision", f"{np.mean(precision):.4f}")
                    col3.metric("Avg Recall", f"{np.mean(recall):.4f}")
                    col4.metric("Avg F1-Score", f"{np.mean(f1):.4f}")
                    st.divider()

                    # Laporan Klasifikasi
                    st.subheader("Laporan Klasifikasi")
                    report = classification_report(true_labels, predictions, target_names=['Negatif', 'Netral', 'Positif'], output_dict=True, labels=[0, 1, 2])
                    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=False, width=650, height=250)

                    # Confusion Matrix di bawah Laporan Klasifikasi
                    st.subheader("Confusion Matrix")
                    st.plotly_chart(create_confusion_matrix(true_labels, predictions, size=600), use_container_width=False)
                    
                    st.divider()
                    st.subheader("Visualisasi Metrik per Kelas")
                    st.plotly_chart(create_metrics_plot(precision, recall, f1), use_container_width=True)

    # --- Bagian Analisis Detail Model telah dihapus ---


# --- 5. MAIN APPLICATION LOGIC ---

# Konfigurasi model dan path (kunci 'history' dihapus)
model_options = {
    'BiLSTM': {
        'model': 'modelskripsi/eksperimen1b.h5',
        'tokenizer': 'modelskripsi/tokenizer1.pickle',
        'test_data': 'modelskripsi/testdata1.csv'
    },
    'BiLSTM + FastText': {
        'model': 'modelskripsi/eksperimen1c.h5',
        'tokenizer': 'modelskripsi/tokenizer2.pkl',
        'test_data': 'modelskripsi/testdata2.csv'
    }
}

# Load assets
stemmer, slang_dict = load_preprocessing_assets()
if not all([stemmer, slang_dict]):
    st.warning("Aplikasi tidak dapat berjalan karena aset penting gagal dimuat.")
    st.stop()

# --- Sidebar untuk pemilihan model ---
st.sidebar.title("Pilih Model")
selected_model_name = st.sidebar.radio(
    "Pilih jenis model untuk analisis:",
    list(model_options.keys())
)
st.sidebar.divider()

# Muat model yang dipilih
model_paths = model_options[selected_model_name]
model, tokenizer = load_assets(model_paths['model'], model_paths['tokenizer'])

# Judul Utama
st.markdown('<h3 class="main-header">ANALISIS SENTIMEN KOMENTAR PENGGUNA TERHADAP LAYANAN PAYLATER DI PLATFORM X MENGGUNAKAN KOMBINASI BI-DIRECTIONAL LONG SHORT-TERM MEMORY (BILSTM) DAN FASTTEXT</h3>', unsafe_allow_html=True)

# --- Area Input File (Sesuai rancangan) ---
st.subheader("Input File CSV")
uploaded_file = st.file_uploader("Upload File CSV Anda:", type=['csv'], help="Unggah file CSV untuk memuat dan menganalisis data.")
if uploaded_file is not None:
    st.success("File berhasil diunggah!")
    try:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Gagal memuat file: {e}")
        
st.divider()

# --- Area Input Teks dan Prediksi (Sesuai rancangan) ---
st.subheader("Analisis Sentimen Teks Individual")
# Menggunakan st.form untuk mengelompokkan input dan tombol
with st.form("sentiment_form"):
    user_input = st.text_area(
        "Masukkan teks ulasan untuk dianalisis:",
        placeholder="Contoh: Paylater sangat membantu saya dalam mengatur keuangan, transaksi jadi lebih mudah!",
        height=150,
        label_visibility="collapsed"
    )
    submit_button = st.form_submit_button("Analisis", type="primary")

if submit_button:
    if user_input.strip() and model and tokenizer:
        with st.spinner("Menganalisis sentimen..."):
            probabilities = predict_sentiment(user_input, model, tokenizer, stemmer, slang_dict)
            predicted_class = np.argmax(probabilities)
            class_names = ['Negatif', 'Netral', 'Positif']
            emojis = ['😞', '😐', '😊']
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2>Hasil Analisis dan Confidence Score</h2>
                <h3>Sentimen: {class_names[predicted_class]} {emojis[predicted_class]}</h3>
                <h3>Probabilitas: {probabilities[predicted_class]:.3f}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            fig_prob = create_probability_chart(probabilities)
            st.plotly_chart(fig_prob, use_container_width=True)
    elif not user_input.strip():
        st.warning("Harap masukkan teks ulasan untuk dianalisis.")
    else:
        st.warning("Model atau aset lainnya gagal dimuat. Tidak dapat melakukan prediksi.")

st.divider()

# --- Area Evaluasi dan Analisis Model (Sesuai rancangan) ---
run_evaluation_and_analysis(
    model, 
    tokenizer, 
    model_paths['test_data'], 
    selected_model_name
)

st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚀 Aplikasi Analisis Sentimen | Dibangun dengan Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)