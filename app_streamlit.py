"""
XGBoost Heart Disease Prediction - Streamlit App
Tugas Akhir 14 - Praktikum Machine Learning Semester 5

App ini memprediksi risiko penyakit jantung menggunakan model XGBoost
yang sudah dilatih dengan data dari UCI Heart Disease Dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM STYLING
# =====================================================

st.markdown("""
<style>
    /* Main background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }
    
    /* Remove padding top */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Card styling */
    [data-testid="stMetricLabel"] {
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 18px;
        font-weight: 700;
        border-radius: 15px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Input fields */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 10px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
        border-radius: 10px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL & METADATA
# =====================================================

@st.cache_resource
def load_model_and_metadata():
    """Load model and metadata"""
    try:
        model = joblib.load('xgboost_heart_disease_model.joblib')
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return model, metadata
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def get_risk_interpretation(probability, optimal_threshold=0.5):
    """Get risk level interpretation"""
    if probability >= optimal_threshold:
        return {
            'status': 'BERISIKO TINGGI',
            'color': '#fa709a',
            'message': 'Hasil menunjukkan TINGGI RISIKO penyakit jantung. Disarankan untuk berkonsultasi dengan dokter.',
            'emoji': '⚠️'
        }
    else:
        return {
            'status': 'RENDAH RISIKO',
            'color': '#43e97b',
            'message': 'Hasil menunjukkan RENDAH RISIKO penyakit jantung. Tetap jaga kesehatan dengan pola hidup sehat.',
            'emoji': '✅'
        }

def prepare_features(data_dict):
    """Prepare features array"""
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    return np.array([data_dict[f] for f in features]).reshape(1, -1)

# =====================================================
# LOAD MODEL
# =====================================================

model, metadata = load_model_and_metadata()

if model is None or metadata is None:
    st.error("❌ Gagal memuat model. Pastikan file model ada di direktori.")
    st.stop()

# Extract optimal threshold
optimal_threshold = metadata.get('hyperparameters', {}).get('optimal_threshold', 0.5)
if not (0.3 <= optimal_threshold <= 0.7):
    optimal_threshold = 0.5

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #667eea;'>❤️ Heart Disease<br>Prediction</h1>
        <p style='font-size: 14px; color: #666; font-weight: 600;'>Powered by XGBoost ML Model</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Info
    st.markdown("### 📊 Model Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #43e97b22 0%, #43e97b08 100%); 
        padding: 15px; border-radius: 10px; text-align: center;'>
            <div style='font-size: 0.8rem; color: #666; margin-bottom: 5px;'>Accuracy</div>
            <div style='font-size: 1.5rem; font-weight: 700; color: #43e97b;'>{metadata.get('accuracy', 0):.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea22 0%, #667eea08 100%); 
        padding: 15px; border-radius: 10px; text-align: center;'>
            <div style='font-size: 0.8rem; color: #666; margin-bottom: 5px;'>ROC-AUC</div>
            <div style='font-size: 1.5rem; font-weight: 700; color: #667eea;'>{metadata.get('roc_auc', 0):.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #fa709a22 0%, #fa709a08 100%); 
    padding: 15px; border-radius: 10px; text-align: center;'>
        <div style='font-size: 0.8rem; color: #666; margin-bottom: 5px;'>Optimal Threshold</div>
        <div style='font-size: 1.8rem; font-weight: 700; color: #fa709a;'>{optimal_threshold:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset Info
    st.markdown("### 📈 Dataset Info")
    class_dist = metadata.get('class_distribution', {})
    
    st.markdown(f"""
    <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #667eea;'>
        <div style='margin-bottom: 8px;'><strong>Total Samples:</strong> {metadata.get('dataset_size', 'N/A')}</div>
        <div style='margin-bottom: 8px;'><strong>Healthy (Class 0):</strong> {class_dist.get('class_0_healthy', 'N/A')}</div>
        <div><strong>Disease (Class 1):</strong> {class_dist.get('class_1_disease', 'N/A')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Disclaimer
    st.markdown("""
    <div style='background: linear-gradient(135deg, #feca5722 0%, #feca5708 100%); 
    padding: 15px; border-radius: 10px; border-left: 4px solid #feca57;'>
        <div style='font-weight: 700; margin-bottom: 8px; color: #d68910;'>⚠️ DISCLAIMER</div>
        <div style='font-size: 0.85rem; color: #666; line-height: 1.5;'>
        Aplikasi ini hanya untuk tujuan edukasi. Hasil prediksi bukan pengganti diagnosa medis profesional.
        <br><br>
        <strong>Selalu konsultasikan dengan dokter!</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# MAIN PAGE - HEADER
# =====================================================

st.markdown("""
<div style='text-align: center; padding: 30px 20px; 
background: rgba(255, 255, 255, 0.15); 
backdrop-filter: blur(10px); 
border-radius: 20px; 
margin-bottom: 30px;
box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);'>
    <h1 style='color: white; font-size: 2.8rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>
        🏥 Prediksi Risiko Penyakit Jantung
    </h1>
    <p style='font-size: 1.2rem; color: rgba(255,255,255,0.95); font-weight: 500;'>
        Masukkan data kesehatan Anda untuk mendapatkan prediksi menggunakan AI
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# INPUT FORM
# =====================================================

st.markdown("""
<div style='background: white; padding: 30px; border-radius: 20px; 
box-shadow: 0 10px 40px rgba(0,0,0,0.1); margin-bottom: 30px;'>
""", unsafe_allow_html=True)

st.markdown("### 📋 Data Kesehatan Pasien")
st.markdown("*Lengkapi semua field berikut dengan data yang akurat*")
st.markdown("<br>", unsafe_allow_html=True)

# Create form columns
col1, col2, col3 = st.columns(3)

patient_data = {}

with col1:
    st.markdown("#### 👤 Informasi Dasar")
    patient_data['age'] = st.number_input(
        "Usia (tahun)",
        min_value=1, max_value=120, value=55,
        help="Usia pasien dalam tahun"
    )
    
    patient_data['sex'] = st.selectbox(
        "Jenis Kelamin",
        options=[0, 1],
        format_func=lambda x: ['👩 Perempuan', '👨 Laki-laki'][x],
        help="Jenis kelamin pasien"
    )
    
    patient_data['cp'] = st.selectbox(
        "Tipe Nyeri Dada",
        options=[0, 1, 2, 3],
        format_func=lambda x: ['Tidak Ada', 'Typical Angina', 'Atypical Angina', 'Non-anginal'][x],
        help="Jenis nyeri dada yang dialami"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 💗 Aktivitas Jantung")
    
    patient_data['thalach'] = st.number_input(
        "Detak Jantung Maksimum (bpm)",
        min_value=40, max_value=250, value=150,
        help="Detak jantung maksimum yang dicapai"
    )
    
    patient_data['exang'] = st.selectbox(
        "Angina Saat Olahraga",
        options=[0, 1],
        format_func=lambda x: ['❌ Tidak', '✅ Ya'][x],
        help="Angina yang dipicu oleh aktivitas fisik"
    )

with col2:
    st.markdown("#### 🩸 Pemeriksaan Laboratorium")
    
    patient_data['trestbps'] = st.number_input(
        "Tekanan Darah (mmHg)",
        min_value=50, max_value=250, value=120,
        help="Tekanan darah saat istirahat"
    )
    
    patient_data['chol'] = st.number_input(
        "Kolesterol (mg/dl)",
        min_value=100, max_value=600, value=200,
        help="Kadar kolesterol serum"
    )
    
    patient_data['fbs'] = st.selectbox(
        "Gula Darah Puasa",
        options=[0, 1],
        format_func=lambda x: ['≤ 120 mg/dl', '> 120 mg/dl'][x],
        help="Kadar gula darah saat puasa"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📊 Hasil EKG")
    
    patient_data['restecg'] = st.selectbox(
        "EKG Istirahat",
        options=[0, 1, 2],
        format_func=lambda x: ['Normal', 'ST-T Abnormal', 'LVH'][x],
        help="Hasil elektrokardiogram"
    )
    
    patient_data['oldpeak'] = st.slider(
        "Depresi ST",
        min_value=0.0, max_value=7.0, value=1.0, step=0.1,
        help="ST depression induced by exercise"
    )

with col3:
    st.markdown("#### 🫀 Kondisi Jantung")
    
    patient_data['slope'] = st.selectbox(
        "Slope ST Segment",
        options=[0, 1, 2],
        format_func=lambda x: ['Downsloping ↘', 'Flat →', 'Upsloping ↗'][x],
        help="Kemiringan segmen ST"
    )
    
    patient_data['ca'] = st.selectbox(
        "Pembuluh Darah Tersumbat",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: f'{x} pembuluh' if x > 0 else 'Tidak ada',
        help="Jumlah pembuluh darah mayor yang tersumbat"
    )
    
    patient_data['thal'] = st.selectbox(
        "Thalassemia",
        options=[0, 1, 2, 3],
        format_func=lambda x: ['Normal', 'Fixed Defect', 'Reversible Defect', 'Other'][x] if x != 0 else 'Normal',
        help="Status thalassemia"
    )

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PREDICTION BUTTON
# =====================================================

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 PREDIKSI RISIKO SEKARANG"):
    
    with st.spinner('⏳ Menganalisis data kesehatan Anda...'):
        # Prepare features
        X_input = prepare_features(patient_data)
        
        # Get prediction
        prediction_proba = model.predict_proba(X_input)[0][1]
        prediction = 1 if prediction_proba >= optimal_threshold else 0
        
        # Get interpretation
        interpretation = get_risk_interpretation(prediction_proba, optimal_threshold)
    
    # Display result
    st.markdown("---")
    
    # Main result - Big Badge
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, {interpretation["color"]}22 0%, {interpretation["color"]}08 100%);
        border: 4px solid {interpretation["color"]};
        border-radius: 25px;
        padding: 50px 30px;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 15px 50px rgba(0,0,0,0.15);
    '>
        <div style='font-size: 80px; margin-bottom: 15px;'>{interpretation["emoji"]}</div>
        <h1 style='margin: 15px 0; color: {interpretation["color"]}; font-size: 3rem; font-weight: 800; text-transform: uppercase; letter-spacing: 2px;'>
            {interpretation["status"]}
        </h1>
        <p style='font-size: 1.3rem; color: #2d3436; margin-top: 20px; line-height: 1.6;'>
            {interpretation["message"]}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Metrics Row dengan card yang lebih besar
    st.markdown("### 📊 Detail Probabilitas")
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    
    with col_metric1:
        st.markdown(f"""
        <div style='background: white; padding: 30px 20px; border-radius: 20px; 
        box-shadow: 0 8px 25px rgba(0,0,0,0.12); text-align: center; border-top: 5px solid {interpretation["color"]};'>
            <div style='color: #666; font-size: 1rem; margin-bottom: 10px; font-weight: 600;'>
                🔴 Probabilitas Penyakit
            </div>
            <div style='font-size: 3.5rem; font-weight: 800; color: {interpretation["color"]};'>
                {prediction_proba*100:.1f}%
            </div>
            <div style='color: #999; font-size: 0.9rem; margin-top: 10px;'>
                Risk Score
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_metric2:
        st.markdown(f"""
        <div style='background: white; padding: 30px 20px; border-radius: 20px; 
        box-shadow: 0 8px 25px rgba(0,0,0,0.12); text-align: center; border-top: 5px solid #43e97b;'>
            <div style='color: #666; font-size: 1rem; margin-bottom: 10px; font-weight: 600;'>
                🟢 Probabilitas Sehat
            </div>
            <div style='font-size: 3.5rem; font-weight: 800; color: #43e97b;'>
                {(1-prediction_proba)*100:.1f}%
            </div>
            <div style='color: #999; font-size: 0.9rem; margin-top: 10px;'>
                Health Score
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_metric3:
        st.markdown(f"""
        <div style='background: white; padding: 30px 20px; border-radius: 20px; 
        box-shadow: 0 8px 25px rgba(0,0,0,0.12); text-align: center; border-top: 5px solid #667eea;'>
            <div style='color: #666; font-size: 1rem; margin-bottom: 10px; font-weight: 600;'>
                ⚖️ Model Threshold
            </div>
            <div style='font-size: 3.5rem; font-weight: 800; color: #667eea;'>
                {optimal_threshold*100:.0f}%
            </div>
            <div style='color: #999; font-size: 0.9rem; margin-top: 10px;'>
                Decision Boundary
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Probability visualization dengan style lebih bagus
    st.markdown("### 📈 Visualisasi Distribusi Probabilitas")
    
    col_viz1, col_viz2 = st.columns([1, 1])
    
    with col_viz1:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        """, unsafe_allow_html=True)
        st.markdown(f"**🔴 Risiko Penyakit: {prediction_proba:.1%}**")
        st.progress(float(prediction_proba))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_viz2:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        """, unsafe_allow_html=True)
        st.markdown(f"**🟢 Probabilitas Sehat: {(1-prediction_proba):.1%}**")
        st.progress(float(1 - prediction_proba))
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Divider
    st.markdown("---")
    
    # Model details dengan card
    st.markdown("### ℹ️ Informasi Model & Training")
    
    col_detail1, col_detail2, col_detail3, col_detail4 = st.columns(4)
    
    with col_detail1:
        st.markdown(f"""
        <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center;'>
            <div style='font-size: 0.85rem; color: #666; margin-bottom: 5px;'>Threshold</div>
            <div style='font-size: 1.5rem; font-weight: 700; color: #667eea;'>{optimal_threshold:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_detail2:
        st.markdown(f"""
        <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center;'>
            <div style='font-size: 0.85rem; color: #666; margin-bottom: 5px;'>Accuracy</div>
            <div style='font-size: 1.5rem; font-weight: 700; color: #43e97b;'>{metadata.get('accuracy', 0):.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_detail3:
        st.markdown(f"""
        <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center;'>
            <div style='font-size: 0.85rem; color: #666; margin-bottom: 5px;'>Model</div>
            <div style='font-size: 1.2rem; font-weight: 700; color: #667eea;'>XGBoost</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_detail4:
        training_date = metadata.get('training_date', 'N/A')
        if training_date != 'N/A':
            training_date = training_date.split()[0]  # Only date
        st.markdown(f"""
        <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center;'>
            <div style='font-size: 0.85rem; color: #666; margin-bottom: 5px;'>Trained</div>
            <div style='font-size: 0.95rem; font-weight: 700; color: #666;'>{training_date}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional info dengan styling lebih baik
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
    padding: 25px; border-radius: 15px; border-left: 5px solid #667eea;'>
        <h4 style='color: #667eea; margin-bottom: 15px;'>🔍 Cara Membaca Hasil</h4>
        <ul style='line-height: 2; color: #2d3436;'>
            <li><strong>Probabilitas:</strong> Nilai 0-100% yang menunjukkan kemungkinan memiliki penyakit jantung berdasarkan data input</li>
            <li><strong>Threshold ({optimal_threshold:.0%}):</strong> Batas keputusan model. Jika probabilitas ≥ threshold → Berisiko Tinggi</li>
            <li><strong>Prediksi:</strong> Hasil akhir klasifikasi berdasarkan perbandingan probabilitas dengan threshold</li>
            <li><strong>Confidence:</strong> Semakin jauh probabilitas dari threshold (50%), semakin yakin prediksi model</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# INFO TABS
# =====================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📖 Tentang Model",
    "🎯 Penjelasan Fitur",
    "📊 Feature Importance",
    "💡 Tips Kesehatan"
])

with tab1:
    st.markdown("""
    <div style='background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);'>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🤖 Tentang Model XGBoost
    
    Model prediksi ini menggunakan **XGBoost (Extreme Gradient Boosting)**, salah satu algoritma machine learning 
    terbaik untuk klasifikasi dengan dataset tabular.
    
    #### ✨ Keunggulan XGBoost:
    - **Akurasi Tinggi**: Mampu menangkap pola kompleks dalam data medis
    - **Robust**: Tahan terhadap data yang tidak seimbang (imbalanced)
    - **Interpretable**: Feature importance dapat dijelaskan untuk interpretasi medis
    - **Efisien**: Proses prediksi cepat, cocok untuk deployment real-time
    
    ---
    
    #### 📊 Dataset Training:
    - **Sumber**: UCI Heart Disease Dataset (Cleveland)
    - **Total Sampel**: 1,026 pasien
    - **Fitur**: 13 indikator kesehatan klinis
    - **Split**: 80% Training, 20% Testing (Stratified)
    
    ---
    
    #### ⚙️ Preprocessing & Tuning:
    1. **Stratified Sampling**: Mempertahankan proporsi kelas pada train-test split
    2. **Imbalance Handling**: Menggunakan `scale_pos_weight` untuk menangani ketidakseimbangan kelas
    3. **Threshold Optimization**: Mencari threshold optimal berdasarkan F1-Score (bukan default 0.5)
    4. **Hyperparameter Tuning**: Grid search untuk max_depth dan learning_rate
    
    ---
    
    #### 🎯 Performa Model:
    """)
    
    col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
    with col_perf1:
        st.metric("Accuracy", f"{metadata.get('accuracy', 0):.2%}")
    with col_perf2:
        st.metric("Precision", f"{metadata.get('precision', 0):.2%}")
    with col_perf3:
        st.metric("Recall", f"{metadata.get('recall', 0):.2%}")
    with col_perf4:
        st.metric("F1-Score", f"{metadata.get('f1_score', 0):.2%}")
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div style='background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);'>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📝 Penjelasan Fitur Input")
    
    # Create DataFrame for better display
    features_df = pd.DataFrame({
        'Fitur': [
            '👤 Usia', '⚧️ Jenis Kelamin', '💔 Tipe Nyeri Dada', '🩸 Tekanan Darah',
            '🧬 Kolesterol', '🍬 Gula Darah', '📊 EKG Istirahat', '💗 Detak Jantung Max',
            '🏃 Angina Olahraga', '📉 Depresi ST', '⛰️ Slope ST', '🫀 Pembuluh Darah', '🩸 Thalassemia'
        ],
        'Deskripsi': [
            'Usia pasien dalam tahun',
            '0 = Perempuan, 1 = Laki-laki',
            '0 = Asymptomatic, 1 = Typical Angina, 2 = Atypical, 3 = Non-anginal',
            'Tekanan darah istirahat (mmHg)',
            'Kadar kolesterol serum (mg/dl)',
            '0 = ≤120 mg/dl, 1 = >120 mg/dl',
            '0 = Normal, 1 = ST-T Abnormal, 2 = LVH',
            'Detak jantung maksimum yang dicapai',
            '0 = Tidak ada, 1 = Ada angina saat olahraga',
            'ST depression induced by exercise (0-7)',
            '0 = Downsloping, 1 = Flat, 2 = Upsloping',
            'Jumlah pembuluh darah mayor tersumbat (0-4)',
            '0 = Normal, 1-3 = Tipe defect'
        ],
        'Range Normal': [
            '20-80 tahun',
            '-',
            'Tipe 0 (Tidak ada)',
            '90-120 mmHg',
            '<200 mg/dl',
            '≤120 mg/dl',
            'Normal',
            '60-100 bpm (istirahat)',
            'Tidak ada',
            '0-1',
            'Upsloping',
            '0 pembuluh',
            'Normal'
        ]
    })
    
    st.dataframe(features_df, use_container_width=True, hide_index=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div style='background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);'>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Feature Importance")
    st.markdown("*Fitur-fitur yang paling berpengaruh dalam prediksi model*")
    
    feature_importance = metadata.get('feature_importance', {})
    if feature_importance:
        importance_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        # Bar chart with better styling
        st.bar_chart(importance_df.set_index('Feature')['Importance'], height=400)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Top 5 features
        st.markdown("#### 🏆 Top 5 Fitur Paling Penting:")
        for idx, row in importance_df.head(5).iterrows():
            st.markdown(f"""
            <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #667eea;'>
                <strong>{row['Feature']}</strong>: {row['Importance']:.4f}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📌 Feature importance data tidak tersedia dalam metadata model")
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown("""
    <div style='background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);'>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 💚 Tips Menjaga Kesehatan Jantung
    
    #### 1️⃣ Olahraga Teratur
    - **Minimal** 150 menit per minggu (30 menit x 5 hari)
    - Pilih aktivitas yang Anda sukai: jalan cepat, jogging, berenang, bersepeda
    - Mulai dengan intensitas ringan, tingkatkan bertahap
    
    #### 2️⃣ Pola Makan Sehat
    - **Perbanyak**: Buah, sayur, biji-bijian utuh, ikan
    - **Kurangi**: Lemak jenuh, kolesterol, garam, gula
    - **Hindari**: Makanan olahan, fast food, gorengan berlebihan
    - **Porsi**: Makan teratur 3x sehari dengan porsi secukupnya
    
    #### 3️⃣ Kelola Stress
    - Teknik relaksasi: Meditasi, yoga, pernapasan dalam
    - Hobi yang menyenangkan
    - Tidur cukup 7-9 jam per hari
    - Manajemen waktu yang baik
    
    #### 4️⃣ Kontrol Berat Badan
    - BMI ideal: 18.5 - 24.9
    - Lingkar pinggang: <90cm (pria), <80cm (wanita)
    - Konsultasi dengan nutrisionis jika perlu
    
    #### 5️⃣ Stop Rokok & Alkohol
    - **Rokok**: Meningkatkan risiko penyakit jantung 2-4x lipat
    - **Alkohol**: Maksimal 1 gelas/hari (wanita), 2 gelas/hari (pria)
    - Cari dukungan profesional untuk berhenti
    
    #### 6️⃣ Pemeriksaan Rutin
    - Check-up kesehatan **minimal 1x per tahun**
    - Monitor tekanan darah dan kolesterol secara berkala
    - Konsultasi dengan dokter jika ada gejala
    
    ---
    
    ### ⚠️ Gejala yang Harus Diwaspadai:
    
    🚨 **SEGERA KE DOKTER** jika mengalami:
    - Nyeri dada atau rasa tertekan
    - Sesak nafas tiba-tiba
    - Denyut jantung tidak teratur atau terlalu cepat
    - Kelelahan ekstrem tanpa sebab
    - Pusing, mual, atau keringat dingin
    - Nyeri yang menjalar ke lengan, rahang, atau punggung
    
    **📞 DARURAT (119/118)** jika gejala berat atau tiba-tiba!
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
<div style='text-align: center; padding: 30px; background: rgba(255,255,255,0.1); 
border-radius: 15px; backdrop-filter: blur(10px);'>
    <p style='font-size: 1.2rem; color: white; font-weight: 600; margin-bottom: 10px;'>
        ❤️ XGBoost Heart Disease Prediction Model
    </p>
    <p style='color: rgba(255,255,255,0.9); font-size: 0.95rem; margin-bottom: 5px;'>
        Tugas Akhir 14 - Praktikum Machine Learning | Semester 5
    </p>
    <p style='color: rgba(255,255,255,0.7); font-size: 0.85rem; margin-top: 15px; font-style: italic;'>
        <strong>Disclaimer:</strong> Aplikasi ini hanya untuk tujuan edukasi.<br>
        Hasil prediksi bukan pengganti diagnosa medis profesional.
    </p>
</div>
""", unsafe_allow_html=True)
