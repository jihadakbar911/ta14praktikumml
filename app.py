"""
Flask Web Application untuk XGBoost Heart Disease Prediction
Tugas Akhir 14 - Praktikum Machine Learning Semester 5

Author: Student
Date: 2026
Description: Web application untuk memprediksi risiko penyakit jantung menggunakan XGBoost model
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pickle
import logging
from datetime import datetime
import os

# =============================================
# FLASK APP INITIALIZATION
# =============================================

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================
# MODEL LOADING
# =============================================

# Global variables untuk model
MODEL = None
MODEL_METADATA = None
MODEL_LOADED = False
OPTIMAL_THRESHOLD = 0.5  # Default fallback (reasonable untuk balanced prediction)

def load_model():
    """Load XGBoost model dan metadata saat startup"""
    global MODEL, MODEL_METADATA, MODEL_LOADED, OPTIMAL_THRESHOLD
    
    try:
        # Path ke model file
        model_path = os.path.join(os.path.dirname(__file__), 'xgboost_heart_disease_model.joblib')
        metadata_path = os.path.join(os.path.dirname(__file__), 'model_metadata.pkl')
        
        # Load model dengan joblib
        if os.path.exists(model_path):
            MODEL = joblib.load(model_path)
            logger.info("✅ Model berhasil di-load dengan joblib")
        else:
            logger.error(f"❌ Model file tidak ditemukan: {model_path}")
            return False
        
        # Load metadata jika ada
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                MODEL_METADATA = pickle.load(f)
            logger.info("✅ Model metadata berhasil di-load")
            
            # Extract optimal threshold dari metadata
            if MODEL_METADATA and 'hyperparameters' in MODEL_METADATA:
                threshold_from_metadata = MODEL_METADATA['hyperparameters'].get('optimal_threshold', None)
                
                # VALIDASI: Threshold harus masuk akal (antara 0.3 - 0.7)
                # Jika di luar range ini, kemungkinan dari model lama atau belum optimal
                if threshold_from_metadata and 0.3 <= threshold_from_metadata <= 0.7:
                    OPTIMAL_THRESHOLD = threshold_from_metadata
                    logger.info(f"🎯 Optimal threshold dari metadata: {OPTIMAL_THRESHOLD:.2f}")
                else:
                    logger.warning(f"⚠️ Threshold dari metadata ({threshold_from_metadata}) tidak masuk akal")
                    logger.warning(f"⚠️ Menggunakan default threshold: {OPTIMAL_THRESHOLD}")
            else:
                logger.warning("⚠️ Optimal threshold tidak ditemukan di metadata, menggunakan default 0.5")
        else:
            logger.warning("⚠️ Model metadata file tidak ditemukan")
            logger.warning(f"⚠️ Menggunakan default threshold: {OPTIMAL_THRESHOLD}")
        
        MODEL_LOADED = True
        return True
    
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False

# =============================================
# HELPER FUNCTIONS
# =============================================

# Nama fitur dalam urutan yang benar
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Validasi range untuk setiap fitur
FEATURE_RANGES = {
    'age': (1, 120),
    'sex': (0, 1),
    'cp': (0, 3),
    'trestbps': (50, 250),
    'chol': (100, 600),
    'fbs': (0, 1),
    'restecg': (0, 2),
    'thalach': (40, 250),
    'exang': (0, 1),
    'oldpeak': (0, 7),
    'slope': (0, 2),
    'ca': (0, 4),
    'thal': (0, 3)
}

# Deskripsi fitur untuk UI
FEATURE_DESCRIPTIONS = {
    'age': 'Usia Pasien (tahun)',
    'sex': 'Jenis Kelamin (0=Wanita, 1=Pria)',
    'cp': 'Tipe Nyeri Dada (0-3)',
    'trestbps': 'Tekanan Darah saat Istirahat (mmHg)',
    'chol': 'Kolesterol Serum (mg/dl)',
    'fbs': 'Gula Darah Puasa > 120 (0=Tidak, 1=Ya)',
    'restecg': 'Hasil EKG saat Istirahat (0-2)',
    'thalach': 'Detak Jantung Maksimum',
    'exang': 'Angina akibat Olahraga (0=Tidak, 1=Ya)',
    'oldpeak': 'Depresi ST akibat Olahraga',
    'slope': 'Kemiringan Segmen ST (0-2)',
    'ca': 'Jumlah Pembuluh Darah (0-4)',
    'thal': 'Thalassemia (0=Normal, 1=Fixed, 2=Reversable, 3=Other)'
}

def validate_input(data):
    """
    Validasi input data dari form
    
    Args:
        data: dictionary dengan feature values
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Cek apakah semua fitur ada
        missing_features = [f for f in FEATURES if f not in data or data[f] == '']
        if missing_features:
            return False, f"Fitur yang hilang: {', '.join(missing_features)}"
        
        # Cek range setiap fitur
        for feature in FEATURES:
            try:
                value = float(data[feature])
                min_val, max_val = FEATURE_RANGES[feature]
                
                if value < min_val or value > max_val:
                    return False, f"{feature}: nilai harus antara {min_val} - {max_val} (input: {value})"
            
            except ValueError:
                return False, f"{feature}: nilai harus berupa angka"
        
        return True, "Valid"
    
    except Exception as e:
        return False, f"Error validasi: {str(e)}"

def prepare_features(data):
    """
    Siapkan features dalam urutan yang benar untuk model
    
    Args:
        data: dictionary dengan feature values
    
    Returns:
        numpy array dengan shape (1, 13)
    """
    features_array = []
    for feature in FEATURES:
        features_array.append(float(data[feature]))
    
    return np.array([features_array])

def interpret_prediction(prediction, probability):
    """
    Interpretasi hasil prediksi
    
    Args:
        prediction: 0 atau 1 (hasil prediksi)
        probability: float antara 0-1 (probabilitas kelas 1)
    
    Returns:
        dictionary dengan interpretation text
    """
    if prediction == 0:
        risk_level = "RENDAH ✅"
        interpretation = "Hasil menunjukkan RENDAH RISIKO penyakit jantung."
        color = "success"
    else:
        risk_level = "TINGGI ⚠️"
        interpretation = "Hasil menunjukkan TINGGI RISIKO penyakit jantung. Disarankan berkonsultasi dengan dokter."
        color = "danger"
    
    confidence = probability if prediction == 1 else (1 - probability)
    
    return {
        'risk_level': risk_level,
        'interpretation': interpretation,
        'confidence': f"{confidence*100:.2f}%",
        'color': color,
        'probability_disease': f"{probability*100:.2f}%",
        'probability_no_disease': f"{(1-probability)*100:.2f}%"
    }

# =============================================
# FLASK ROUTES
# =============================================

@app.route('/')
def home():
    """Home page dengan form input"""
    logger.info("📄 Akses ke home page")
    return render_template('index.html', features=FEATURES, descriptions=FEATURE_DESCRIPTIONS)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk prediksi
    Accept: POST request dengan data dari form atau JSON
    """
    try:
        # Ambil data dari form atau JSON
        if request.form:
            data = request.form.to_dict()
            # Ambil custom threshold jika ada (default 0.5)
            threshold = float(data.get('threshold', 0.5))
        else:
            data = request.get_json()
            threshold = float(data.get('threshold', 0.5))
        
        logger.info(f"📨 Received prediction request with data: {data}")
        logger.info(f"🎯 Using threshold: {threshold}")
        
        # Validasi input
        is_valid, message = validate_input(data)
        if not is_valid:
            logger.warning(f"⚠️ Validasi gagal: {message}")
            return jsonify({
                'success': False,
                'error': message
            }), 400
        
        # Prepare features
        features_array = prepare_features(data)
        
        # Lakukan prediksi
        if not MODEL_LOADED:
            return jsonify({
                'success': False,
                'error': 'Model belum ter-load. Silakan hubungi administrator.'
            }), 500
        
        # Prediksi dengan optimal threshold (dari metadata atau default)
        prediction_proba = MODEL.predict_proba(features_array)[0][1]
        
        # Gunakan optimal threshold dari metadata jika tersedia
        # Jika tidak ada, gunakan custom threshold atau default 0.5
        if 'threshold' in data or 'threshold' in request.form:
            # User provided custom threshold
            threshold = float(data.get('threshold', OPTIMAL_THRESHOLD))
        else:
            # Gunakan optimal threshold dari metadata
            threshold = OPTIMAL_THRESHOLD
        
        prediction = 1 if prediction_proba >= threshold else 0
        
        # Interpretasi hasil
        interpretation = interpret_prediction(prediction, prediction_proba)
        
        # Log prediction
        logger.info(f"✅ Probability: {prediction_proba:.4f}")
        logger.info(f"✅ Threshold: {threshold:.4f}")
        logger.info(f"✅ Prediction: {prediction}")
        
        # Return result dengan diagnostic info
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': float(prediction_proba),
            'threshold_used': float(threshold),
            'optimal_threshold': float(OPTIMAL_THRESHOLD),
            'interpretation': interpretation,
            'diagnostic': {
                'probability_disease': f"{prediction_proba*100:.2f}%",
                'probability_healthy': f"{(1-prediction_proba)*100:.2f}%",
                'threshold': f"{threshold*100:.0f}%",
                'confidence': f"{max(prediction_proba, 1-prediction_proba)*100:.2f}%"
            },
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'OK',
        'model_loaded': MODEL_LOADED,
        'model_name': 'XGBoost Heart Disease Classifier',
        'optimal_threshold': float(OPTIMAL_THRESHOLD),
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/model-info', methods=['GET'])
def model_info():
    """Return model metadata"""
    if MODEL_METADATA:
        return jsonify({
            'success': True,
            'model_info': MODEL_METADATA
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Model metadata tidak tersedia'
        }), 404

@app.route('/features', methods=['GET'])
def get_features():
    """Return informasi tentang semua features"""
    features_info = []
    for feature in FEATURES:
        min_val, max_val = FEATURE_RANGES[feature]
        features_info.append({
            'name': feature,
            'description': FEATURE_DESCRIPTIONS[feature],
            'min': min_val,
            'max': max_val
        })
    
    return jsonify({
        'success': True,
        'features': features_info,
        'total_features': len(features_info)
    })

@app.route('/diagnostic', methods=['GET'])
def diagnostic():
    """Endpoint untuk diagnostic model"""
    if not MODEL_LOADED:
        return jsonify({
            'success': False,
            'error': 'Model belum ter-load'
        }), 500
    
    try:
        # Load dataset untuk cek distribusi
        import pandas as pd
        dataset_path = os.path.join(os.path.dirname(__file__), 'heart.csv')
        
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            target_col = 'target' if 'target' in df.columns else df.columns[-1]
            
            class_0 = (df[target_col] == 0).sum()
            class_1 = (df[target_col] == 1).sum()
            total = len(df)
            
            return jsonify({
                'success': True,
                'dataset_info': {
                    'total_samples': total,
                    'class_0_healthy': int(class_0),
                    'class_1_disease': int(class_1),
                    'ratio': f"{class_1/class_0:.2f}",
                    'percentage_disease': f"{(class_1/total)*100:.2f}%",
                    'percentage_healthy': f"{(class_0/total)*100:.2f}%",
                    'imbalanced': class_0 != class_1
                },
                'model_info': {
                    'model_type': str(type(MODEL).__name__),
                    'n_features': len(FEATURES),
                    'features': FEATURES
                },
                'recommendation': {
                    'suggested_threshold': 0.3 if class_0 > class_1 * 2 else 0.5,
                    'reason': 'Dataset imbalanced, gunakan threshold lebih rendah' if class_0 > class_1 * 2 else 'Dataset balanced'
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Dataset file tidak ditemukan'
            }), 404
    
    except Exception as e:
        logger.error(f"❌ Error diagnostic: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# =============================================
# ERROR HANDLERS
# =============================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 error"""
    return jsonify({
        'success': False,
        'error': 'Endpoint tidak ditemukan'
    }), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 error"""
    logger.error(f"❌ Server error: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# =============================================
# APPLICATION STARTUP
# =============================================

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("🚀 Starting XGBoost Heart Disease Prediction App")
    logger.info("="*60)
    
    # Load model saat startup
    if load_model():
        logger.info("✅ Aplikasi siap!")
        logger.info("📍 Akses: http://localhost:5000")
        logger.info("="*60)
        
        # Run Flask app
        # Development mode
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("❌ Gagal memuat model. Aplikasi tidak dapat dimulai.")
