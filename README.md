# XGBoost Heart Disease Prediction Web Application

Aplikasi web untuk prediksi risiko penyakit jantung menggunakan model XGBoost yang dilatih pada dataset UCI Heart Disease.

## 🏥 Deskripsi Proyek

Tugas Akhir 14 - Praktikum Machine Learning Semester 5. Aplikasi ini menggabungkan model machine learning XGBoost dengan web interface yang user-friendly untuk memberikan prediksi risiko penyakit jantung berdasarkan data medis pasien.

**⚠️ DISCLAIMER:** Aplikasi ini hanya untuk tujuan edukasi dan bukan merupakan pengganti dari diagnosa profesional dokter. Selalu konsultasikan dengan tenaga medis profesional.

## 📋 Fitur Aplikasi

- ✅ Form input interaktif untuk 13 fitur medis
- ✅ Validasi input dengan range checking otomatis
- ✅ Model XGBoost pre-trained dengan performa tinggi
- ✅ Tampilan hasil prediksi dengan probabilitas confidence
- ✅ Responsive design untuk mobile & desktop
- ✅ REST API endpoint untuk integrasi
- ✅ Health check dan model info endpoints
- ✅ Error handling yang komprehensif
- ✅ Professional UI dengan Bootstrap 5

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### 1. Clone / Download Repository

```bash
cd "c:\Tugas Kuliah\SEMESTER 5\Praktikum Machine Learning\Pertemuan 14\TA-14"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Jika menggunakan virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Pastikan Model File Ada

Sebelum menjalankan aplikasi, pastikan file model ada di folder yang sama dengan `app.py`:
- `xgboost_heart_disease_model.joblib` (dari notebook training)
- `model_metadata.pkl` (opsional, untuk informasi model)

### 4. Run Flask Application

```bash
python app.py
```

Output:
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://0.0.0.0:5000
```

### 5. Akses Aplikasi

Buka browser dan akses:
```
http://localhost:5000
```

## 📁 Struktur Folder

```
TA-14/
├── app.py                                    # Main Flask application
├── requirements.txt                          # Python dependencies
├── README.md                                 # File ini
├── xgboost_heart_disease_model.joblib       # Trained model (required)
├── model_metadata.pkl                        # Model metadata (optional)
├── heart_disease.csv                         # Dataset (reference)
├── TA14_XGBoost_Heart_Disease.ipynb         # Training notebook
├── templates/
│   └── index.html                           # Main page dengan form
└── static/
    └── css/
        └── style.css                        # Custom styling
```

## 📊 Input Features

Aplikasi memerlukan 13 fitur medis pasien:

| No | Fitur | Range | Deskripsi |
|----|-------|-------|-----------|
| 1 | age | 1-120 | Usia pasien (tahun) |
| 2 | sex | 0-1 | Jenis kelamin (0=Wanita, 1=Pria) |
| 3 | cp | 0-3 | Tipe nyeri dada |
| 4 | trestbps | 50-250 | Tekanan darah istirahat (mmHg) |
| 5 | chol | 100-600 | Kolesterol serum (mg/dl) |
| 6 | fbs | 0-1 | Gula darah puasa > 120 (0=Tidak, 1=Ya) |
| 7 | restecg | 0-2 | Hasil EKG istirahat |
| 8 | thalach | 40-250 | Detak jantung maksimum |
| 9 | exang | 0-1 | Angina akibat olahraga (0=Tidak, 1=Ya) |
| 10 | oldpeak | 0-7 | Depresi ST akibat olahraga |
| 11 | slope | 0-2 | Kemiringan segmen ST |
| 12 | ca | 0-4 | Jumlah pembuluh darah besar |
| 13 | thal | 0-3 | Thalassemia (0=Normal, 1=Fixed, 2=Reversable, 3=Other) |

## 🔗 API Endpoints

### 1. **GET /home**
Halaman utama dengan form input

```bash
curl http://localhost:5000/
```

### 2. **POST /predict**
Endpoint untuk prediksi. Accept form data atau JSON.

**Form Data Example:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "age=55" \
  -F "sex=1" \
  -F "cp=2" \
  -F "trestbps=140" \
  -F "chol=250" \
  -F "fbs=0" \
  -F "restecg=1" \
  -F "thalach=150" \
  -F "exang=1" \
  -F "oldpeak=2.5" \
  -F "slope=1" \
  -F "ca=1" \
  -F "thal=3"
```

**JSON Example:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 1,
    "oldpeak": 2.5,
    "slope": 1,
    "ca": 1,
    "thal": 3
  }'
```

**Response Success (200):**
```json
{
  "success": true,
  "prediction": 1,
  "probability": 0.8234,
  "interpretation": {
    "risk_level": "TINGGI ⚠️",
    "interpretation": "Hasil menunjukkan TINGGI RISIKO penyakit jantung...",
    "confidence": "82.34%",
    "color": "danger",
    "probability_disease": "82.34%",
    "probability_no_disease": "17.66%"
  },
  "timestamp": "2026-01-06T10:30:45.123456"
}
```

**Response Error (400):**
```json
{
  "success": false,
  "error": "age: nilai harus antara 1 - 120 (input: 150)"
}
```

### 3. **GET /health**
Health check endpoint

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "OK",
  "model_loaded": true,
  "model_name": "XGBoost Heart Disease Classifier",
  "timestamp": "2026-01-06T10:30:45.123456"
}
```

### 4. **GET /model-info**
Informasi model dan metadata

```bash
curl http://localhost:5000/model-info
```

**Response:**
```json
{
  "success": true,
  "model_info": {
    "model_name": "XGBoost Heart Disease Classifier",
    "training_date": "2026-01-06 10:15:30",
    "dataset_size": 1026,
    "features": ["age", "sex", "cp", ...],
    "accuracy": 0.8543,
    "roc_auc": 0.9123,
    "hyperparameters": {...}
  }
}
```

### 5. **GET /features**
List semua features dengan deskripsi

```bash
curl http://localhost:5000/features
```

## 🌐 Production Deployment

### Option 1: Render (Recommended - Free)

1. **Push ke GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Connect ke Render:**
   - Buka [render.com](https://render.com)
   - Create new Web Service
   - Connect GitHub repository
   - Configure:
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `gunicorn app:app`
     - Environment: Python 3.11

3. **Deploy:**
   - Klik Deploy
   - Tunggu hingga selesai (~5 menit)
   - Aplikasi akan accessible di URL yang diberikan Render

### Option 2: Railway

1. Connect GitHub repository
2. Railway auto-detects Flask app
3. Set `requirements.txt` sebagai dependencies
4. Deploy

### Option 3: PythonAnywhere

1. Upload files ke akun PythonAnywhere
2. Setup virtual environment
3. Configure WSGI file
4. Reload web app

### Option 4: Heroku (Requires Credit Card)

1. Create `Procfile`:
   ```
   web: gunicorn app:app
   ```

2. Deploy:
   ```bash
   heroku login
   heroku create app-name
   git push heroku main
   ```

## 🔧 Configuration

### Environment Variables (Optional)

Buat file `.env` untuk konfigurasi:
```
FLASK_ENV=production
FLASK_DEBUG=0
MODEL_PATH=xgboost_heart_disease_model.joblib
```

Load di `app.py`:
```python
from dotenv import load_dotenv
load_dotenv()
```

## 📈 Model Performance

- **Accuracy:** ~85-90%
- **ROC-AUC:** ~0.91
- **Training Dataset:** 1,026 sampel
- **Features:** 13 medis features
- **Algorithm:** XGBoost Classifier
- **Hyperparameters:** Tuned dengan GridSearch

## 🧪 Testing

### Local Testing

```bash
# Test home page
curl http://localhost:5000/

# Test health check
curl http://localhost:5000/health

# Test prediction
curl -X POST http://localhost:5000/predict \
  -F "age=55" -F "sex=1" -F "cp=2" ... (dst)
```

### Browser Testing

1. Buka `http://localhost:5000` di browser
2. Isi semua field dengan nilai yang sesuai
3. Klik "Prediksi Sekarang"
4. Lihat hasil prediksi dengan visualisasi

## 🐛 Troubleshooting

### Error: "Model file tidak ditemukan"
- Pastikan `xgboost_heart_disease_model.joblib` ada di folder yang sama dengan `app.py`
- Periksa path di `app.py`: `model_path = os.path.join(os.path.dirname(__file__), 'xgboost_heart_disease_model.joblib')`

### Error: "Module not found"
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Port 5000 sudah digunakan
```bash
# Run di port berbeda
python app.py
# Kemudian akses: http://localhost:5000
# Atau ubah di app.py: app.run(port=8000)
```

### Input Validation Error
- Pastikan semua nilai dalam range yang valid
- Check dokumentasi di field deskripsi setiap input

## 📝 Development Notes

### Menambah Fitur Baru

1. Update `FEATURES` list di `app.py`
2. Tambah validasi range di `FEATURE_RANGES`
3. Tambah deskripsi di `FEATURE_DESCRIPTIONS`
4. Update HTML form di `index.html`

### Mengubah Model

1. Ubah model file name di `app.py`: `load_model()`
2. Jika fitur berbeda, update sesuai langkah di atas

### Styling Kustom

Edit `static/css/style.css` untuk mengubah tampilan.

## 📚 Referensi

- [Flask Documentation](https://flask.palletsprojects.com/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Bootstrap 5](https://getbootstrap.com/)
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)

## 👨‍💻 Author

- **Student Name:** [Your Name]
- **Institution:** [University Name]
- **Course:** Praktikum Machine Learning
- **Semester:** 5
- **Academic Year:** 2025/2026

## 📄 License

Educational Purpose Only - Tahun Akademik 2025/2026

## ⚠️ Important Disclaimer

```
DISCLAIMER FOR MEDICAL APPLICATION

Aplikasi ini adalah proyek edukasi dan penelitian yang tidak dimaksudkan untuk 
penggunaan medis profesional. Hasil prediksi dari aplikasi ini BUKAN merupakan 
diagnosa medis yang sah.

Untuk diagnosa medis yang akurat dan terpercaya, SELALU berkonsultasi dengan 
tenaga medis profesional yang bersertifikat.

Pengembang aplikasi ini tidak bertanggung jawab atas:
- Kesalahan interpretasi hasil prediksi
- Keputusan medis berdasarkan hasil aplikasi ini
- Kehilangan data atau privasi pengguna
- Akurasi prediksi yang kurang sempurna

GUNAKAN APLIKASI INI HANYA UNTUK TUJUAN PEMBELAJARAN DAN PENELITIAN.
```

---

**Last Updated:** 6 Januari 2026  
**Version:** 1.0.0  
**Status:** Ready for Deployment ✅
