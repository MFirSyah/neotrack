import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from supabase import create_client, Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# ==========================================
# 1. KONFIGURASI & SETUP
# ==========================================
# Mengambil kredensial dari Environment Variables (GitHub Secrets)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
UID = os.getenv("USER_ID") # UID kamu

BUCKET_NAME = "models"
MODEL_H5_NAME = "forecaster_champion.h5" # Model asli untuk ditraining ulang
MODEL_TFLITE_NAME = "forecaster_v1.tflite" # Model untuk di-download Flutter
SCALER_NAME = "scaler_params.json"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 2. DATA PROCESSING & MAD FILTERING
# ==========================================
def mask_outliers_mad(series, threshold=3.0):
    """Filter MAD 3.0 untuk menyembunyikan transaksi outlier (Misal: Self Reward 500rb)"""
    median = series.median()
    abs_deviation = (series - median).abs()
    mad = abs_deviation.median()
    
    if mad == 0: return series # Hindari pembagian dengan nol
    
    modified_z_score = 0.6745 * abs_deviation / mad
    cleaned_series = series.copy()
    cleaned_series[modified_z_score > threshold] = median
    return cleaned_series

def fetch_and_prep_data():
    print("📥 Mengambil data dari Supabase...")
    response = supabase.table("transactions").select("amount, date, type").eq("user_id", UID).eq("type", "expense").execute()
    df = pd.DataFrame(response.data)
    
    if df.empty:
        raise ValueError("Data kosong! AI tidak bisa belajar.")

    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Resampling: Jumlahkan pengeluaran per hari & isi hari yang kosong dengan 0
    daily_series = df.groupby('date')['amount'].sum().reset_index()
    all_days = pd.date_range(start=daily_series['date'].min(), end=daily_series['date'].max(), freq='D')
    daily_series = daily_series.set_index('date').reindex(all_days.date).fillna(0).reset_index()
    daily_series.rename(columns={'index': 'date'}, inplace=True)
    
    # Terapkan The Guardian Filter (MAD 3.0)
    daily_series['cleaned_amount'] = mask_outliers_mad(daily_series['amount'], threshold=3.0)
    return daily_series

# ==========================================
# 3. PREPARATION UNTUK LSTM
# ==========================================
def create_sequences(data, window_size=7):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# ==========================================
# 4. DOWNLOAD CHAMPION MODEL & SCALER
# ==========================================
def download_champion():
    print("🔍 Mencari Champion Model di Supabase Storage...")
    try:
        # Download scaler lama
        scaler_res = supabase.storage.from_(BUCKET_NAME).download(SCALER_NAME)
        with open(SCALER_NAME, 'wb') as f: f.write(scaler_res)
        
        # Download h5 lama
        h5_res = supabase.storage.from_(BUCKET_NAME).download(MODEL_H5_NAME)
        with open(MODEL_H5_NAME, 'wb') as f: f.write(h5_res)
        
        return True
    except Exception as e:
        print("⚠️ Champion tidak ditemukan. Kita akan buat model dari nol (Base Model).")
        return False

# ==========================================
# 5. MAIN PIPELINE (CHAMPION VS CHALLENGER)
# ==========================================
def main():
    # 1. Siapkan Data
    df = fetch_and_prep_data()
    values = df['cleaned_amount'].values.reshape(-1, 1)

    # 2. Coba Download Model Lama
    has_champion = download_champion()

    # 3. Scaling Logic (Penting untuk konsistensi Flutter!)
    scaler = MinMaxScaler(feature_range=(0, 1))
    if has_champion:
        with open(SCALER_NAME, 'r') as f:
            old_params = json.load(f)
        # Paksa Min-Max lama agar "penggaris" AI tidak berubah drastis
        scaler.data_min_ = np.array([old_params['min']])
        scaler.data_max_ = np.array([old_params['max']])
        scaler.min_ = np.array([old_params['scale_min']])
        scaler.scale_ = np.array([old_params['scale']])
        scaled_data = scaler.transform(values)
    else:
        # Kalau baru pertama kali, fit data dari nol
        scaled_data = scaler.fit_transform(values)

    # Buat Dataset (X = 7 hari ke belakang, y = hari ini)
    X, y = create_sequences(scaled_data, window_size=7)

    # Pisahkan 80% untuk Training, 20% untuk Test (Evaluasi)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    # 4. Transfer Learning Logic
    if has_champion:
        print("🧠 Melakukan Fine-Tuning pada Champion Model...")
        model = load_model(MODEL_H5_NAME)
        # Learning rate sangat kecil agar tidak lupa ingatan (Catastrophic Forgetting)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
        
        # Evaluasi Champion sebelum diubah
        champion_loss, champion_mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"🏆 Skor Champion (MAE): {champion_mae:.4f}")
    else:
        print("🧠 Membangun Arsitektur Baru...")
        model = Sequential([
            Input(shape=(7, 1), batch_size=1),
            LSTM(16, activation='tanh', return_sequences=False),
            Dense(8, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        champion_mae = float('inf') # Set ke angka tak terhingga agar pasti di-update

    # 5. Latih Challenger
    print("🥊 Melatih Challenger Model...")
    # Epoch kecil karena kita cuma menyesuaikan (fine-tuning)
    history = model.fit(X_train, y_train, epochs=10, batch_size=1, validation_data=(X_test, y_test), verbose=0)
    
    challenger_mae = history.history['val_mae'][-1]
    print(f"🥊 Skor Challenger (MAE): {challenger_mae:.4f}")

    # 6. PENJURIAN (The Judge)
    if challenger_mae < champion_mae or not has_champion:
        print("✅ CHALLENGER MENANG! Akurasi lebih baik. Bersiap Upload...")
        
        # Simpan format H5 (Untuk di-retrain minggu depan)
        model.save(MODEL_H5_NAME)
        
        # Konversi ke TFLite (Untuk didownload Flutter) dengan Flex Ops
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        
        with open(MODEL_TFLITE_NAME, 'wb') as f: f.write(tflite_model)
        
        # Ekspor Parameter Scaler
        new_params = {
            "min": float(scaler.data_min_[0]),
            "max": float(scaler.data_max_[0]), # Flutter akan pakai angka ini
            "scale_min": float(scaler.min_[0]),
            "scale": float(scaler.scale_[0])
        }
        with open(SCALER_NAME, 'w') as f: json.dump(new_params, f)

        # 7. UPLOAD KE SUPABASE
        for file_name in [MODEL_H5_NAME, MODEL_TFLITE_NAME, SCALER_NAME]:
            with open(file_name, 'rb') as f:
                supabase.storage.from_(BUCKET_NAME).upload(
                    path=file_name, file=f, file_options={"x-upsert": "true"}
                )
        
        # Update Registry untuk Flutter
        supabase.table("ai_model_registry").insert({
            "version": 2 if has_champion else 1, 
            "file_name": MODEL_TFLITE_NAME,
            "model_type": "forecaster"
        }).execute()
        
        print("🚀 DEPLOYMENT SUKSES! Otak AI terbaru sudah mengudara.")
    else:
        print("❌ CHALLENGER KALAH. Model lama masih lebih akurat. Tidak ada perubahan yang di-upload.")

if __name__ == "__main__":
    main()
