import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Aplikasi Prediksi Pelanggan", layout="wide")
st.title("🎯 Aplikasi Analisis & Prediksi Respon Pelanggan")
st.markdown("Aplikasi simulasi prediksi pemasaran untuk mengukur potensi respon pelanggan berdasarkan profil demografi Indonesia.")
st.write("---")

# --- DAFTAR KOLOM DARI MODEL ---
semua_kolom = [
    'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 
    'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
    'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 
    'AcceptedCmp2', 'Complain', 'Age', 'Education_Basic', 'Education_Graduation', 
    'Education_Master', 'Education_PhD', 'Marital_Status_Alone', 'Marital_Status_Divorced', 
    'Marital_Status_Married', 'Marital_Status_Single', 'Marital_Status_Together', 
    'Marital_Status_Widow', 'Marital_Status_YOLO'
]

# --- SIDEBAR INPUT KONTEKS INDONESIA ---
with st.sidebar:
    st.header("📝 Profil Pelanggan (Indonesia)")
    st.write("Silakan isi data simulasi di bawah ini:")
    
    # 1. Demografi Lokalan
    st.subheader("1. Data Demografi")
    age = st.number_input("Usia Pelanggan (Tahun)", min_value=18, max_value=90, value=25)
    
    # Input Rupiah per Bulan
    pendapatan_rupiah = st.number_input(
        "Pendapatan Bulanan (Rupiah)", 
        min_value=0, 
        value=7000000, 
        step=500000,
        help="Masukkan dalam Rupiah. Sistem akan otomatis mengonversi ke USD Tahunan untuk kebutuhan Model ML."
    )
    
    # Konversi otomatis: (Rupiah * 12 bulan) / Kurs (Asumsi $1 = Rp 16.000)
    income_converted = (pendapatan_rupiah * 12) / 16000
    
    # Pemetaan Pendidikan (Disesuaikan agar mudah dipahami)
    pendidikan_pilihan = st.selectbox(
        "Tingkat Pendidikan Terakhir", 
        options=["SMA/Diploma (Basic)", "S1 (Graduation)", "S2 (Master)", "S3 (PhD)"]
    )
    # Ambil kata kunci aslinya untuk model
    map_edu = {"SMA/Diploma (Basic)": "Basic", "S1 (Graduation)": "Graduation", "S2 (Master)": "Master", "S3 (PhD)": "PhD"}
    pendidikan = map_edu[pendidikan_pilihan]
    
    # Pemetaan Status Pernikahan
    status_pilihan = st.selectbox(
        "Status Pernikahan", 
        options=["Belum Menikah (Single)", "Menikah (Married)", "Tinggal Bersama (Together)", "Cerai (Divorced)", "Janda/Duda (Widow)"]
    )
    map_status = {"Belum Menikah (Single)": "Single", "Menikah (Married)": "Married", "Tinggal Bersama (Together)": "Together", "Cerai (Divorced)": "Divorced", "Janda/Duda (Widow)": "Widow"}
    status_pernikahan = map_status[status_pilihan]
    
    # 2. Perilaku Belanja
    st.subheader("2. Aktivitas Belanja")
    recency = st.sidebar.number_input("Berapa hari yang lalu pelanggan terakhir kali berbelanja?", min_value=0, value=10,help="Isi 0 jika baru belanja hari ini. Isi 30 jika terakhir kali belanja adalah sebulan yang lalu.")
    web_visits = st.number_input("Jumlah Buka Website Per Bulan (Kali)", min_value=0, value=4)
    catalog_purchases = st.number_input("Pembelian Lewat Katalog (Total)", min_value=0, value=1)
    
    complain_pilihan = st.selectbox("Pernah Komplain/Mengeluh?", options=["Tidak", "Ya"])
    complain = 1 if complain_pilihan == "Ya" else 0

    st.write("---")
    submit_button = st.button("🚀 Hitung Prediksi")

# --- PROSES SIMULASI ---
if submit_button:
    with st.spinner("Model sedang menganalisis data..."):
        try:
            # 1. Buat data kosong
            input_data = pd.DataFrame(np.zeros((1, len(semua_kolom))), columns=semua_kolom)
            
            # 2. Masukkan data yang sudah dikonversi
            input_data['Age'] = age
            input_data['Income'] = income_converted # Menggunakan hasil konversi USD
            input_data['Recency'] = recency
            input_data['NumWebVisitsMonth'] = web_visits
            input_data['NumCatalogPurchases'] = catalog_purchases
            input_data['Complain'] = complain
            
            # One-Hot Encoding Otomatis
            kolom_pendidikan = f"Education_{pendidikan}"
            if kolom_pendidikan in input_data.columns:
                input_data[kolom_pendidikan] = 1
                
            kolom_pernikahan = f"Marital_Status_{status_pernikahan}"
            if kolom_pernikahan in input_data.columns:
                input_data[kolom_pernikahan] = 1

            # 3. Panggil Model
            model = joblib.load('model.pkl')
            scaler = joblib.load('scaler.pkl')
            
            # 4. Prediksi
            data_scaled = scaler.transform(input_data)
            prediction = model.predict(data_scaled)
            
            # 5. Tampilan Output Bahasa Indonesia
            st.header("📋 Hasil Analisis Kecerdasan Buatan")
            st.info(f"Catatan Sistem: Pendapatan Rp {pendapatan_rupiah:,.0f}/bulan setara dengan Angka Model: ${income_converted:,.2f}/tahun")
            
            if prediction[0] == 1:
                st.success("🎯 **HASIL: PELANGGAN POTENSIAL**")
                st.write("Pelanggan dengan profil ini memiliki kecenderungan **TINGGI** untuk merespons positif kampanye pemasaran kita. Direkomendasikan untuk diberikan prioritas promosi.")
            else:
                st.error("📉 **HASIL: PELANGGAN KURANG POTENSIAL**")
                st.write("Pelanggan dengan profil ini memiliki kecenderungan **RENDAH** untuk merespons kampanye. Disarankan untuk menunda promosi guna menghemat biaya operasional.")

        except Exception as e:
            st.error(f"Terjadi error pada sistem: {e}")