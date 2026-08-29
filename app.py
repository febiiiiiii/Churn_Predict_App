"""
Aplikasi sederhana untuk memprediksi customer churn.
Cara menjalankan:
    pip install streamlit joblib pandas scikit-learn
    streamlit run app.py

File ini membaca 3 hasil dari notebook ML_Churn.ipynb:
    - logistic_regression_churn.pkl  (model)
    - scaler.pkl                     (standard scaler)
    - feature_columns.pkl            (urutan kolom fitur hasil encoding)
Pastikan ketiga file .pkl tersebut ada di folder yang sama dengan app.py ini.
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Customer Churn", page_icon="📉")

# ---- Load model, scaler, dan daftar kolom fitur ----
@st.cache_resource
def load_artifacts():
    model = joblib.load("logistic_regression_churn.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

st.title("📉 Prediksi Customer Churn")
st.write(
    "Isi data pelanggan di bawah ini, lalu klik **Prediksi** untuk melihat "
    "apakah pelanggan tersebut berpotensi berhenti berlangganan (churn)."
)

# ---- Form input ----
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Punya Partner", ["No", "Yes"])
        dependents = st.selectbox("Punya Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (bulan berlangganan)", min_value=0, max_value=100, value=12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=1.0)

    submitted = st.form_submit_button("🔮 Prediksi")

if submitted:
    # Encoding manual satu baris data, meniru hasil pd.get_dummies(drop_first=True)
    # saat training. Catatan: pd.get_dummies TIDAK bisa dipakai langsung di sini —
    # untuk 1 baris data, tiap kolom cuma punya 1 kategori, sehingga drop_first
    # akan mengosongkan kolom itu sama sekali dan hasilnya jadi salah.
    # Solusinya: mulai dari semua kolom fitur = 0, lalu nyalakan (=1) kolom yang
    # sesuai kategori yang dipilih user. Kalau kategori yang dipilih adalah
    # kategori "referensi" (yang di-drop saat training), kolomnya memang tetap 0 —
    # itu perilaku yang benar.
    encoded_dict = {col: 0 for col in feature_columns}
    encoded_dict["SeniorCitizen"] = 1 if senior_citizen == "Yes" else 0
    encoded_dict["tenure"] = tenure
    encoded_dict["MonthlyCharges"] = monthly_charges
    encoded_dict["TotalCharges"] = total_charges

    categorical_inputs = {
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
    }
    for col_prefix, value in categorical_inputs.items():
        dummy_col = f"{col_prefix}_{value}"
        if dummy_col in encoded_dict:
            encoded_dict[dummy_col] = 1

    # Susun jadi 1 baris DataFrame dengan urutan kolom PERSIS seperti saat training
    encoded = pd.DataFrame([encoded_dict])[feature_columns]

    # Scaling, lalu prediksi
    scaled_input = scaler.transform(encoded)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ Pelanggan diprediksi **AKAN CHURN** (probabilitas: {probability*100:.1f}%)")
    else:
        st.success(f"✅ Pelanggan diprediksi **TIDAK akan churn** (probabilitas churn: {probability*100:.1f}%)")

    st.progress(min(int(probability * 100), 100))
