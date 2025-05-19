# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    df = pd.read_csv('C:/Users/Acer/Documents/berkas kuliah/smt 4/Penambangan Data/covid_19_indonesia_time_series_all.csv')
    df['Case Fatality Rate'] = df['Case Fatality Rate'].str.replace('%', '', regex=False)
    df['Case Fatality Rate'] = pd.to_numeric(df['Case Fatality Rate'], errors='coerce')
    df = df.dropna(subset=['Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate'])
    latest = df.sort_values('Date').groupby('Location').tail(1)
    return latest

df = load_data()

st.title("📊 Dashboard COVID-19 Indonesia")
st.write("Data terakhir per lokasi")
st.dataframe(df[['Location', 'Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate']])

# ===================== PREDIKSI =====================
st.header("📈 Prediksi Jumlah Kasus (Regresi)")

X = df[['Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate']].apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(df['Total Cases'], errors='coerce')

# Filter NaN secara aman
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

# Cek apakah ada data cukup
if len(X) < 5:
    st.error("❌ Data tidak cukup untuk training. Periksa apakah data Anda valid.")
    st.stop()

# Split dan training model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.metric("MAE", f"{mae:.2f}")
st.metric("RMSE", f"{rmse:.2f}")

# ===================== CLUSTERING =====================
st.header("🧠 Clustering Lokasi (KMeans)")

features = ['Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']
X_cluster = df[features].apply(pd.to_numeric, errors='coerce').dropna()

if len(X_cluster) < 3:
    st.error("❌ Data tidak cukup untuk clustering. Minimal 3 lokasi dengan data lengkap.")
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df.loc[X_cluster.index, 'Cluster'] = kmeans.fit_predict(X_scaled)

fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Population Density', y='Total Cases', hue='Cluster', palette='Set2', ax=ax)
ax.set_title("Clustering Berdasarkan Kasus & Densitas Penduduk")
st.pyplot(fig)

