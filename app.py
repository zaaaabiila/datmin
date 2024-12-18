# Install library
# Jalankan perintah ini di terminal jika belum terinstal: pip install streamlit pandas numpy scikit-learn matplotlib seaborn

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Judul aplikasi
st.title("Aplikasi Clustering Dataset Alkohol")
st.write("Upload dataset Anda untuk melakukan clustering dan visualisasi hasilnya.")

# Upload file dataset
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Awal:")
    st.dataframe(data)

    # Preprocessing
    st.header("Step 1: Preprocessing")
    data = data.drop_duplicates()
    if data.isnull().sum().any():
        data = data.dropna()
        st.write("Baris dengan nilai kosong telah dihapus.")
    st.write("Dataset Setelah Preprocessing:")
    st.dataframe(data)

    # Encoding fitur kategorikal
    st.write("Encoding fitur kategorikal...")
    categorical_cols = ['Jenis', 'Nama Produk', 'Merk', 'Kemasan', 'Importir']
    encoder = LabelEncoder()
    for col in categorical_cols:
        if col in data.columns:
            data[col] = encoder.fit_transform(data[col])

    # Normalisasi data numerik
    st.write("Normalisasi data numerik...")
    numerical_cols = ['Kadar (%)', 'Volume (ml)']
    scaler = StandardScaler()
    if all(col in data.columns for col in numerical_cols):
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    st.write("Dataset Setelah Normalisasi:")
    st.dataframe(data)

    # Clustering dengan K-Means
    st.header("Step 2: K-Means Clustering")
    X = data[numerical_cols]
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Elbow method plot
    st.write("Metode Elbow untuk menentukan jumlah cluster:")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    st.pyplot(plt)

    # Pilih jumlah cluster
    optimal_k = st.slider("Pilih jumlah cluster untuk K-Means:", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    data['KMeans_Cluster'] = kmeans_labels

    st.write(f"Silhouette Score untuk K-Means: {silhouette_score(X, kmeans_labels):.2f}")

    # Visualisasi K-Means
    st.write("Visualisasi Hasil Clustering dengan K-Means:")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=kmeans_labels, palette='viridis')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    st.pyplot(plt)

    # Agglomerative Clustering
    st.header("Step 3: Agglomerative Clustering")
    agglo = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    agglo_labels = agglo.fit_predict(X)
    data['Agglo_Cluster'] = agglo_labels

    st.write(f"Silhouette Score untuk Agglomerative Clustering: {silhouette_score(X, agglo_labels):.2f}")

    # Visualisasi Agglomerative Clustering
    st.write("Visualisasi Hasil Clustering dengan Agglomerative Clustering:")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=agglo_labels, palette='viridis')
    plt.title('Agglomerative Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    st.pyplot(plt)

    # Dendrogram
    st.write("Dendrogram untuk Agglomerative Clustering:")
    linked = linkage(X, method='ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linked, truncate_mode='level', p=5)
    plt.title('Dendrogram')
    st.pyplot(plt)

    # Download hasil clustering
    st.header("Step 4: Unduh Hasil Clustering")
    csv = data.to_csv(index=False)
    st.download_button("Download Hasil Clustering", csv, "hasil_clustering.csv", "text/csv")

else:
    st.info("Silakan upload file CSV untuk memulai.")
