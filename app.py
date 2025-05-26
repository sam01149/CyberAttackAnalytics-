import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Cyber Attack Analytics", layout="wide")

st.title('Cybersecurity Data Science Dashboard')
st.markdown("Upload dataset insiden serangan siber dan eksplorasi lengkap!")

uploaded_file = st.sidebar.file_uploader('Upload file CSV/Excel', type=['csv','xlsx'])
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.header("Data Overview")
    st.write(df.head())
    st.write("Dimensi data:", df.shape)
    st.write("Summary Numeric:", df.describe())
    st.write("Missing value per kolom:", df.isnull().sum())

    # --- Data Preparation ---
    cat_cols = df.select_dtypes('object').columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col+"_le"] = le.fit_transform(df[col])
        label_encoders[col] = le

    df_prep = df.copy()
    for col in cat_cols:
        df_prep[col] = df_prep[col+"_le"]

    num_cols = ['Financial Loss (in Million $)', 'Number of Affected Users', 'Incident Resolution Time (in Hours)']
    scaler = StandardScaler()
    df_prep[num_cols] = scaler.fit_transform(df_prep[num_cols])

    # ---- FILTER DATA ----
    st.sidebar.subheader("Filter data")
    tahun_opsi = st.sidebar.multiselect("Tahun", sorted(df['Year'].unique()), default=sorted(df['Year'].unique()))
    negara_opsi = st.sidebar.multiselect("Negara", sorted(df['Country'].unique()), default=sorted(df['Country'].unique()))
    ind_opsi = st.sidebar.multiselect("Industri", sorted(df['Target Industry'].unique()), default=sorted(df['Target Industry'].unique()))
    filtered = df[
        (df['Year'].isin(tahun_opsi)) &
        (df['Country'].isin(negara_opsi)) &
        (df['Target Industry'].isin(ind_opsi))
    ]
    filtered_prep = df_prep.loc[filtered.index]
    st.write("Sisa data setelah filter:", filtered.shape)

    # ---- DATA EXPLORER ----
    st.header("Data Explorer")
    tab1, tab2 = st.tabs(["Attack Type & Industri", "Negara & Tahun"])
    with tab1:
        f, ax = plt.subplots(1,2,figsize=(12,4))
        sns.countplot(data=filtered, y='Attack Type', ax=ax[0])
        sns.countplot(data=filtered, y='Target Industry', ax=ax[1])
        plt.tight_layout()
        st.pyplot(f)
    with tab2:
        f, ax = plt.subplots(1,2,figsize=(12,4))
        sns.countplot(data=filtered, y='Country', ax=ax[0])
        sns.countplot(data=filtered, x='Year', ax=ax[1])
        plt.tight_layout()
        st.pyplot(f)

        # --- FITUR untuk Supervised dan Unsupervised ---
    nb_features = [
        'Country_le', 'Year', 'Target Industry_le',
        'Financial Loss (in Million $)', 'Number of Affected Users',
        'Attack Source_le', 'Security Vulnerability Type_le',
        'Defense Mechanism Used_le', 'Incident Resolution Time (in Hours)'
    ]

    # ---- K-MEANS CLUSTERING ----
    st.header("K-Means Clustering")
    k = st.slider("Jumlah cluster (k)", 2, 8, 4)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(filtered_prep[nb_features])   # <--- hanya nb_features!
    filtered['Cluster'] = clusters

    st.write("Jumlah data tiap cluster:", filtered['Cluster'].value_counts())
    st.write("Rata-rata numerik per cluster:", filtered.groupby('Cluster')[num_cols].mean())

    # Visualisasi cluster (PCA 2D, nb_features saja)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(filtered_prep[nb_features])
    f,ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=clusters, palette="Set2", alpha=0.7)
    plt.title("Visualisasi Cluster (PCA 2D)")
    st.pyplot(f)

    # ---- NAIVE BAYES CLASSIFIER ----
    st.header("Naive Bayes Classifier (Predict Attack Type)")
    X_nb = filtered_prep[nb_features]
    y_nb = filtered_prep['Attack Type_le']
    X_train, X_test, y_train, y_test = train_test_split(X_nb, y_nb, test_size=0.2, stratify=y_nb, random_state=42)
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write("Akurasi (test set):", f"{acc:.2f}")

    # Confusion matrix
    f,ax = plt.subplots(figsize=(7,5))
    cfmat = confusion_matrix(y_test, y_pred)
    attack_type_classes = label_encoders['Attack Type'].classes_
    sns.heatmap(cfmat, annot=True, cmap="Blues", xticklabels=attack_type_classes, yticklabels=attack_type_classes)
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix")
    st.pyplot(f)

    # Classification report
    st.text(classification_report(y_test, y_pred, target_names=attack_type_classes))

    # ---- PREDICT ATTACK TYPE: FORM ----
    st.subheader("Prediksi Attack Type dari Data Baru")
    with st.form("PrediksiNB"):
        country = st.selectbox("Country", label_encoders['Country'].classes_)
        year = st.selectbox("Year", sorted(df['Year'].unique()))
        target_ind = st.selectbox("Target Industry", label_encoders['Target Industry'].classes_)
        fl = st.number_input("Financial Loss (in Million $)", min_value=0.0, max_value=1e7, step=0.01)
        n_affected = st.number_input("Number of Affected Users", min_value=0, max_value=int(1e8), step=1)
        attack_source = st.selectbox("Attack Source", label_encoders['Attack Source'].classes_)
        vuln = st.selectbox("Security Vulnerability Type", label_encoders['Security Vulnerability Type'].classes_)
        defense = st.selectbox("Defense Mechanism Used", label_encoders['Defense Mechanism Used'].classes_)
        rt_hr = st.number_input("Incident Resolution Time (in Hours)", min_value=0, max_value=10000, step=1)
        sbmt = st.form_submit_button("Prediksi")

    if sbmt:
        numerik_scaled = scaler.transform([[fl, n_affected, rt_hr]])[0]
        form_row = {
            "Country_le": label_encoders['Country'].transform([country])[0],
            "Year": year,
            "Target Industry_le": label_encoders['Target Industry'].transform([target_ind])[0],
            "Financial Loss (in Million $)": numerik_scaled[0],
            "Number of Affected Users": numerik_scaled[1],
            "Attack Source_le": label_encoders['Attack Source'].transform([attack_source])[0],
            "Security Vulnerability Type_le": label_encoders['Security Vulnerability Type'].transform([vuln])[0],
            "Defense Mechanism Used_le": label_encoders['Defense Mechanism Used'].transform([defense])[0],
            "Incident Resolution Time (in Hours)": numerik_scaled[2]
        }
        Xbaru = pd.DataFrame([form_row])[nb_features]
        y_pred_new = nb.predict(Xbaru)
        pred_label = label_encoders['Attack Type'].inverse_transform(y_pred_new)[0]
        cluster_pred = km.predict(Xbaru)[0]
        st.success(f"Prediksi Attack Type: **{pred_label}** (masuk cluster {cluster_pred})")

    # --- Download hasil cluster ---
    st.download_button("Download hasil clustering", data=filtered.to_csv(index=False), file_name="hasil_clustering.csv")



