import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np 
import pickle 

st.set_page_config(page_title="DataSoc | Model Automator", layout="wide")

# --- CUSTOM CSS: SMOOTH UI & SCROLLBAR FIX ---
st.markdown("""
    <style>
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; margin: 4px; }
    ::-webkit-scrollbar-thumb { background: rgba(136, 136, 136, 0.4); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(136, 136, 136, 0.8); }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 DataSoc: Dynamic Model Automator")
st.markdown("""
Upload any CSV dataset to dynamically train, evaluate, and export Machine Learning models. 
This is a real-world prototype for the Technology Portfolio's internal tools.
""")

# --- DATA UPLOAD ---
uploaded_file = st.sidebar.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.dropna()

    st.sidebar.success("Dataset uploaded successfully!")
    
    # --- DATA CONFIGURATION ---
    st.sidebar.header("Data Configuration")
    target_col = st.sidebar.selectbox("🎯 Select Target Column (To Predict)", df.columns)
    
    is_continuous = pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 20
    
    if is_continuous:
        st.sidebar.error(f"⚠️ '{target_col}' has {df[target_col].nunique()} unique numeric values. Classification accuracy will likely be 0%. Please select a categorical column.")
    else:
        st.sidebar.success(f"✅ '{target_col}' is suitable for classification.")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if df[target_col].dtype == 'object':
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])

    # --- MODEL CONFIGURATION ---
    st.sidebar.header("Model Configuration")
    model_name = st.sidebar.selectbox("Select Algorithm", ["Random Forest", "KNN", "Logistic Regression"])

    if model_name == "Random Forest":
        param = st.sidebar.slider("Number of estimators", 1, 100, 50)
        clf = RandomForestClassifier(n_estimators=param, random_state=42)
    elif model_name == "KNN":
        param = st.sidebar.slider("Number of neighbors (K)", 1, 15, 3)
        clf = KNeighborsClassifier(n_neighbors=param)
    else: 
        param = st.sidebar.select_slider(
            "Regularization strength (C)", 
            options=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 
            value=1.0
        )
        clf = LogisticRegression(C=param, max_iter=2000, random_state=42)

    # --- MAIN PAGE: DATA EXPLORATION & VISUALIZATION ---
    col_data, col_viz = st.columns([4, 5]) 

    with col_data:
        st.subheader("📊 Dataset Preview")
        preview_df = df.head(15).copy().astype(str)
        preview_df.loc[' '] = '' 
        st.dataframe(preview_df, height=500, use_container_width=True)

    with col_viz:
        st.subheader("📈 Dynamic Visualization")
        if len(numeric_cols) >= 2:
            def format_column_name(col):
                words = str(col).split('_')
                units = {'mm': 'In MM', 'cm': 'In CM', 'm': 'In M', 'g': 'In G', 'kg': 'In KG', 'usd': 'In USD', 'id': '(Raw ID)'}
                if words[-1].lower() in units:
                    words[-1] = units[words[-1].lower()]
                readable = " ".join(words).capitalize()
                return f"{readable} - ({col})"

            col_x = st.selectbox("Select X-axis", numeric_cols, index=0, format_func=format_column_name)
            col_y = st.selectbox("Select Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, format_func=format_column_name)
            
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=col_x, y=col_y, hue=target_col, palette="viridis", ax=ax)
            if df[col_x].nunique() <= 10: ax.set_xticks(sorted(df[col_x].unique()))
            if df[col_y].nunique() <= 10: ax.set_yticks(sorted(df[col_y].unique()))
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns for scatter plot.")

    # --- MODEL TRAINING & RESULTS ---
    st.divider()
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    if len(feature_cols) > 0:
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with st.spinner('Training model...'):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

        # --- UPGRADE: MODEL EVALUATION UI ---
        # 1. TOP SECTION: Independent header and metric
        st.subheader(f"🏆 Model Evaluation ({model_name})")
        st.metric(label="Overall Accuracy", value=f"{acc*100:.2f}%")
        
        # Add a little spacing before the charts
        st.write("") 
        
        # 2. ROW 1: Chart Headers (Aligned horizontally)
        col_head_left, col_head_right = st.columns([1, 1.2])
        
        with col_head_left:
            st.write("**Confusion Matrix Heatmap**")
            
        with col_head_right:
            # Using st.write with bold markdown makes it smaller than subheader 
            # and perfectly matches the size of the heatmap header.
            st.write("**🔍 Feature Importance**")

        # 3. ROW 2: The Charts (Aligned horizontally)
        col_chart_left, col_chart_right = st.columns([1, 1.2])
        
        with col_chart_left:
            fig_cm, ax_cm = plt.subplots(figsize=(5, 3.5)) 
            labels = sorted(df[target_col].unique()) 
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, xticklabels=labels, yticklabels=labels, cbar=False)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig_cm)

        with col_chart_right:
            if model_name == "Random Forest":
                importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': clf.feature_importances_}).sort_values(by='Importance', ascending=False)
                fig_imp, ax_imp = plt.subplots(figsize=(4.9, 3.5)) 
                sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax_imp, palette="mako")
                st.pyplot(fig_imp)
                
            elif model_name == "Logistic Regression":
                importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': np.abs(clf.coef_[0])}).sort_values(by='Importance', ascending=False)
                fig_imp, ax_imp = plt.subplots(figsize=(4.9, 3.5)) 
                sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax_imp, palette="rocket")
                st.pyplot(fig_imp)
                
            else:
                # Add vertical spacing to keep layout balanced if KNN is selected
                st.markdown("<div style='height: 150px;'></div>", unsafe_allow_html=True)
                st.info("💡 KNN is distance-based and does not provide Feature Importance.")

        # 4. ROW 3: Captions (Forced to align perfectly at the bottom)
        col_cap_left, col_cap_right = st.columns([1, 1.2])
        
        with col_cap_left:
            st.caption("ℹ️ *Accuracy shows the overall success, while the heatmap reveals specific misclassifications.*")
            
        with col_cap_right:
            st.caption("ℹ️ *Variables with the strongest influence on model decisions. Higher bars indicate greater impact.*")

        # --- FINAL SECTION: EXPORT MODEL ---
        st.divider()
        st.subheader("💾 Export Model")
        col_exp1, col_exp2 = st.columns([2, 1])
        with col_exp1:
            st.write("Download this trained model as a `.pkl` file for integration into production systems without retraining.")
        with col_exp2:
            model_bytes = pickle.dumps(clf)
            st.download_button(
                label="⬇️ Download Trained Model (.pkl)",
                data=model_bytes,
                file_name=f"datasoc_{model_name.replace(' ', '_').lower()}.pkl",
                mime="application/octet-stream"
            )
            
    else:
        st.error("Please ensure your dataset has numeric features to train the model.")

else:
    st.info("👈 Please upload a CSV file from the sidebar to start exploring.")
