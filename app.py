import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle # Thư viện dùng để lưu và tải mô hình

st.set_page_config(page_title="DataSoc | Model Automator", layout="wide")

st.title("🚀 DataSoc: Dynamic Model Automator")
st.markdown("""
Upload any CSV dataset to dynamically train, evaluate, and export Machine Learning models. 
This is a real-world prototype for the Technology Portfolio's internal tools.
""")

# --- TẢI DỮ LIỆU LÊN ---
uploaded_file = st.sidebar.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.dropna()

    st.sidebar.success("Dataset uploaded successfully!")
    
    # --- CẤU HÌNH DỮ LIỆU ---
    st.sidebar.header("Data Configuration")
    target_col = st.sidebar.selectbox("🎯 Select Target Column (To Predict)", df.columns)
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if df[target_col].dtype == 'object':
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])

    # --- CẤU HÌNH MÔ HÌNH ---
    st.sidebar.header("Model Configuration")
    model_name = st.sidebar.selectbox("Select Algorithm", ["Random Forest", "KNN"])

    if model_name == "Random Forest":
        param = st.sidebar.slider("Number of estimators", 1, 100, 50)
        clf = RandomForestClassifier(n_estimators=param, random_state=42)
    else:
        param = st.sidebar.slider("Number of neighbors (K)", 1, 15, 3)
        clf = KNeighborsClassifier(n_neighbors=param)

    # --- TRANG CHÍNH: KHÁM PHÁ DỮ LIỆU & TRỰC QUAN HÓA ---
    # Sử dụng columns để đặt bảng và biểu đồ nằm ngang hàng
    # Tỉ lệ [1, 2] giúp bảng chiếm 1/3 và biểu đồ chiếm 2/3 chiều rộng
    col_data, col_viz = st.columns([4, 5]) 

    with col_data:
        st.subheader("📊 Dataset Preview")
        # Thêm tham số height để giới hạn chiều cao bảng
        st.dataframe(df.head(15), height=500, use_container_width=True)

    with col_viz:
        st.subheader("📈 Dynamic Visualization")
        if len(numeric_cols) >= 2:
            col_x = st.selectbox("Select X-axis", numeric_cols, index=0)
            col_y = st.selectbox("Select Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=col_x, y=col_y, hue=target_col, palette="viridis", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns for scatter plot.")

    # --- HUẤN LUYỆN MÔ HÌNH ---
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
            
        st.subheader(f"🏆 Prediction Results ({model_name})")
        st.metric(label="Model Accuracy", value=f"{acc*100:.2f}%")
        
        # ==========================================================
        # TÍNH NĂNG NÂNG CAO: FEATURE IMPORTANCE & MODEL EXPORT
        # ==========================================================
        st.divider()
        col_feat, col_export = st.columns(2)
        
        with col_feat:
            st.subheader("🔍 Feature Importance")
            if model_name == "Random Forest":
                # Lấy độ quan trọng của từng biến và vẽ biểu đồ
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': clf.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                fig_imp, ax_imp = plt.subplots()
                sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax_imp, palette="mako")
                st.pyplot(fig_imp)
            else:
                st.info("💡 Thuật toán KNN không hỗ trợ tính toán Feature Importance.")
                
        with col_export:
            st.subheader("💾 Export Model")
            st.write("Tải mô hình đã huấn luyện về máy để tích hợp vào các hệ thống khác mà không cần train lại.")
            
            # Đóng gói mô hình thành byte
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
