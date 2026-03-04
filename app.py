import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="DataSoc | Model Explorer", layout="wide")

st.title("🚀 DataSoc: Interactive Model Explorer")
st.markdown("""
This application serves as a prototype for a **Model Automator**, allowing users to 
explore datasets and experiment with Machine Learning models directly in the browser.
""")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

st.sidebar.header("Model Configuration")
model_name = st.sidebar.selectbox("Select Algorithm", ["Random Forest", "KNN"])

if model_name == "Random Forest":
    param = st.sidebar.slider("Number of estimators (n_trees)", 1, 100, 50)
    clf = RandomForestClassifier(n_estimators=param)
else:
    param = st.sidebar.slider("Number of neighbors (K)", 1, 15, 3)
    clf = KNeighborsClassifier(n_neighbors=param)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10))

with col2:
    st.subheader("📈 Data Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=iris.feature_names[0], y=iris.feature_names[1], hue='species', ax=ax)
    st.pyplot(fig)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.divider()
st.subheader(f"🏆 Prediction Results ({model_name})")
st.metric(label="Model Accuracy", value=f"{acc*100:.2f}%")