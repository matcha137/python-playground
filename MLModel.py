import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

st.title("SVM 決定境界シミュレーター")

# --- 入力層: パラメータ設定 ---
st.sidebar.header("パラメータ")
noise = st.sidebar.slider("データのノイズ", 0.0, 0.5, 0.2)
c_param = st.sidebar.select_slider("C (正則化)", options=[0.01, 1, 100])
gamma = st.sidebar.select_slider("Gamma (カーネルの広がり)", options=[0.1, 1, 10])

# --- 計算層: データ生成と学習 ---
X, y = make_moons(n_samples=200, noise=noise, random_state=42)
model = SVC(C=c_param, gamma=gamma)
model.fit(X, y)

# --- 出力層: 可視化 ---
xx, yy = np.meshgrid(np.linspace(-2, 3, 100), np.linspace(-1.5, 2, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k")
ax.set_title(f"SVM Boundary (C={c_param}, gamma={gamma})")

st.pyplot(fig)