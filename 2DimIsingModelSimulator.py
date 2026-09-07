import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# スピン格子の初期化
def init_lattice(L):
    return np.random.choice([1, -1], size=(L, L))

# 1ステップの更新（メトロポリス法）
def update_lattice(lattice, T):
    L = lattice.shape[0]
    for _ in range(L*L): # 格子点と同じ数だけ試行
        i, j = np.random.randint(0, L, 2)
        # 近傍スピンとの相互作用エネルギー変化
        # 周期境界条件（端がつながっている）を考慮
        S = lattice[i, j]
        neighbors = (lattice[(i+1)%L, j] + lattice[(i-1)%L, j] +
                     lattice[i, (j+1)%L] + lattice[i, (j-1)%L])
        dE = 2 * S * neighbors

        # 反転条件の判定
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            lattice[i, j] *= -1
    return lattice

st.title("2Dイジング模型シミュレーター")

# サイドバーでパラメータ設定
L = st.sidebar.slider("格子サイズ L", 10, 50, 20)
T = st.sidebar.slider("温度 T", 0.1, 5.0, 2.27) # 2.27付近が転移点
steps = st.sidebar.number_input("更新回数", 1, 100, 10)

# セッション状態で格子を保持（再実行してもリセットされないように）
if 'lattice' not in st.session_state or st.sidebar.button("リセット"):
    st.session_state.lattice = init_lattice(L)

# シミュレーション実行ボタン
if st.button("シミュレーション開始"):
    progress_bar = st.progress(0)
    img_plot = st.empty() # 画像更新用の空枠

    for s in range(steps):
        st.session_state.lattice = update_lattice(st.session_state.lattice, T)

        # 可視化
        fig, ax = plt.subplots()
        ax.imshow(st.session_state.lattice, cmap='binary')
        ax.axis('off')
        img_plot.pyplot(fig)
        plt.close(fig)

        progress_bar.progress((s + 1) / steps)