import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 定数
L = 10.0      # 箱のサイズ
N = 512       # 分割数
dx = L / N
x = np.linspace(0, L, N, endpoint=False)

st.title("量子力学：箱の中の波束シミュレーション")

# パラメータ設定
st.sidebar.header("初期状態の設定")
x0 = st.sidebar.slider("初期位置 (x0)", 2.0, 8.0, 5.0)
k0 = st.sidebar.slider("初期運動量 (k0)", -20.0, 20.0, 5.0)
sigma = st.sidebar.slider("波束の広がり (sigma)", 0.2, 1.0, 0.5)
dt = 0.01

# 初期波動関数の作成（ガウス波束）
def init_wave(x, x0, k0, sigma):
    psi = np.exp(-0.5 * ((x - x0) / sigma)**2) * np.exp(1j * k0 * x)
    return psi / np.sqrt(np.sum(np.abs(psi)**2) * dx)

if 'psi' not in st.session_state or st.sidebar.button("リセット"):
    st.session_state.psi = init_wave(x, x0, k0, sigma)

# 計算エンジン（スプリット・ステップ法）
def step(psi, dt):
    # 1. 運動量空間での位相回転 (キネティック・エネルギー)
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    psi_k = np.fft.fft(psi)
    psi_k *= np.exp(-0.5j * (k**2) * dt)
    psi = np.fft.ifft(psi_k)
    # 本来はポテンシャルV(x)の項が必要だが、箱の中(V=0)なので省略
    return psi

# 実行
if st.button("時間を進める"):
    img_plot = st.empty()
    for t in range(50):
        st.session_state.psi = step(st.session_state.psi, dt)

        # 可視化
        prob_density = np.abs(st.session_state.psi)**2
        fig, ax = plt.subplots()
        ax.plot(x, prob_density, color='blue', label='Probability Density')
        ax.set_ylim(0, 1.5)
        ax.set_xlim(0, L)
        ax.set_xlabel("Position (x)")
        ax.set_ylabel("|ψ|^2")
        ax.fill_between(x, prob_density, color='blue', alpha=0.3)
        img_plot.pyplot(fig)
        plt.close(fig)