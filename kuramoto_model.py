import numpy as np
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""Update oscillator phase (Kuramoto model)"""
def update_phases(phases, N, K):
    # 行列演算で位相差を一括計算
    phase_mat = np.tile(phases, (N, 1))
    coupling = (K / N) * np.sum(np.sin(phase_mat.T - phase_mat), axis=1)
    new_phases = phases + coupling
    return np.mod(new_phases, 2 * np.pi)

"""Generate positions randomly"""
def make_positions_random(N, L): # 引数 N, L を追加
    pos_x_t = np.random.uniform(0, L, N)
    pos_y_t = np.random.uniform(0, L, N)
    return pos_x_t, pos_y_t

"""Generate animation"""
def plot_animation(timerange, pos_x, pos_y, phase, L, folderpath):
    fig_scale = 0.1
    # 描画サイズを調整 (L=100だと大きすぎる場合があるため調整)
    fig = plt.figure(figsize=(L*fig_scale, L*fig_scale), dpi=100.0)
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.set_aspect('equal')
    ax1.set_xlim(0, L); ax1.set_ylim(0, L)

    all_ims = []
    # カラーバーの初期化用
    im_dummy = ax1.scatter([], [], c=[], cmap=cm.seismic, vmin=0, vmax=2 * np.pi)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im_dummy, cax=cax)

    print("Generating animation frames...")
    for s in tqdm(range(timerange)):
        ims = []
        # scatterの描画
        im = ax1.scatter(pos_x, pos_y, s=60, c=phase[s], cmap=cm.seismic,
                         vmin=0, vmax=2 * np.pi,
                         linewidths=0.5, edgecolors='grey', animated=True)

        # Step数のテキスト
        im_text = ax1.text(L/2 - 8, L + 2, f'Step={s}', size=15, animated=True)
        ims.append(im)
        ims.append(im_text)
        all_ims.append(ims)

    ani = animation.ArtistAnimation(fig, all_ims, interval=50, repeat=True)
    plt.show()

    save_path = f'{folderpath}/Movie.mp4'
    print(f"Saving movie to {save_path}...")
    # ffmpegがインストールされていない場合は writer="pillow" などに変更して .gif で保存してください
    try:
        ani.save(save_path, writer="ffmpeg", fps=20)
    except Exception as e:
        print(f"FFmpeg save failed, saving as GIF instead. Error: {e}")
        ani.save(f'{folderpath}/Movie.gif', writer="pillow", fps=20)

if __name__ == "__main__":
    # Parameters
    K, delta, N, T, L = 0.1, 0.5, 100, 1000, 100

    # Fix seed
    np.random.seed(3)

    # Generate initial positions and phases
    phase_t = np.zeros((T, N))
    phase_t[0] = np.random.uniform(0, 2 * np.pi, N)
    pos_x_t, pos_y_t = make_positions_random(N, L) # 引数を渡す

    phase_diffs, sync_step = [], -1
    folderpath = './SaveKuramoto'
    os.makedirs(folderpath, exist_ok=True)

    # Start simulation
    print("Simulating...")
    for t in tqdm(range(1, T)):
        # Update phases
        phase = update_phases(phase_t[t-1], N, K)
        phase_t[t] = phase

        # Calculate evaluation function (ベクトル演算で高速化)
        # 位相差の平方平均を計算
        diff = phase[:, np.newaxis] - phase[np.newaxis, :]
        avg_phase_diff = np.sqrt(np.mean(np.sin(diff/2)**2)) # 統計力学で一般的な指標

        phase_diffs.append(avg_phase_diff)
        if avg_phase_diff < delta and sync_step == -1:
            sync_step = t

    print(f'Synchronization achieved at timestep {sync_step}')

    # --- Plotting ---
    # Position
    plt.figure(figsize=(6, 6))
    plt.scatter(pos_x_t, pos_y_t)
    plt.title('Initial Position')
    plt.xlim(0, L); plt.ylim(0, L)
    plt.grid()
    plt.savefig(f'{folderpath}/Position.png')
    plt.close()

    # Phase
    plt.figure()
    plt.plot(phase_t)
    plt.title('Phase Evolution')
    plt.savefig(f'{folderpath}/Phase.png')
    plt.close()

    # Error
    plt.figure()
    plt.plot(phase_diffs)
    plt.title('Average Phase Difference')
    plt.savefig(f'{folderpath}/Error.png')
    plt.close()

    # Animation generation (400ステップ分)
    plot_animation(400, pos_x_t, pos_y_t, phase_t, L, folderpath)