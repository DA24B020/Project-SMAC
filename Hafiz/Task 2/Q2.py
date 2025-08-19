import argparse
import numpy as np
import matplotlib.pyplot as plt

def initialize_T(N=100, radius=25, T_hot=100.0):
    T = np.zeros((N, N), dtype=float)
    cx = cy = (N - 1) / 2.0
    y, x = np.ogrid[0:N, 0:N]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    T[mask] = T_hot
    return T

def diffuse(T, lam):
    Tn = T.copy()
    Tn[1:-1,1:-1] = (
        T[1:-1,1:-1]
        + lam * (
            T[2:  ,1:-1] + T[ :-2,1:-1]
          + T[1:-1,2:  ] + T[1:-1, :-2]
          - 4*T[1:-1,1:-1]
        )
    )
    Tn[:,  0] = Tn[:,  1]  
    Tn[:, -1] = Tn[:, -2]   
    Tn[ 0, :] = Tn[ 1, :]   
    Tn[-1, :] = Tn[-2, :]   
    return Tn

def plot_snapshots(snapshots, lam):
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (step, T) in zip(axes, snapshots):
        pcm = ax.pcolor(T, cmap='hot', vmin=0, vmax=100)
        ax.set_title(f"Step {step}")
        ax.set_aspect('equal')
        ax.invert_yaxis()
    fig.suptitle(f"2D Diffusion (λ={lam})")
    fig.colorbar(pcm, ax=axes, label='T')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", type=float, default=0.2, dest="lam",
                        help="Diffusion coefficient λ (≤0.25 for stability)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Total number of diffusion steps")
    parser.add_argument("--snapshots", type=int, nargs="+", default=[0, 10, 100],
                        help="Time‑steps at which to plot T")
    args = parser.parse_args()
    T = initialize_T(N=100, radius=25, T_hot=100.0)
    snaps = []
    for step in range(args.steps + 1):
        if step in args.snapshots:
            snaps.append((step, T.copy()))
        T = diffuse(T, args.lam)
    plot_snapshots(snaps, args.lam)

if __name__ == "__main__":
    main()
