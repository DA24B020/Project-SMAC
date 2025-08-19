import numpy as np
import matplotlib.pyplot as plt

N = 20
BOX_SIZE = 100

points = np.random.rand(N, 2) * BOX_SIZE

dists = np.hypot(points[:,0], points[:,1])
order = np.argsort(dists)
sorted_pts = points[order]

with open("Q2.txt", "w") as f:
    for x, y in sorted_pts:
        f.write(f"{x:.6f} {y:.6f}\n")

from scipy.spatial import distance_matrix
dist_mat = distance_matrix(points, points)

neighbors = []
for i in range(N):
    nn = np.argsort(dist_mat[i])[1:4]
    neighbors.append(nn)

plt.figure(figsize=(6,6))
plt.scatter(points[:,0], points[:,1], c='blue', s=50, zorder=2)
for i, nbrs in enumerate(neighbors):
    x0, y0 = points[i]
    for j in nbrs:
        x1, y1 = points[j]
        plt.plot([x0, x1], [y0, y1], 'k-', lw=0.8, zorder=1)

plt.xlim(0, BOX_SIZE)
plt.ylim(0, BOX_SIZE)
plt.title("20 Random Points & Their 3 Nearest Neighbors")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().set_aspect('equal', 'box')
plt.tight_layout()
plt.savefig("Q2.png", dpi=150)
plt.close()
