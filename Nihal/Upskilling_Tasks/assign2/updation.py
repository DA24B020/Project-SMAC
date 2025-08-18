import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

T = np.zeros((100,100))
center = np.array([49,49])
for i in range(100):
    for j in range(100):
        point = np.array([i,j])
        if np.linalg.norm(point-center)<25:
            T[i][j] = 100


param = 0.2
n = 10

for _ in range(n):
    for i in range(1,99):
        for j in range(1,99):
            T[i][j] += param*(T[i+1][j]+T[i-1][j]+T[i][j+1]+T[i][j-1])
    for j in range(100):
        T[0][j] = T[1][j]
        T[99][j] = T[98][j]
    for i in range(100):
        T[i][0] = T[i][1]
        T[i][99] = T[i][98]

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x,y = np.meshgrid(np.arange(100), np.arange(100))

ax.plot_surface(x+1, y+1, T)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()