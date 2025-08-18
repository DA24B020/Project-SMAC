import matplotlib.pyplot as plt
import numpy as np
def draw_arrow(start, end):
    dx, dy = end - start
    plt.arrow(start[0], start[1], dx, dy, color='red',
              head_width=1, length_includes_head=True)

pointsList = np.random.rand(20,2)*100
pointsList = np.array(sorted(pointsList, key= lambda x: np.linalg.norm(x)))
with open("pointslist.txt", mode='w') as f:
    for i in pointsList:
        f.write(f"({i[0]},{i[1]})\n")
plt.scatter(pointsList[:,0], pointsList[:,1])
disp = [[np.linalg.norm(i-j) if not np.array_equal(i, j) else np.inf for i in pointsList] for j in pointsList]
for i,norm in enumerate(disp):
    temp1 = norm.index(min(norm))
    norm[temp1] = np.inf
    temp2 = norm.index(min(norm))
    norm[temp2] = np.inf
    temp3 = norm.index(min(norm))
    draw_arrow(pointsList[i], pointsList[temp1])
    draw_arrow(pointsList[i], pointsList[temp2])
    draw_arrow(pointsList[i], pointsList[temp3])
plt.grid(True)
plt.xlim(0,100)
plt.ylim(0,100)
plt.show()
