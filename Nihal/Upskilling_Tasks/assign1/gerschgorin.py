import numpy as np
import matplotlib.pyplot as plt
def f(x):
    return (np.pow(x,5)-13*np.pow(x,3)-np.pow(x,2)+10*x+170)

def abs_adder(array, ind):
    res = 0
    for i in range(len(array)):
        if i!=ind:
            res += np.abs(array[i])
    return res
dx = 1e-3
x = np.arange(-4, 4, dx)
y = f(x)
plt.figure()
plt.plot(x,y)
plt.grid(True)
plt.axhline(0, color='red', linewidth=1)
plt.xlim(-4, 4)
plt.figure()
poly = [170, 10, -1, -13, 0, 1]
comp_mat = np.zeros((5,5))
for i in range(5):
    if i<4:
        comp_mat[i+1][i] = 1
    comp_mat[i][4] = -poly[i]
eigvals = np.linalg.eigvals(comp_mat)
plt.scatter(eigvals.real, eigvals.imag, c = 'red')
for i in range(1,5):
    centre = (comp_mat[i][i].real, comp_mat[i][i].imag)
    r = abs_adder(comp_mat[i], i)
    circle = plt.Circle(centre, radius = r, fill = False)
    plt.gca().add_patch(circle)
plt.axhline(0, color = 'black')
plt.axvline(0, color = 'black')
plt.show()