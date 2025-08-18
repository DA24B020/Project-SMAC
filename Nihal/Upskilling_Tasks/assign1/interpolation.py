from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

class PolynomialInterpolation:
    def __init__(self, x_arr ,y_arr):
        self.x = x_arr
        self.y = y_arr
        n = len(x_arr)
        coeff = np.zeros(n)
        for i in range(n):
            roots = np.delete(self.x, i)
            num = np.poly(roots)
            den = np.prod(self.x[i]-roots)
            num = num*(y[i]/den)
            coeff += num
        self.poly = np.poly1d(coeff)
    def __call__(self, x):
        return self.poly(x)


x = np.array([0,1,2,3,4,5,9])
y = np.array([2.5, 0.5, 0.5, 2, 2, 1.5, 0])

ncs = CubicSpline(x, y, bc_type = 'natural', extrapolate = True)
pi = PolynomialInterpolation(x,y)
plt.scatter(x,y)
x_plot= np.arange(0,9,0.01)

plt.plot(x_plot, ncs(x_plot), label = "Natural Cubic Spine")
plt.plot(x_plot, pi(x_plot), label = 'Polynomial Interpolation')
plt.legend()
plt.grid(True)
plt.show()