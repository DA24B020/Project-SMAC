# q2_fit_cubic.py

import numpy as np
import matplotlib.pyplot as plt

# 1. Generate x in [0,1]
n_points = 100
x = np.linspace(0, 1, n_points)

# 2. Define original cubic f(x) = a x^3 + b x^2 + c x + d
a, b, c, d = 2.5, -1.3,  4.0, 0.5
y_true = a*x**3 + b*x**2 + c*x + d

# 3. Add uniform noise in [-δ, +δ]
delta = 0.02
noise = np.random.uniform(-delta, delta, size=n_points)
y_noisy = y_true + noise

# 4. Fit a 3rd‑degree polynomial to (x, y_noisy)
#    np.polyfit returns [â, b̂, ĉ, d̂]
fitted_coeffs = np.polyfit(x, y_noisy, deg=3)
a_hat, b_hat, c_hat, d_hat = fitted_coeffs

# Generate fitted curve
y_fitted = np.polyval(fitted_coeffs, x)

# 5. Print comparison of constants
print("Original constants:")
print(f"  a = {a:.4f}, b = {b:.4f}, c = {c:.4f}, d = {d:.4f}")
print("Fitted constants:")
print(f"  â = {a_hat:.4f}, b̂ = {b_hat:.4f}, ĉ = {c_hat:.4f}, d̂ = {d_hat:.4f}")

# 6. Plot everything
plt.figure(figsize=(8,5))
plt.plot(x, y_true,   linewidth=2, label='Original f(x)')
plt.scatter(x, y_noisy, s=20,       label='Noisy samples', alpha=0.6)
plt.plot(x, y_fitted, linestyle='--', linewidth=2, label='Fitted cubic')
plt.title('Q2: Original vs. Noisy vs. Fitted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()
