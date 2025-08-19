set -euo pipefail

python3 <<'PYCODE'
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import sympy as sp

x_data = np.array([0, 1, 2, 3, 4, 5, 9])
y_data = np.array([2.5, 0.5, 0.5, 2, 2, 1.5, 0])

cs = CubicSpline(x_data, y_data, bc_type='natural')

x_fine = np.linspace(0, 9, 300)
y_spline = cs(x_fine)
coeffs = cs.c
for i in range(len(x_data) - 1):
    a, b, c, d = coeffs[:, i]
    print(f"  Segment [{x_data[i]}, {x_data[i+1]}]: {a:.6f}x³ + {b:.6f}x² + {c:.6f}x + {d:.6f}")

degree = len(x_data) - 1  # degree 6 for 7 points
poly_coeffs = np.polyfit(x_data, y_data, degree)
poly_func = np.poly1d(poly_coeffs)

y_poly = poly_func(x_fine)
poly_str = ""
for i, coeff in enumerate(poly_coeffs):
    power = degree - i
    if i == 0:
        poly_str += f"{coeff:.6f}x^{power}"
    else:
        sign = "+" if coeff >= 0 else "-"
        if power > 1:
            poly_str += f" {sign} {abs(coeff):.6f}x^{power}"
        elif power == 1:
            poly_str += f" {sign} {abs(coeff):.6f}x"
        else:
            poly_str += f" {sign} {abs(coeff):.6f}"
print(f"  p(x) = {poly_str}")

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(x_fine, y_spline, 'b-', linewidth=2, label='Natural Cubic Spline')
plt.plot(x_fine, y_poly, 'r--', linewidth=2, label='Polynomial Interpolation')
plt.plot(x_data, y_data, 'ko', markersize=8, label='Data Points')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Natural Cubic Spline vs Polynomial Interpolation')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
zoom_mask = (x_fine >= 0) & (x_fine <= 6)
plt.plot(x_fine[zoom_mask], y_spline[zoom_mask], 'b-', linewidth=2, label='Cubic Spline')
plt.plot(x_fine[zoom_mask], y_poly[zoom_mask], 'r--', linewidth=2, label='Polynomial')
plt.plot(x_data[x_data <= 6], y_data[x_data <= 6], 'ko', markersize=8, label='Data Points')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Zoomed View (x = 0 to 6)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
difference = y_spline - y_poly
plt.plot(x_fine, difference, 'g-', linewidth=2)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xlabel('x')
plt.ylabel('Spline - Polynomial')
plt.title('Difference Between Methods')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
y_spline_deriv = cs(x_fine, 1)
# Polynomial derivatives
poly_deriv = np.polyder(poly_func)
y_poly_deriv = poly_deriv(x_fine)

plt.plot(x_fine, y_spline_deriv, 'b-', linewidth=2, label="Spline f'(x)")
plt.plot(x_fine, y_poly_deriv, 'r--', linewidth=2, label="Polynomial f'(x)")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('First Derivatives Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Q4_interpolation_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

spline_errors = cs(x_data) - y_data
poly_errors = poly_func(x_data) - y_data

print("Interpolation errors at data points:")
print("  Cubic Spline max error:", np.max(np.abs(spline_errors)))
print("  Polynomial max error:", np.max(np.abs(poly_errors)))

# Calculate smoothness measures
print(f"\nSmoothness comparison:")
print(f"  Cubic spline is C² continuous (smooth up to 2nd derivative)")
print(f"  Polynomial is C^∞ continuous (infinitely smooth)")

# Oscillation analysis
max_diff = np.max(np.abs(difference))
print(f"\nMaximum difference between methods: {max_diff:.6f}")

# Behavior near boundaries
print(f"\nBoundary behavior:")
print(f"  At x=0: Spline={cs(0):.3f}, Polynomial={poly_func(0):.3f}")
print(f"  At x=9: Spline={cs(9):.3f}, Polynomial={poly_func(9):.3f}")

# Extrapolation warning
x_extrap = np.array([10, 11])
spline_extrap = cs(x_extrap)
poly_extrap = poly_func(x_extrap)
print(f"\nExtrapolation at x=10:")
print(f"  Spline: {spline_extrap[0]:.3f}")
print(f"  Polynomial: {poly_extrap[0]:.3f}")
print("\n=== MATHEMATICAL VALIDATION ===")

# Check natural boundary conditions for spline
second_deriv_start = cs(x_data[0], 2)
second_deriv_end = cs(x_data[-1], 2)
print(f"Natural boundary conditions:")
print(f"  f''({x_data[0]}) = {second_deriv_start:.2e} (should be ≈ 0)")
print(f"  f''({x_data[-1]}) = {second_deriv_end:.2e} (should be ≈ 0)")

PYCODE
