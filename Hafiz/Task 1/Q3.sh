set -euo pipefail

python3 <<'PYCODE'
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x = sp.symbols('x')
f = x**5 - 13*x**3 - x**2 + 10*x + 170

f_num = sp.lambdify(x, f, "numpy")

def companion_matrix_roots(coeffs):
    coeffs = [float(c) for c in coeffs]
    n = len(coeffs) - 1
    if coeffs[0] != 0:
        coeffs = [c/coeffs[0] for c in coeffs]
    C = np.zeros((n, n))
    for j in range(n):
        C[0, j] = -coeffs[j+1]
    for i in range(1, n):
        C[i, i-1] = 1.0
    eigenvalues = np.linalg.eigvals(C)
    return eigenvalues

def improved_bairstow(coeffs, max_iter=50, tol=1e-10):
    coeffs = [float(c) for c in coeffs]
    roots = []
    n = len(coeffs) - 1
    
    print(f"Starting improved Bairstow's method with degree {n}")

    initial_guesses = [
        (0.5, 0.5),   # Standard guess
        (-0.5, -0.5), # Negative guess
        (1.0, 1.0),   # Larger positive
        (-1.0, 1.0),  # Mixed signs
        (0.1, -0.1)   # Small mixed
    ]
    
    current_coeffs = coeffs[:]
    current_n = n
    
    while current_n > 2:
        found_root = False
        
        for r_init, s_init in initial_guesses:
            try:
                r, s = r_init, s_init
                
                for iteration in range(max_iter):
                    b = [0.0] * (current_n + 1)
                    c = [0.0] * current_n

                    b[0] = current_coeffs[0]
                    b[1] = current_coeffs[1] + r * b[0]
                    
                    for i in range(2, current_n + 1):
                        b[i] = current_coeffs[i] + r * b[i-1] + s * b[i-2]

                    c[0] = b[0]
                    c[1] = b[1] + r * c[0]
                    
                    for i in range(2, current_n):
                        c[i] = b[i] + r * c[i-1] + s * c[i-2]

                    if abs(b[current_n]) + abs(b[current_n-1]) < tol:
                        break
                    if current_n >= 3:
                        det = c[current_n-3]**2 - c[current_n-2] * (c[current_n-4] if current_n >= 4 else 0)
                    else:
                        det = c[0]**2
                    
                    if abs(det) < 1e-15:
                        break

                    dr = (-b[current_n-1] * c[current_n-3] + b[current_n] * (c[current_n-4] if current_n >= 4 else 0)) / det
                    ds = (-b[current_n] * c[current_n-3] + b[current_n-1] * c[current_n-2]) / det
                    
                    r += dr
                    s += ds
                    
                    if abs(dr) + abs(ds) < tol:
                        found_root = True
                        break
                
                if found_root:
                    discriminant = r**2 - 4*s
                    if discriminant >= 0:
                        root1 = (-r + np.sqrt(discriminant)) / 2
                        root2 = (-r - np.sqrt(discriminant)) / 2
                        roots.extend([root1, root2])
                        print(f"  Found real roots: {root1:.6f}, {root2:.6f}")
                    else:
                        real_part = -r / 2
                        imag_part = np.sqrt(-discriminant) / 2
                        root1 = complex(real_part, imag_part)
                        root2 = complex(real_part, -imag_part)
                        roots.extend([root1, root2])
                        print(f"  Found complex roots: {root1:.6f}, {root2:.6f}")

                    current_coeffs = b[:-2]
                    current_n -= 2
                    break
                    
            except (ZeroDivisionError, ValueError, OverflowError):
                continue  # Try next initial guess
        
        if not found_root:
            print(f"  Bairstow's method failed at degree {current_n}, switching to NumPy")
            # Fall back to NumPy for remaining polynomial
            remaining_roots = np.roots(current_coeffs)
            roots.extend(remaining_roots)
            break
    
    if current_n == 2:
        a, b_coef, c_coef = current_coeffs
        disc = b_coef**2 - 4*a*c_coef
        if disc >= 0:
            root1 = (-b_coef + np.sqrt(disc)) / (2*a)
            root2 = (-b_coef - np.sqrt(disc)) / (2*a)
            roots.extend([root1, root2])
        else:
            real_part = -b_coef / (2*a)
            imag_part = np.sqrt(-disc) / (2*a)
            roots.extend([complex(real_part, imag_part), complex(real_part, -imag_part)])
    elif current_n == 1:
        root = -current_coeffs[1] / current_coeffs[0]
        roots.append(root)
    
    return roots

poly = sp.Poly(f, x)
coeffs = poly.all_coeffs()
print("\n1. Companion Matrix Method:")
companion_roots = companion_matrix_roots(coeffs)
print("Companion matrix roots:")
for i, root in enumerate(companion_roots):
    print(f"  Root {i+1}: {root}")

print("\n2. Improved Bairstow's Method:")
try:
    bairstow_roots = improved_bairstow(coeffs)
    print("Bairstow's method roots:")
    for i, root in enumerate(bairstow_roots):
        print(f"  Root {i+1}: {root}")
except Exception as e:
    print(f"Bairstow's method failed: {e}")
    bairstow_roots = companion_roots  # Use companion matrix as fallback

print("\n3. NumPy roots (reference):")
numpy_roots = np.roots([float(c) for c in coeffs])
for i, root in enumerate(numpy_roots):
    print(f"  Root {i+1}: {root}")

print("\n4. SymPy roots (exact):")
sympy_roots = sp.nroots(f)
real_roots = [complex(r).real for r in sympy_roots if abs(sp.im(r)) < 1e-6]

for i, root in enumerate(sympy_roots):
    print(f"  Root {i+1}: {root}")

plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
xmin, xmax = -6, 6
x_vals = np.linspace(xmin, xmax, 600)
y_vals = f_num(x_vals)
plt.plot(x_vals, y_vals, 'b-', lw=2, label='$f(x)$')
plt.axhline(0, color='black', lw=0.7, alpha=0.7)
plt.axvline(0, color='black', lw=0.7, alpha=0.7)

for r in real_roots:
    plt.plot(r, 0, 'ro', markersize=10, label=f'Real root: {r:.3f}')

plt.title('Function and Real Roots')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
coeffs_norm = [float(c) for c in coeffs]
if coeffs_norm[0] != 1:
    coeffs_norm = [c/coeffs_norm[0] for c in coeffs_norm]
    
n = len(coeffs_norm) - 1
C = np.zeros((n, n))
for j in range(n):
    C[0, j] = -coeffs_norm[j+1]
for i in range(1, n):
    C[i, i-1] = 1.0

centers = np.diag(C)
radii = np.sum(np.abs(C), axis=1) - np.abs(centers)

theta = np.linspace(0, 2*np.pi, 300)
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i, (c, r) in enumerate(zip(centers, radii)):
    plt.plot(c + r*np.cos(theta), r*np.sin(theta), 
             color=colors[i], lw=2, alpha=0.7, label=f'Disc {i+1}')

for root in companion_roots:
    plt.plot(root.real, root.imag, 'ko', markersize=8)

plt.axhline(0, color='black', lw=0.5, alpha=0.7)
plt.axvline(0, color='black', lw=0.5, alpha=0.7)
plt.title('Gerschgorin Discs with Roots')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.gca().set_aspect('equal', 'box')
plt.grid(True, alpha=0.3)

methods = ['Companion', 'Bairstow', 'NumPy', 'SymPy']
root_sets = [companion_roots, bairstow_roots, numpy_roots, 
             [complex(float(sp.re(r)), float(sp.im(r))) for r in sympy_roots]]
colors = ['red', 'green', 'blue', 'black']
markers = ['o', '^', 's', 'x']

for i, (method, roots, color, marker) in enumerate(zip(methods, root_sets, colors, markers)):
    plt.subplot(2, 3, 3+i)
    real_parts = [r.real for r in roots]
    imag_parts = [r.imag for r in roots]
    plt.scatter(real_parts, imag_parts, c=color, s=100, alpha=0.7, marker=marker)
    plt.axhline(0, color='black', lw=0.5, alpha=0.5)
    plt.axvline(0, color='black', lw=0.5, alpha=0.5)
    plt.title(f'{method} Method Roots')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Q3_comprehensive_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

methods = ['Companion Matrix', 'Bairstow', 'NumPy']
root_sets = [companion_roots, bairstow_roots, numpy_roots]

for method, roots in zip(methods, root_sets):
    print(f"\n{method} vs SymPy:")
    if len(roots) == len(sympy_roots):
        total_error = 0
        for i, (sym_root, method_root) in enumerate(zip(sympy_roots, roots)):
            sym_complex = complex(float(sp.re(sym_root)), float(sp.im(sym_root)))
            method_complex = complex(method_root) if hasattr(method_root, 'real') else complex(method_root, 0)
            error = abs(sym_complex - method_complex)
            total_error += error
            print(f"  Root {i+1} error: {error:.2e}")
        print(f"  Average error: {total_error/len(roots):.2e}")

PYCODE