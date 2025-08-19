import sys
import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(n_points):
    # sample (x,y) uniformly in [0,1]^2
    x = np.random.rand(n_points)
    y = np.random.rand(n_points)
    # count how many fall within the unit quarter‐circle x^2+y^2 <= 1
    inside = np.sum(x*x + y*y <= 1.0)
    return 4.0 * inside / n_points

def main(sample_sizes):
    results = []
    for n in sample_sizes:
        pi_est = estimate_pi(int(n))
        error = abs(np.pi - pi_est)
        results.append((n, pi_est, error))
        print(f"N={int(n):>8,} → π≈{pi_est:.6f}   error={error:.6f}")

    # Plotting
    sizes = [r[0] for r in results]
    estimates = [r[1] for r in results]

    plt.figure(figsize=(8,5))
    plt.plot(sizes, estimates, marker='o', linestyle='-')
    plt.axhline(np.pi, color='k', linestyle='--', label='True π')
    plt.xscale('log')
    plt.xlabel("Sample Size (log scale)")
    plt.ylabel("Estimated π")
    plt.title("Monte Carlo π Estimation")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # default sample sizes if none given
    if len(sys.argv) > 1:
        sizes = [float(arg) for arg in sys.argv[1:]]
    else:
        sizes = [1_000, 10_000, 100_000]
    main(sizes)
