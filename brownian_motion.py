import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1.0          # total time
N = 1000         # number of time steps
dt = T / N       # size of each step
n_paths = 5      # number of sample paths

# Time array: [0, dt, 2*dt, ..., T]
t = np.linspace(0, T, N + 1)

# Plot 1: Sample Paths
plt.figure(figsize=(10, 5))

for i in range(n_paths):
    # Step 1: draw N random samples from N(0,1)
    Z = np.random.randn(N)

    # Step 2: scale by sqrt(dt) to get increments dW
    dW = np.sqrt(dt) * Z

    # Step 3: cumulative sum gives the path, start from 0
    W = np.concatenate(([0], np.cumsum(dW)))

    plt.plot(t, W, linewidth=0.9)

plt.title("Brownian Motion - Sample Paths")
plt.xlabel("Time (t)")
plt.ylabel("W(t)")
plt.axhline(0, color="grey", linestyle="--", linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("bm_paths.png", dpi=150)
plt.show()

# Plot 2: Terminal Distribution
# Run 10000 paths, collect only the final value W(T)
n_mc = 10000
W_T = np.sqrt(T) * np.random.randn(n_mc)  # W(T) ~ N(0, T), so this is a shortcut

plt.figure(figsize=(8, 5))
plt.hist(W_T, bins=80, density=True, alpha=0.6, edgecolor="white", label="Simulated W(T)")

# Theoretical N(0, T) density for comparison
x = np.linspace(-4, 4, 300)
pdf = (1 / np.sqrt(2 * np.pi * T)) * np.exp(-x**2 / (2 * T))
plt.plot(x, pdf, "r-", linewidth=2, label=f"Theoretical N(0, {T})")

plt.title(f"Distribution of W(T={T}) from {n_mc} paths")
plt.xlabel("W(T)")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("bm_distribution.png", dpi=150)
plt.show()