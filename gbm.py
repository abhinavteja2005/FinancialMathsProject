import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 100       # initial stock price
mu = 0.10      # drift (expected return)
sigma = 0.30   # volatility
T = 1.0        # 1 year
N = 1000       # time steps
dt = T / N
n_paths = 5

t = np.linspace(0, T, N + 1)

# Plot 1: Sample Paths 
plt.figure(figsize=(10, 5))

for i in range(n_paths):
    # Generate Brownian motion path
    Z = np.random.randn(N)
    dW = np.sqrt(dt) * Z
    W = np.concatenate(([0], np.cumsum(dW)))

    # GBM exact solution: plug W(t) into the formula
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

    plt.plot(t, S, linewidth=0.9)

# Analytical expected value E[S(t)] = S0 * exp(mu * t)
mean_line = S0 * np.exp(mu * t)
plt.plot(t, mean_line, "k--", linewidth=1.5, label=f"E[S(t)] = {S0}·exp({mu}t)")

plt.title("Geometric Brownian Motion - Sample Paths")
plt.xlabel("Time (t)")
plt.ylabel("S(t)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gbm_paths.png", dpi=150)
plt.show()

# Plot 2: Terminal Distribution (log-normal)
n_mc = 10000
W_T = np.sqrt(T) * np.random.randn(n_mc)
S_T = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)

plt.figure(figsize=(8, 5))
plt.hist(S_T, bins=80, density=True, alpha=0.6, edgecolor="white", label="Simulated S(T)")
plt.axvline(np.mean(S_T), color="k", linestyle="--", linewidth=1, label=f"Mean = {np.mean(S_T):.1f}")

plt.title(f"Distribution of S(T={T}) - Log-Normal")
plt.xlabel("S(T)")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gbm_distribution.png", dpi=150)
plt.show()