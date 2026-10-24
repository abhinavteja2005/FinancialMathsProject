import numpy as np
import matplotlib.pyplot as plt

# Parameters
theta = 5.0    # mean-reversion speed
mu = 0.0       # long-run mean
sigma = 0.5    # volatility
X0 = 2.0       # start far from mu to see the pull
T = 3.0
N = 3000
dt = T / N
n_paths = 5

t = np.linspace(0, T, N + 1)

# Plot 1: Sample Paths
plt.figure(figsize=(10, 5))

for i in range(n_paths):
    X = np.zeros(N + 1)
    X[0] = X0
    for k in range(N):
        Z = np.random.randn()
        X[k + 1] = X[k] + theta * (mu - X[k]) * dt + sigma * np.sqrt(dt) * Z
    plt.plot(t, X, linewidth=0.9)

plt.axhline(mu, color="black", linestyle="--", linewidth=1.5, label=f"Long-run mean μ = {mu}")
plt.title("Ornstein-Uhlenbeck Process (Mean-Reverting)")
plt.xlabel("Time (t)")
plt.ylabel("X(t)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ou_paths.png", dpi=150)
plt.show()

# Plot 2: Stationary Distribution
# Run many paths to T, collect final values
n_mc = 10000
X_final = np.zeros(n_mc)
for j in range(n_mc):
    X_val = X0
    for k in range(N):
        X_val += theta * (mu - X_val) * dt + sigma * np.sqrt(dt) * np.random.randn()
    X_final[j] = X_val

plt.figure(figsize=(8, 5))
plt.hist(X_final, bins=80, density=True, alpha=0.6, edgecolor="white", label="Simulated X(T)")

# Theoretical stationary distribution: N(mu, sigma^2 / (2*theta))
stat_var = sigma**2 / (2 * theta)
x = np.linspace(mu - 4 * np.sqrt(stat_var), mu + 4 * np.sqrt(stat_var), 300)
pdf = (1 / np.sqrt(2 * np.pi * stat_var)) * np.exp(-(x - mu)**2 / (2 * stat_var))
plt.plot(x, pdf, "r-", linewidth=2, label=f"Stationary N({mu}, {stat_var:.3f})")

plt.title(f"Stationary Distribution at T={T}")
plt.xlabel("X(T)")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ou_distribution.png", dpi=150)
plt.show()