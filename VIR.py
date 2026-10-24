import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 0.5        # speed of mean reversion
b = 0.05       # long-run mean rate (5%)
sigma = 0.02   # volatility
r0 = 0.08      # initial rate (8%)
T = 10.0       # 10 years
N = 5000
dt = T / N
n_paths = 5

t = np.linspace(0, T, N + 1)

# Plot 1: Sample Paths
plt.figure(figsize=(10, 5))

for i in range(n_paths):
    r = np.zeros(N + 1)
    r[0] = r0
    for k in range(N):
        Z = np.random.randn()
        r[k + 1] = r[k] + a * (b - r[k]) * dt + sigma * np.sqrt(dt) * Z
    plt.plot(t, r * 100, linewidth=0.9)   # multiply by 100 for percentage

plt.axhline(b * 100, color="black", linestyle="--", linewidth=1.5, label=f"Long-run mean b = {b*100:.1f}%")
plt.title("Vasicek Interest Rate Model")
plt.xlabel("Time (years)")
plt.ylabel("Interest Rate (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("vasicek_paths.png", dpi=150)
plt.show()

# Plot 2: Terminal Distribution
n_mc = 10000
r_T = np.zeros(n_mc)
for j in range(n_mc):
    r_val = r0
    for k in range(N):
        r_val += a * (b - r_val) * dt + sigma * np.sqrt(dt) * np.random.randn()
    r_T[j] = r_val

plt.figure(figsize=(8, 5))
plt.hist(r_T * 100, bins=80, density=True, alpha=0.6, edgecolor="white", label="Simulated r(T)")

# Analytical: E[r(T)] = b + (r0 - b)*exp(-a*T)
#             Var[r(T)] = (sigma^2 / (2a)) * (1 - exp(-2aT))
mean_rT = b + (r0 - b) * np.exp(-a * T)
var_rT = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * T))

x = np.linspace((mean_rT - 4*np.sqrt(var_rT))*100, (mean_rT + 4*np.sqrt(var_rT))*100, 300)
pdf = (1 / np.sqrt(2 * np.pi * var_rT * 10000)) * np.exp(-(x - mean_rT*100)**2 / (2 * var_rT * 10000))
plt.plot(x, pdf, "r-", linewidth=2, label=f"N({mean_rT*100:.2f}%, {np.sqrt(var_rT)*100:.2f}%)")

plt.title(f"Distribution of r(T={int(T)} yrs)")
plt.xlabel("Interest Rate (%)")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("vasicek_distribution.png", dpi=150)
plt.show()