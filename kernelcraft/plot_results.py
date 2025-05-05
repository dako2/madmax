#!/usr/bin/env python3
"""
plot_results.py - Plot benchmark results
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv("results/log.csv")

# Constants (for RTX 3090)
peak_memory_bw = 936       # GB/s
peak_compute = 35600       # GFLOP/s
bytes_per_element = 4 * 3  # float A + B + C = 3 * 4 bytes

# Estimate operational intensity and performance
df["intensity"] = df["throughput_gbps"] * 1e3 / (bytes_per_element)  # MB/s -> MB
df["perf"] = df["compute_percent"] / 100 * peak_compute

# Roofline plot
x = np.logspace(-1, 4, 500)
y_mem = np.minimum(x * peak_memory_bw, peak_compute)
y_peak = np.full_like(x, peak_compute)

plt.figure(figsize=(10, 6))
plt.plot(x, y_mem, label="Memory BW Limit", linestyle="--", color="red")
plt.plot(x, y_peak, label="Compute Limit", linestyle="--", color="green")

# Scatter points
for i, row in df.iterrows():
    plt.scatter(row["intensity"], row["perf"], label=row["kernel"], s=100)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Operational Intensity (FLOPs / byte)")
plt.ylabel("Performance (GFLOP/s)")
plt.title("Roofline Model - GPU Kernel Performance")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("results/roofline_chart.png")
plt.show()
