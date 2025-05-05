import ctypes
import time
import os

def run_kernel(name: str, N=1_000_000):
    print(f"Running kernel: {name}")
    so_path = f"./build/{name}.so"
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, f"run_{name}")
    func.argtypes = [ctypes.c_int]

    start = time.time()
    func(N)
    end = time.time()

    runtime_ms = (end - start) * 1000
    throughput = (N * 3 * 4) / (runtime_ms / 1000) / 1e9  # GB/s

    print(f"Runtime: {runtime_ms:.3f} ms, Throughput: {throughput:.2f} GB/s")

    return {"kernel": name, "runtime_ms": runtime_ms, "throughput_gbps": throughput}

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    results = []

    for N in [int(1e5), int(1e6), int(1e7), int(1e8), int(1e9)]:
        for kernel in ["baseline", "shared_mem", "tiled", "tiled_shared"]:
            results.append(run_kernel(kernel, N))

    with open("results/log.csv", "w") as f:
        f.write("kernel,runtime_ms,throughput_gbps\n")
        for r in results:
            f.write(f"{r['kernel']},{r['runtime_ms']:.3f},{r['throughput_gbps']:.2f}\n")


#sudo /usr/local/NVIDIA-Nsight-Compute/ncu /home/dako/miniconda3/bin/python benchmark.py