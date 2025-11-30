# performance_comparison.py

import time
import os
import matplotlib.pyplot as plt
from des_implementation_final import des_encrypt_block, generate_round_keys  # Import from des_implementation
from Crypto.Cipher import DES  # Import library DES

def measure_library_des_performance(key):
    cipher = DES.new(key, DES.MODE_ECB)

    sizes = [2**i for i in range(10, 27)]  # Message sizes: 2^10, 2^11, ..., 2^26 bytes
    times = []
    throughputs = []

    for size in sizes:
        message = os.urandom(size)

        start = time.time()
        cipher.encrypt(message)
        end = time.time()

        elapsed = end - start
        throughput = size / elapsed / 1e6  # MB per second

        times.append(elapsed)
        throughputs.append(throughput)

        print(f"Library DES - Size: {size:10d} bytes | Time: {elapsed:.4f} s | Speed: {throughput:.2f} MB/s")

    return sizes, times, throughputs

def compare_performance():
    key = b"8bytekey"
    round_keys = generate_round_keys(key)

    print("Running performance test for custom DES...\n")
    sizes, times_custom, throughputs_custom = measure_des_performance(round_keys)

    print("Running performance test for library DES...\n")
    sizes, times_lib, throughputs_lib = measure_library_des_performance(key)

    # --- Plot: Comparison of Throughputs ---
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, throughputs_custom, marker="o", label="Custom DES", color="blue")
    plt.plot(sizes, throughputs_lib, marker="o", label="Library DES", color="red")
    plt.xscale("log", base=2)
    plt.xlabel("Message Size (bytes, log scale)")
    plt.ylabel("Throughput (MB/s)")
    plt.title("Performance Comparison: Custom DES vs Library DES")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_performance()
