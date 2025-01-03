import pandas as pd 
import matplotlib.pyplot as plt
import os

# Function to read benchmark results
def read_benchmark_results(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None

# Read data for all versions
versions = ['serial', 'omp', 'acc', 'cuda']  # Added 'cuda' version
data = {}
for version in versions:
    filename = f'benchmark_results_{version}.txt'
    data[version] = read_benchmark_results(filename)
    if data[version] is None:
        print(f"Skipping {version} version due to missing data.")

# Create plots
plt.figure(figsize=(12, 8))

# GFLOPS vs Matrix Size
plt.subplot(2, 2, 1)
for version in versions:
    if data[version] is not None:
        plt.plot(data[version]['Matrix_Size'], data[version]['GFLOPS'], marker='o', label=version.upper())
plt.xlabel('Matrix Size')
plt.ylabel('GFLOPS')
plt.title('GFLOPS vs Matrix Size')
plt.xscale('log', base=2)
plt.legend()
plt.grid(True)

# Wall Time vs Matrix Size
plt.subplot(2, 2, 2)
for version in versions:
    if data[version] is not None:
        plt.plot(data[version]['Matrix_Size'], data[version]['Time_Seconds'], marker='o', label=version.upper())
plt.xlabel('Matrix Size')
plt.ylabel('Wall Time (seconds)')
plt.title('Wall Time vs Matrix Size')
plt.xscale('log', base=2)
plt.yscale('log')
plt.legend()
plt.grid(True)

# Bandwidth vs Matrix Size
plt.subplot(2, 2, 3)
for version in versions:
    if data[version] is not None:
        plt.plot(data[version]['Matrix_Size'], data[version]['Bandwidth_GB_per_S'], marker='o', label=version.upper())
plt.xlabel('Matrix Size')
plt.ylabel('Bandwidth (GB/s)')
plt.title('Bandwidth vs Matrix Size')
plt.xscale('log', base=2)
plt.legend()
plt.grid(True)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('benchmark_comparison.png')
plt.close()

print("Benchmark comparison plot saved as 'benchmark_comparison.png'")

# Create a table of results
results = []
for version in versions:
    if data[version] is not None:
        max_gflops = data[version]['GFLOPS'].max()
        max_bandwidth = data[version]['Bandwidth_GB_per_S'].max()
        results.append({
            'Version': version.upper(),
            'Max GFLOPS': max_gflops,
            'Max Bandwidth (GB/s)': max_bandwidth
        })

results_df = pd.DataFrame(results)
print("\nSummary of Results:")
print(results_df.to_string(index=False))

# Save the summary to a CSV file
results_df.to_csv('benchmark_summary.csv', index=False)
print("\nSummary saved to 'benchmark_summary.csv'")

