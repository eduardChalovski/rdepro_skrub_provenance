import subprocess
import sys

commands = [
    [sys.executable, "-m", "benchmarks.benchmark_memory"],
    [sys.executable, "-m", "benchmarks.benchmark_runtime_n_operators"],
    [sys.executable, "-m", "benchmarks.benchmark_runtime_n_rows"],
]

for cmd in commands:
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"❌ Command failed: {' '.join(cmd)}")
        sys.exit(result.returncode)

print("\n✅ All benchmarks finished successfully")
