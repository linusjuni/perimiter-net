import numpy as np
import os

# Update path if needed
base_dir = "/work3/s225224/ucf-crime/features/rgb"

# Pick a few randoms
test_files = [
    "Test/Explosion022_x264.npy", 
    "Train/Abuse001_x264.npy",
    "Test/Burglary079_x264.npy"
]

print(f"{'Video':<25} | {'Shape':<15} | {'Status'}")
print("-" * 50)

for f in test_files:
    path = os.path.join(base_dir, f)
    if os.path.exists(path):
        data = np.load(path)
        # Check if shape is (N, 512) and N > 50
        status = "✅ OK" if data.shape[0] > 50 else "⚠️ Suspiciously Short"
        print(f"{f:<25} | {str(data.shape):<15} | {status}")
    else:
        print(f"{f:<25} | {'MISSING':<15} | ❌")