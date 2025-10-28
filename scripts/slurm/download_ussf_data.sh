#!/bin/bash
#SBATCH --job-name=download_ussf
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/download_ussf_%j.out
#SBATCH --error=logs/download_ussf_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install gdown
pip install gdown --quiet

# Download the USSF data file
echo "Downloading USSF data from Google Drive..."
gdown 1TsFyeEKZ0sxjVV4u63omZhrKviBcmy_E -O data/raw/ussf_data_sample.pkl

# Inspect the file
echo ""
echo "File downloaded successfully!"
echo "File size:"
ls -lh data/raw/ussf_data_sample.pkl

echo ""
echo "Inspecting data structure..."
python << 'EOF'
import pickle
import numpy as np

# Load the data
with open('data/raw/ussf_data_sample.pkl', 'rb') as f:
    data = pickle.load(f)

print("\n=== DATA STRUCTURE ===")
print(f"Type: {type(data)}")
print(f"Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")

if isinstance(data, dict):
    for key, value in data.items():
        print(f"\n--- {key} ---")
        print(f"Type: {type(value)}")

        if isinstance(value, dict):
            print(f"Sub-keys: {list(value.keys())}")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, list) and len(subvalue) > 0:
                    print(f"  {subkey}: List of {len(subvalue)} items")
                    print(f"    First item type: {type(subvalue[0])}")
                    if isinstance(subvalue[0], np.ndarray):
                        print(f"    First item shape: {subvalue[0].shape}")
                elif isinstance(subvalue, np.ndarray):
                    print(f"  {subkey}: Array shape {subvalue.shape}")
                else:
                    print(f"  {subkey}: {type(subvalue)}")

        elif isinstance(value, list) and len(value) > 0:
            print(f"List of {len(value)} items")
            print(f"First item type: {type(value[0])}")
            if isinstance(value[0], np.ndarray):
                print(f"First item shape: {value[0].shape}")
            if hasattr(value[0], '__len__') and not isinstance(value[0], str):
                print(f"First item content: {value[0]}")

print("\n=== SAMPLE DATA ===")
# Print first few samples if possible
if isinstance(data, dict):
    if 'normal' in data and isinstance(data['normal'], dict):
        if 'x' in data['normal'] and len(data['normal']['x']) > 0:
            print(f"\nNode features (x) - First sample:")
            print(f"Shape: {data['normal']['x'][0].shape}")
            print(f"Sample:\n{data['normal']['x'][0][:3]}")  # First 3 nodes

        if 'a' in data['normal'] and len(data['normal']['a']) > 0:
            print(f"\nAdjacency matrix (a) - First sample:")
            print(f"Shape: {data['normal']['a'][0].shape}")
            print(f"Number of edges: {np.sum(data['normal']['a'][0])}")

        if 'e' in data['normal'] and len(data['normal']['e']) > 0:
            print(f"\nEdge features (e) - First sample:")
            print(f"Shape: {data['normal']['e'][0].shape}")
            print(f"Sample:\n{data['normal']['e'][0][:3]}")  # First 3 edges

    if 'binary' in data and len(data['binary']) > 0:
        print(f"\nLabels (binary):")
        print(f"Total samples: {len(data['binary'])}")
        print(f"Positive samples: {sum([x[0] for x in data['binary']])}")
        print(f"Negative samples: {len(data['binary']) - sum([x[0] for x in data['binary']])}")
        print(f"First 10 labels: {[x[0] for x in data['binary'][:10]]}")

EOF

echo ""
echo "Download and inspection complete!"
