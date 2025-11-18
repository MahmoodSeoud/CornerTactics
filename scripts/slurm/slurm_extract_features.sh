#!/bin/bash
#SBATCH --job-name=extract_features_49
#SBATCH --output=logs/extract_features_%j.out
#SBATCH --error=logs/extract_features_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=scavenge

# Load required modules
module load GCC/14.2.0

# Activate conda environment
source ~/.bashrc
conda activate robo

# Fix library path for pandas compatibility
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run feature extraction
echo "Starting feature extraction with 49 features..."
echo "Start time: $(date)"

python scripts/03_extract_features.py

echo "Feature extraction complete!"
echo "End time: $(date)"

# Verify output
if [ -f "data/processed/corners_with_features.csv" ]; then
    echo "Success! Output file created."
    echo "Checking file size and shape..."
    python -c "import pandas as pd; df = pd.read_csv('data/processed/corners_with_features.csv'); print(f'Shape: {df.shape}'); print(f'Columns: {len(df.columns)}')"
else
    echo "ERROR: Output file not created!"
    exit 1
fi
