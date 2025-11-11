#!/bin/bash
#SBATCH --job-name=corner_analysis
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/corner_analysis_%j.out
#SBATCH --error=logs/corner_analysis_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Navigate to project directory
cd /home/mseo/CornerTactics

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
pip install requests tqdm pandas numpy --quiet

echo "=========================================="
echo "StatsBomb Corner Transition Analysis"
echo "=========================================="
echo ""

# Step 1: Download raw JSON files
echo "📥 Step 1: Downloading StatsBomb raw JSON files..."
python scripts/download_statsbomb_raw_jsons.py

echo ""
echo "✅ Download complete!"
echo ""

# Step 2: Analyze corner transitions
echo "🔍 Step 2: Analyzing corner transitions P(a_{t+1} | corner_t)..."
python scripts/analyze_corner_transitions.py

echo ""
echo "✅ Analysis complete!"
echo ""
echo "📊 Results saved to: data/analysis/"
echo "   - corner_transition_matrix.csv"
echo "   - corner_sequences_detailed.json"
echo "   - corner_transition_report.md"