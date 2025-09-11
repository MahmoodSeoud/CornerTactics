#!/bin/bash
#SBATCH --job-name=download_labels
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/labels_%j.log
#SBATCH --error=logs/labels_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

# Load required modules
module load python/3.9

# Set work directory
cd /home/mseo/CornerTactics

echo "Starting labels download at $(date)"
echo "Node: $(hostname)"

# Create data directory
mkdir -p data/raw/soccernet_labels

echo "Downloading Labels-v2.json files to organized data structure..."
python src/download_soccernet.py --labels v2 --data-dir data/raw/soccernet_labels

# Count the results
echo ""
echo "Download completed at $(date)"
echo "Total Labels-v2.json files downloaded: $(find data/raw/soccernet_labels -name 'Labels-v2.json' | wc -l)"
echo ""

# Show distribution by league
echo "Labels by league:"
for league in data/raw/soccernet_labels/*/; do
    if [ -d "$league" ]; then
        league_name=$(basename "$league")
        count=$(find "$league" -name "Labels-v2.json" | wc -l)
        echo "  $league_name: $count games"
    fi
done