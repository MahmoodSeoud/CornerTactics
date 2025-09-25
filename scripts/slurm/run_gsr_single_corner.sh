#!/bin/bash
#SBATCH --job-name=gsr_corner_test
#SBATCH --output=/home/mseo/CornerTactics/logs/gsr_corner_test_%j.log
#SBATCH --error=/home/mseo/CornerTactics/logs/gsr_corner_test_%j.err
#SBATCH --partition=acltr
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4

# Create logs directory if it doesn't exist
mkdir -p /home/mseo/CornerTactics/logs

echo "=========================================="
echo "Starting GSR Single Corner Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Navigate to sn-gamestate directory
cd /home/mseo/CornerTactics/sn-gamestate

# Fix SSL certificates for UV container
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

# Check GPU availability
echo "GPU Information:"
nvidia-smi

# Process single corner clip for testing
VIDEO_PATH="/home/mseo/CornerTactics/data/datasets/soccernet/corner_clips/visible/england_epl_2014-2015_2015-02-21-18-00Ch_1H_03m02s_away.mp4"
VIDEO_NAME="england_epl_2014-2015_2015-02-21-18-00Ch_1H_03m02s_away"
OUTPUT_DIR="/home/mseo/CornerTactics/data/gsr_outputs"

mkdir -p $OUTPUT_DIR

echo ""
echo "Processing single test video: $VIDEO_NAME"
echo "Video path: $VIDEO_PATH"
echo "Output directory: $OUTPUT_DIR/${VIDEO_NAME}"
echo ""

# Run GSR pipeline with corner clips config (everything defined in corner_clips.yaml)
uv run tracklab -cn corner_clips \
    dataset.video_path="$VIDEO_PATH" \
    experiment_name="corner_${VIDEO_NAME}" \
    hydra.run.dir="$OUTPUT_DIR/${VIDEO_NAME}" \
    num_cores=4

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Successfully processed: $VIDEO_NAME"
    echo "Check output at: $OUTPUT_DIR/${VIDEO_NAME}"

    # List output files
    echo ""
    echo "Generated files:"
    find "$OUTPUT_DIR/${VIDEO_NAME}" -type f -name "*.mp4" -o -name "*.json" -o -name "*.txt" | head -20
else
    echo ""
    echo "✗ Error processing: $VIDEO_NAME"
fi

echo ""
echo "=========================================="
echo "GSR Single Corner Test Complete"
echo "End time: $(date)"
echo "=========================================="