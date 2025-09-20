# CornerTactics Scripts

## SLURM Jobs

### 1. Download Data
```bash
sbatch scripts/slurm/download_data.sh
```
Downloads all SoccerNet data: Labels-v2.json, Labels-v3.json, videos, and tracking.

### 2. Extract Corner Frames
```bash
sbatch scripts/slurm/extract_corner_frames.sh
```
Extracts single frames at corner kick moments from all games.

## Local Usage

### Extract corner frames locally:
```bash
python extract_corners.py --data-dir data
```

### Download data locally:
```bash
python src/download_soccernet.py --all --password YOUR_PASSWORD
```

## Output

- **Frames**: `data/datasets/soccernet/corner_frames/`
- **Metadata CSV**: `data/insights/corner_frames_metadata.csv`
- **Logs**: `logs/slurm/`