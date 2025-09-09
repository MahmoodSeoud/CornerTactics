# CornerTactics

Simple pipeline for analyzing corner kicks from soccer matches using SoccerNet dataset.

## Setup

```bash
pip install -r requirements.txt
brew install ffmpeg  # macOS
```

## Usage

### Step 1: Download Data (One-time)

```python
from src.data_loader import SoccerNetDataLoader

loader = SoccerNetDataLoader('data/')
loader.download_annotations('train')  # Downloads ALL games in split
loader.download_videos('england_epl/2015-2016/...')  # Download videos for specific game
```

### Step 2: Run Pipeline

```bash
# Extract clips and analyze ALL games
python main.py

# Just analyze (skip video extraction for speed)
python main.py --no-clips

# Custom clip settings
python main.py --duration 20 --before 5
```

## Pipeline Flow

1. **Finds** all games in data folder
2. **Extracts** video clips of every corner kick
3. **Analyzes** all corners across all games
4. **Saves** combined results to CSV

## Output

- **Video clips**: `corner_clips/corner_1H_28m46s_home.mp4`
- **Analysis**: `results.csv` with all corner data

## Project Structure

```
CornerTactics/
├── main.py                 # Run complete pipeline
├── src/
│   ├── data_loader.py     # Download SoccerNet data
│   ├── corner_extractor.py # Extract corner clips
│   └── analyzer.py         # Analyze corners
├── data/                   # SoccerNet dataset
├── corner_clips/           # Extracted video clips
└── results.csv            # Analysis output
```

## Notes

- Processes ALL games automatically
- SoccerNet downloads entire splits (100+ games)
- Each game with videos is ~400MB
- Video resolution: 398x224 (research quality)