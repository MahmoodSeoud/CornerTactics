# CornerTactics

Soccer corner kick analysis toolkit using SoccerNet dataset.

## Quick Start

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
brew install ffmpeg  # macOS
```

2. Download data (one-time setup):
```python
from src.data_loader import SoccerNetDataLoader

loader = SoccerNetDataLoader('data/')
loader.download_annotations('train')  # Downloads ALL games in split

# Download videos for specific games you want to analyze
game = 'england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom'
loader.download_videos(game)
```

### Run Analysis Pipeline

```bash
# List available games
python main.py --list

# Analyze a specific game
python main.py "england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom"

# Skip video extraction (faster, analysis only)
python main.py "england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom" --no-clips

# Save results to CSV
python main.py "england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom" --output results.csv
```

## Pipeline

The analysis follows a simple 3-step pipeline:

1. **Download** (prerequisite): Get SoccerNet data
2. **Extract**: Create video clips of corner kicks
3. **Analyze**: Generate statistics and label outcomes

## Project Structure

```
CornerTactics/
├── main.py                 # Main pipeline script
├── src/
│   ├── data_loader.py     # Download and load data
│   ├── corner_extractor.py # Extract corner kick clips
│   └── analyzer.py         # Analyze corner kicks
├── data/                   # SoccerNet dataset (gitignored)
├── corner_clips/           # Extracted clips (gitignored)
└── results/                # Analysis results (gitignored)
```

## API Usage

```python
from src.data_loader import SoccerNetDataLoader
from src.corner_extractor import CornerKickExtractor
from src.analyzer import CornerKickAnalyzer

# Load data
loader = SoccerNetDataLoader('data/')
annotations = loader.load_annotations(game)

# Extract clips
extractor = CornerKickExtractor(game)
clips = extractor.extract_all()

# Analyze
analyzer = CornerKickAnalyzer('data/')
df = analyzer.analyze_game(game)
outcomes = analyzer.label_outcomes(game)
```

## Notes

- SoccerNet downloads entire splits (cannot limit number of games)
- Each game with videos is ~400MB
- Video resolution is 398x224 (research quality)