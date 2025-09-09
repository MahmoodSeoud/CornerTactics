# CornerTactics: Soccer Corner Kick Analysis Project

## Overview
Analyze corner kick tactics and defensive formations using SoccerNet-Tracking dataset to identify patterns and success factors.

## Phase 1: Data Acquisition & Setup

### 1.1 Download SoccerNet-Tracking Dataset
```bash
# Install SoccerNet package
pip install SoccerNet

# Download SoccerNet-Tracking data
python -c "
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory='data/')

# Download tracking annotations and videos
mySoccerNetDownloader.downloadGames(files=['Frames-v1.zip'], split=['train','valid','test'])
mySoccerNetDownloader.downloadGames(files=['Labels-v2.json'], split=['train','valid','test'])
"
```

### 1.2 Install Dependencies
```bash
pip install kloppy pandas numpy matplotlib seaborn plotly
```

## Phase 2: Extract Corner Kick Data

### 2.1 Load SoccerNet Tracking Data
```python
# Script: load_data.py
import kloppy
import pandas as pd

def load_soccernet_tracking_data():
    """Load SoccerNet tracking data using kloppy"""
    tracking_datasets = []
    
    for game_folder in soccernet_tracking_games:
        dataset = kloppy.load_tracking_data(
            f"data/{game_folder}/tracking_data.json",
            provider="epts"
        )
        tracking_datasets.append(dataset)
    
    return tracking_datasets
```

### 2.2 Extract Corner Kick Moments
```python
# Script: extract_corners.py
def extract_corner_kick_moments(tracking_datasets):
    """Extract all corner kick events with player positions"""
    corner_data = []
    
    for dataset in tracking_datasets:
        # Find corner kick timestamps from annotations
        corner_events = dataset.find_all("corner")
        
        for corner_event in corner_events:
            # Get player positions at corner kick moment
            corner_frame = dataset.get_frame_at_timestamp(corner_event.timestamp)
            
            corner_info = {
                'game_id': dataset.metadata.game_id,
                'timestamp': corner_event.timestamp,
                'corner_side': determine_corner_side(corner_event),
                'player_positions': extract_player_positions(corner_frame),
                'ball_position': corner_frame.ball_coordinates,
                'defending_team': identify_defending_team(corner_frame, corner_event)
            }
            
            corner_data.append(corner_info)
    
    return corner_data
```

## Phase 3: Simple Data Processing

### 3.1 Basic Player Position Analysis
```python
# Script: analyze_positions.py
def analyze_player_positions(corner_data):
    """Basic analysis of player positions during corners"""
    
    processed_data = []
    
    for corner in corner_data:
        defending_players = get_defending_team_players(corner)
        
        basic_stats = {
            'game_id': corner['game_id'],
            'timestamp': corner['timestamp'],
            'corner_side': corner['corner_side'],
            
            # Simple defensive metrics
            'defenders_in_penalty_box': count_players_in_penalty_area(defending_players),
            'total_defending_players': len(defending_players),
            'goalkeeper_x': get_goalkeeper_position(defending_players)['x'],
            'goalkeeper_y': get_goalkeeper_position(defending_players)['y'],
            
            # Simple attacking metrics  
            'attacking_players_in_box': count_attacking_players_in_box(corner['player_positions'])
        }
        
        processed_data.append(basic_stats)
    
    return pd.DataFrame(processed_data)
```

### 3.2 Label Outcomes
```python
# Script: label_outcomes.py
def label_corner_outcomes(corner_data):
    """Simple outcome labeling for corner kicks"""
    
    for corner in corner_data:
        # Check what happened after the corner kick
        # Look at next 15 seconds of tracking data
        outcome = determine_simple_outcome(corner)
        corner['outcome'] = outcome  # 'goal', 'cleared', 'continued_play'
    
    return corner_data
```

## Project Structure
```
CornerTactics/
├── data/                    # SoccerNet dataset
├── src/
│   ├── load_data.py        # Data loading utilities
│   ├── extract_corners.py  # Corner kick extraction
│   ├── analyze_positions.py # Position analysis
│   └── label_outcomes.py   # Outcome labeling
├── notebooks/              # Jupyter notebooks for analysis
├── results/                # Output data and visualizations
└── README.md
```

## Next Steps
1. Set up development environment
2. Download SoccerNet-Tracking dataset
3. Implement data loading pipeline
4. Extract corner kick events
5. Build basic analysis framework