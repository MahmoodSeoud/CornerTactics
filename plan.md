fication. It contains 7 Bundesliga matches. We estimate ~70 corner kicks across these matches.

**The approach**: Pretrain a GNN on open-play sequences (thousands of frames across 7 full matches), then fine-tune on ~70 corner kicks. This is transfer learning that makes small-data viable.

**Reference architecture**: TacticAI (DeepMind + Liverpool FC, Nature Communications 2024) — GATv2 on 7,176 corners with 25Hz tracking. Achieved F1=0.52 on shot prediction. Our goal is not to match TacticAI but to demonstrate that velocity features unlock predictive signal that static features cannot.

---

## Phase 1: Data Loading & Validation

**Goal**: Load DFL tracking data, count corners, extract one complete corner as proof of concept.

**Estimated time**: 2-3 hours o# DFL Corner Kick GNN — LLM Implementation Plan

## Context for the executing agent

You are implementing a spatio-temporal Graph Neural Network (ST-GNN) for corner kick outcome prediction in football/soccer. This is a thesis project at IT University of Copenhagen.

**The core hypothesis**: Static player positions at corner kick delivery contain no predictive signal (AUC ≈ 0.50, validated). Velocity vectors derived from continuous tracking data are hypothesized to be necessary. This pipeline tests that hypothesis using real tracking data.

**Why DFL**: The DFL (Deutsche Fußball Liga) / Sportec open tracking dataset is one of the only publicly available datasets providing continuous 25fps optical tracking of all 22 players + ball with full player identif coding.

### Step 1.1: Install dependencies

```bash
pip install kloppy pandas numpy matplotlib
```

`kloppy` is the standard Python library for loading multi-vendor tracking/event data into a unified format. It handles DFL/Sportec's proprietary XML format.

### Step 1.2: Download DFL open data

The DFL open data is hosted on GitHub by the Sportec company. Search for the repository:

```
Repository: https://github.com/DFL-Scientific/Open-Data (or similar)
```

If not found at that URL, search GitHub for "DFL open data sportec tracking" or "Sportec open tracking Bundesliga". The dataset should contain:
- 7 Bundesliga match files
- Tracking data at 25fps (XML or similar format)
- Event data with timestamps

**Verification checkpoint**: You should have 7 match directories/files. If you find a different number, note it and proceed.

### Step 1.3: Load one match with kloppy

```python
from kloppy import sportec

# Load tracking data for one match
# Adjust path/arguments based on actual file format discovered
dataset = sportec.load_tracking(
    raw_data="path/to/match_tracking.xml",
    metadata="path/to/match_metadata.xml"
)

# Inspect the dataset
print(f"Frames: {len(dataset.records)}")
print(f"Frame rate: {dataset.metadata.frame_rate}")
print(f"Players: {len(dataset.metadata.teams[0].players) + len(dataset.metadata.teams[1].players)}")
print(f"Coordinate system: {dataset.metadata.coordinate_system}")

# Print first frame to understand structure
frame = dataset.records[0]
print(f"Timestamp: {frame.timestamp}")
for player_data in frame.players_data.items():
    print(player_data)
```

**Verification checkpoint**: Confirm you see 22 player positions per frame at 25fps. Note the coordinate system (pitch dimensions, origin location).

### Step 1.4: Load event data and count corners

```python
# Load event data for the same match
event_dataset = sportec.load_event(
    event_data="path/to/match_events.xml",
    metadata="path/to/match_metadata.xml"
)

# Count corner kicks
from kloppy.domain import EventType
corners = [e for e in event_dataset.events if "corner" in str(e.event_type).lower() 
           or (hasattr(e, 'qualifiers') and any('corner' in str(q).lower() for q in e.qualifiers))]
print(f"Corners in this match: {len(corners)}")

# If kloppy's event type filtering doesn't work directly, inspect all event types:
event_types = set(str(e.event_type) for e in event_dataset.events)
print(f"Available event types: {event_types}")
```

**IMPORTANT**: kloppy's API may vary. If `sportec.load_tracking` or `sportec.load_event` don't work with these exact signatures, check `kloppy`'s documentation. Try:
```python
from kloppy import load_sportec_tracking, load_sportec_event
# OR
from kloppy.io import sportec_tracking_data
```

Adapt accordingly. The API is well-documented but has changed across versions.

### Step 1.5: Count corners across ALL 7 matches

```python
import os

total_corners = 0
match_corner_counts = {}

for match_dir in sorted(os.listdir("path/to/dfl_data/")):
    # Load events for each match
    # Count corners
    # Store: match_corner_counts[match_dir] = count
    pass

print(f"Total corners across all matches: {total_corners}")
print(f"Per-match breakdown: {match_corner_counts}")
```

**Decision point**: 
- If total corners >= 50: Proceed with Phase 2.
- If total corners < 30: Also load Metrica Sports data (3 matches, ~30 corners) to supplement. Install: `pip install kloppy` and use `kloppy.metrica` loader.
- If total corners < 15: This approach is not viable with DFL alone. Report finding and pivot to Gaussian Process on handcrafted features (see Appendix A).

### Step 1.6: Extract one complete corner kick sequence

For one corner kick event, extract the tracking data window from **2 seconds before** to **6 seconds after** delivery:

```python
import numpy as np

def extract_corner_sequence(tracking_dataset, corner_event, fps=25, pre_seconds=2, post_seconds=6):
    """
    Extract tracking frames around a corner kick event.
    
    Returns:
        frames: list of dicts, each containing:
            - timestamp: float (seconds)
            - players: dict mapping player_id -> {x, y, team, vx, vy}
            - ball: {x, y}
    """
    corner_time = corner_event.timestamp  # seconds from match start
    start_time = corner_time - pre_seconds
    end_time = corner_time + post_seconds
    
    # Filter tracking frames to this window
    window_frames = [
        f for f in tracking_dataset.records 
        if start_time <= f.timestamp <= end_time
    ]
    
    print(f"Extracted {len(window_frames)} frames ({len(window_frames)/fps:.1f}s)")
    # Expected: (pre_seconds + post_seconds) * fps = 8 * 25 = 200 frames
    
    return window_frames
```

### Step 1.7: Compute velocity vectors

```python
def compute_velocities(frames, fps=25):
    """
    Compute velocity vectors using finite differences.
    v(t) = (pos(t+1) - pos(t-1)) / (2 * dt)  # central difference
    
    Returns frames with added vx, vy per player.
    """
    dt = 1.0 / fps
    
    for i in range(1, len(frames) - 1):
        for player_id in frames[i].players_data:
            prev = frames[i-1].players_data.get(player_id)
            next_ = frames[i+1].players_data.get(player_id)
            curr = frames[i].players_data[player_id]
            
            if prev and next_ and prev.coordinates and next_.coordinates:
                vx = (next_.coordinates.x - prev.coordinates.x) / (2 * dt)
                vy = (next_.coordinates.y - prev.coordinates.y) / (2 * dt)
            else:
                vx, vy = 0.0, 0.0
            
            # Store velocities (attach to frame data structure)
            # Exact implementation depends on kloppy's data model
            # May need to create a parallel dict: velocities[player_id] = (vx, vy)
    
    return frames  # with velocities attached
```

**Velocity sanity checks**:
- Walking speed: ~1.5 m/s
- Jogging: ~3-4 m/s  
- Sprinting: ~8-10 m/s
- If you see velocities > 12 m/s consistently, there's a coordinate system or unit error.

### Step 1.8: Visualize the corner kick

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_corner_frame(frame, velocities=None, title="Corner Kick"):
    """Plot a single frame with player positions and optional velocity arrows."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Draw pitch (105m x 68m standard, but check DFL coordinate system)
    pitch_length, pitch_width = 105, 68
    ax.set_xlim(-5, pitch_length + 5)
    ax.set_ylim(-5, pitch_width + 5)
    
    # Draw penalty box, goal, etc.
    # ... (standard pitch drawing code)
    
    for player_id, pdata in frame.players_data.items():
        if pdata.coordinates:
            x, y = pdata.coordinates.x, pdata.coordinates.y
            # Color by team
            color = 'red' if pdata.team == 'home' else 'blue'  # adapt to actual team identification
            ax.scatter(x, y, c=color, s=100, zorder=5)
            
            # Draw velocity arrows if available
            if velocities and player_id in velocities:
                vx, vy = velocities[player_id]
                ax.arrow(x, y, vx*0.3, vy*0.3, head_width=0.5, color=color, alpha=0.7)
    
    # Ball position
    if hasattr(frame, 'ball_coordinates') and frame.ball_coordinates:
        ax.scatter(frame.ball_coordinates.x, frame.ball_coordinates.y, 
                   c='yellow', s=150, edgecolors='black', zorder=10, marker='o')
    
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.savefig("corner_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()
```

**Phase 1 deliverable**: A saved visualization (`corner_visualization.png`) showing player positions + velocity arrows for one corner kick. This proves the data pipeline works end-to-end.

---

## Phase 2: Graph Construction Pipeline

**Goal**: Convert each tracking frame into a graph. Build the full dataset of labeled corner kick graphs.

**Estimated time**: 4-6 hours of coding.

### Step 2.1: Install graph libraries

```bash
pip install torch torch-geometric networkx
```

If `torch-geometric` installation fails (common), follow: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### Step 2.2: Define graph structure per frame

Each frame becomes a graph with:
- **Nodes**: 22 players + 1 ball = 23 nodes
- **Node features**: [x, y, vx, vy, team_flag, is_kicker, dist_to_goal, dist_to_ball]
  - `team_flag`: 1.0 for attacking team, 0.0 for defending team
  - `is_kicker`: 1.0 for the corner taker, 0.0 otherwise
  - `dist_to_goal`: Euclidean distance to center of goal
  - `dist_to_ball`: Euclidean distance to ball position
  - Total: **8 features per node**

- **Edges**: Two types, combined into one edge index
  1. **Proximity edges (kNN)**: Connect each player to their k=4 nearest neighbors (regardless of team)
  2. **Marking edges**: Connect each attacker to their nearest defender and vice versa

```python
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import cdist

def frame_to_graph(frame, velocities, corner_event, k_neighbors=4):
    """
    Convert a single tracking frame into a PyTorch Geometric graph.
    
    Args:
        frame: tracking frame with player positions
        velocities: dict mapping player_id -> (vx, vy)
        corner_event: the corner kick event (to identify attacking team)
        k_neighbors: number of nearest neighbors for kNN edges
    
    Returns:
        torch_geometric.data.Data object
    """
    node_features = []
    positions = []
    player_ids = []
    teams = []
    
    # Goal position (adapt to coordinate system)
    goal_x, goal_y = 105.0, 34.0  # center of goal, CHECK coordinate system
    
    # Ball position
    ball_x = frame.ball_coordinates.x if frame.ball_coordinates else 0.0
    ball_y = frame.ball_coordinates.y if frame.ball_coordinates else 0.0
    
    # Identify attacking team from corner event
    attacking_team = corner_event.team  # adapt to actual attribute name
    
    for player_id, pdata in frame.players_data.items():
        if pdata.coordinates is None:
            continue
            
        x, y = pdata.coordinates.x, pdata.coordinates.y
        vx, vy = velocities.get(player_id, (0.0, 0.0))
        
        is_attacking = 1.0 if pdata.team == attacking_team else 0.0
        is_kicker_flag = 0.0  # Set to 1.0 for corner taker if identifiable
        dist_to_goal = ((x - goal_x)**2 + (y - goal_y)**2)**0.5
        dist_to_ball = ((x - ball_x)**2 + (y - ball_y)**2)**0.5
        
        node_features.append([x, y, vx, vy, is_attacking, is_kicker_flag, dist_to_goal, dist_to_ball])
        positions.append([x, y])
        player_ids.append(player_id)
        teams.append(is_attacking)
    
    # Add ball as node 
    node_features.append([ball_x, ball_y, 0.0, 0.0, -1.0, 0.0, 
                          ((ball_x - goal_x)**2 + (ball_y - goal_y)**2)**0.5, 0.0])
    positions.append([ball_x, ball_y])
    
    x = torch.tensor(node_features, dtype=torch.float)
    pos = torch.tensor(positions, dtype=torch.float)
    
    # --- Edge construction ---
    n_players = len(positions) - 1  # exclude ball for player edges
    player_pos = np.array(positions[:n_players])
    
    # 1. kNN proximity edges
    if n_players > k_neighbors:
        dist_matrix = cdist(player_pos, player_pos)
        np.fill_diagonal(dist_matrix, np.inf)
        knn_edges = []
        for i in range(n_players):
            neighbors = np.argsort(dist_matrix[i])[:k_neighbors]
            for j in neighbors:
                knn_edges.append([i, j])
    
    # 2. Marking edges (nearest opponent)
    attackers = [i for i in range(n_players) if teams[i] == 1.0]
    defenders = [i for i in range(n_players) if teams[i] == 0.0]
    
    marking_edges = []
    if attackers and defenders:
        att_pos = player_pos[attackers]
        def_pos = player_pos[defenders]
        cross_dist = cdist(att_pos, def_pos)
        
        # Each attacker -> nearest defender
        for i, att_idx in enumerate(attackers):
            nearest_def = defenders[np.argmin(cross_dist[i])]
            marking_edges.append([att_idx, nearest_def])
            marking_edges.append([nearest_def, att_idx])  # bidirectional
    
    # 3. Ball edges: connect ball to all players
    ball_idx = len(positions) - 1
    ball_edges = []
    for i in range(n_players):
        ball_edges.append([i, ball_idx])
        ball_edges.append([ball_idx, i])
    
    # Combine all edges
    all_edges = knn_edges + marking_edges + ball_edges
    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index, pos=pos)
```

### Step 2.3: Build temporal graph sequence per corner

```python
def corner_to_temporal_graphs(tracking_dataset, corner_event, fps=25):
    """
    Convert a corner kick into a sequence of graphs.
    
    Returns:
        List[Data]: one graph per frame, covering -2s to +6s
    """
    frames = extract_corner_sequence(tracking_dataset, corner_event, fps)
    frames_with_vel = compute_velocities(frames, fps)
    
    graphs = []
    for i, frame in enumerate(frames_with_vel):
        if i == 0 or i == len(frames_with_vel) - 1:
            continue  # skip first/last (no velocity from central difference)
        
        vel_dict = {}  # extract velocities for this frame
        # ... (depends on how you stored velocities in Step 1.7)
        
        g = frame_to_graph(frame, vel_dict, corner_event)
        g.frame_idx = i
        g.relative_time = (i / fps) - 2.0  # relative to corner delivery
        graphs.append(g)
    
    return graphs
```

### Step 2.4: Label each corner with outcomes

```python
def label_corner(corner_event, event_dataset, n_subsequent_events=5):
    """
    Create multi-head labels for a corner kick.
    
    Returns dict with:
        - shot_binary: 1 if shot within next 5 events, else 0
        - goal_binary: 1 if goal within next 5 events, else 0
        - first_contact_team: 'attacking' or 'defending'
        - outcome_class: one of ['goal', 'shot_saved', 'shot_blocked', 'clearance', 'ball_receipt', 'other']
    """
    # Get events following this corner
    all_events = list(event_dataset.events)
    corner_idx = None
    for i, e in enumerate(all_events):
        if e.timestamp == corner_event.timestamp:  # or match by event_id
            corner_idx = i
            break
    
    if corner_idx is None:
        return None
    
    subsequent = all_events[corner_idx + 1 : corner_idx + 1 + n_subsequent_events]
    
    labels = {
        'shot_binary': 0,
        'goal_binary': 0,
        'first_contact_team': 'unknown',
        'outcome_class': 'other'
    }
    
    for event in subsequent:
        event_type_str = str(event.event_type).lower()
        
        if 'shot' in event_type_str:
            labels['shot_binary'] = 1
            if 'goal' in event_type_str or getattr(event, 'result', '') == 'goal':
                labels['goal_binary'] = 1
        
        # First contact detection
        if labels['first_contact_team'] == 'unknown':
            if hasattr(event, 'team'):
                if event.team == corner_event.team:
                    labels['first_contact_team'] = 'attacking'
                else:
                    labels['first_contact_team'] = 'defending'
    
    return labels
```

### Step 2.5: Build the full dataset

```python
import pickle

def build_corner_dataset(dfl_data_path):
    """
    Process all 7 DFL matches and build the complete corner kick dataset.
    
    Saves:
        corner_dataset.pkl: list of dicts, each containing:
            - 'graphs': List[Data] (temporal sequence of graphs)
            - 'labels': dict (multi-head labels)
            - 'match_id': str
            - 'corner_time': float
    """
    dataset = []
    
    for match_file in get_all_match_files(dfl_data_path):
        tracking = load_tracking(match_file)
        events = load_events(match_file)
        corners = find_corners(events)
        
        for corner in corners:
            graphs = corner_to_temporal_graphs(tracking, corner)
            labels = label_corner(corner, events)
            
            if graphs and labels:
                dataset.append({
                    'graphs': graphs,
                    'labels': labels,
                    'match_id': match_file,
                    'corner_time': corner.timestamp
                })
    
    print(f"Built dataset: {len(dataset)} corners")
    print(f"Shot rate: {sum(d['labels']['shot_binary'] for d in dataset) / len(dataset):.1%}")
    
    with open('corner_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    
    return dataset
```

**Phase 2 deliverable**: `corner_dataset.pkl` containing ~70 labeled corner kick graph sequences. Print summary statistics: total corners, shot rate, average frames per corner, class distribution.

---

## Phase 3: Model Architecture & Training

**Goal**: Build ST-GNN, pretrain on open play, fine-tune on corners.

**Estimated time**: 8-12 hours of coding + training.

### Step 3.1: Define the Spatio-Temporal GNN

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

class SpatialGNN(nn.Module):
    """Process a single frame's graph."""
    
    def __init__(self, in_channels=8, hidden_channels=64, out_channels=32, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=False)
        self.conv2 = GATv2Conv(hidden_channels, out_channels, heads=heads, concat=False)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(out_channels)
    
    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = torch.relu(x)
        
        # Pool to get graph-level representation
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)  # single graph
        
        return x  # shape: (1, out_channels) per frame


class TemporalAggregator(nn.Module):
    """Aggregate frame-level representations over time."""
    
    def __init__(self, input_dim=32, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, 
                          batch_first=True, dropout=0.1)
    
    def forward(self, frame_embeddings):
        """
        Args:
            frame_embeddings: (batch, seq_len, input_dim)
        Returns:
            (batch, hidden_dim) — final hidden state
        """
        output, hidden = self.gru(frame_embeddings)
        return hidden[-1]  # last layer's final hidden state


class CornerKickPredictor(nn.Module):
    """Full ST-GNN: spatial GNN per frame -> temporal GRU -> multi-head output."""
    
    def __init__(self, node_features=8, gnn_hidden=64, gnn_out=32, 
                 temporal_hidden=64, num_classes_outcome=6):
        super().__init__()
        self.spatial_gnn = SpatialGNN(node_features, gnn_hidden, gnn_out)
        self.temporal = TemporalAggregator(gnn_out, temporal_hidden)
        
        # Multi-head outputs
        self.head_shot = nn.Linear(temporal_hidden, 1)        # binary: shot or not
        self.head_goal = nn.Linear(temporal_hidden, 1)        # binary: goal or not
        self.head_contact = nn.Linear(temporal_hidden, 2)     # first contact team
        self.head_outcome = nn.Linear(temporal_hidden, num_classes_outcome)  # outcome class
    
    def forward(self, graph_sequences):
        """
        Args:
            graph_sequences: list of list of Data objects
                             graph_sequences[i] = [graph_t0, graph_t1, ..., graph_tN] for corner i
        
        Returns:
            dict of predictions for each head
        """
        batch_embeddings = []
        
        for seq in graph_sequences:
            frame_embs = []
            for graph in seq:
                emb = self.spatial_gnn(graph.x, graph.edge_index)
                frame_embs.append(emb.squeeze(0))
            
            # Stack frames: (seq_len, gnn_out)
            frame_embs = torch.stack(frame_embs, dim=0)
            batch_embeddings.append(frame_embs)
        
        # Pad sequences to same length
        max_len = max(emb.shape[0] for emb in batch_embeddings)
        padded = torch.zeros(len(batch_embeddings), max_len, batch_embeddings[0].shape[-1])
        for i, emb in enumerate(batch_embeddings):
            padded[i, :emb.shape[0]] = emb
        
        # Temporal aggregation
        temporal_out = self.temporal(padded)  # (batch, temporal_hidden)
        
        return {
            'shot': torch.sigmoid(self.head_shot(temporal_out)),
            'goal': torch.sigmoid(self.head_goal(temporal_out)),
            'contact': self.head_contact(temporal_out),
            'outcome': self.head_outcome(temporal_out)
        }
```

### Step 3.2: Pretrain on open-play sequences (CRITICAL for small data)

**This is the transfer learning step that makes 70 corners viable.**

Extract ALL possession sequences from the 7 DFL matches (not just corners). This gives thousands of training examples.

```python
def extract_open_play_sequences(tracking_dataset, event_dataset, 
                                 window_seconds=4, stride_seconds=2):
    """
    Extract overlapping windows from open play.
    Label each window with: did a shot happen within 6 seconds?
    
    This gives ~1000-3000 training sequences from 7 matches.
    """
    sequences = []
    fps = 25
    window_frames = window_seconds * fps
    stride_frames = stride_seconds * fps
    
    all_frames = tracking_dataset.records
    shot_events = [e for e in event_dataset.events if 'shot' in str(e.event_type).lower()]
    shot_times = [e.timestamp for e in shot_events]
    
    for start_idx in range(0, len(all_frames) - window_frames, stride_frames):
        window = all_frames[start_idx : start_idx + window_frames]
        window_end_time = window[-1].timestamp
        
        # Label: will there be a shot within 6 seconds of window end?
        shot_within_6s = any(
            0 < (st - window_end_time) < 6.0 for st in shot_times
        )
        
        sequences.append({
            'frames': window,
            'shot_label': int(shot_within_6s),
            'start_time': window[0].timestamp
        })
    
    return sequences
```

**Pretrain the spatial GNN only** (freeze temporal later):

```python
def pretrain_spatial_gnn(model, open_play_sequences, epochs=50, lr=1e-3):
    """
    Pretrain ONLY the spatial GNN + shot head on open-play data.
    Use single-frame prediction (no temporal component).
    """
    optimizer = torch.optim.Adam(
        list(model.spatial_gnn.parameters()) + list(model.head_shot.parameters()),
        lr=lr, weight_decay=1e-4
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))  # handle class imbalance
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for seq in open_play_sequences:
            # Use middle frame of each sequence
            mid_frame = seq['frames'][len(seq['frames']) // 2]
            vel_dict = {}  # compute velocities
            graph = frame_to_graph(mid_frame, vel_dict, corner_event=None)
            
            emb = model.spatial_gnn(graph.x, graph.edge_index)
            pred = model.head_shot(emb)
            
            target = torch.tensor([[float(seq['shot_label'])]])
            loss = criterion(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Pretrain Epoch {epoch}: Loss = {total_loss / len(open_play_sequences):.4f}")
    
    return model
```

### Step 3.3: Fine-tune on corner kicks

```python
def finetune_on_corners(model, corner_dataset, epochs=100, lr=1e-4):
    """
    Fine-tune the FULL model (spatial + temporal + all heads) on corner kicks.
    
    Use leave-one-match-out cross-validation (7 folds, one per match).
    """
    # Match-based split (CRITICAL: no match overlap between train/test)
    matches = list(set(d['match_id'] for d in corner_dataset))
    
    results = []
    
    for test_match in matches:
        train_data = [d for d in corner_dataset if d['match_id'] != test_match]
        test_data = [d for d in corner_dataset if d['match_id'] == test_match]
        
        if len(test_data) == 0:
            continue
        
        # Clone model for this fold
        fold_model = copy.deepcopy(model)
        optimizer = torch.optim.Adam(fold_model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Multi-task loss
        shot_criterion = nn.BCELoss(weight=torch.tensor([3.0]))  # weight minority class
        
        for epoch in range(epochs):
            fold_model.train()
            # ... training loop with multi-head loss
            # loss = shot_loss + goal_loss + contact_loss + outcome_loss
            pass
        
        # Evaluate on test fold
        fold_model.eval()
        with torch.no_grad():
            # ... compute AUC, F1 on test_data
            pass
        
        results.append({'test_match': test_match, 'auc': ..., 'f1': ...})
    
    # Report mean ± std across folds
    print(f"Shot AUC: {np.mean([r['auc'] for r in results]):.3f} ± {np.std([r['auc'] for r in results]):.3f}")
    
    return results
```

### Step 3.4: Ablation — Position-only vs Position+Velocity

**This is the key experiment that tests the velocity hypothesis.**

```python
def run_ablation(corner_dataset):
    """
    Compare:
    A) Position-only features: [x, y, team_flag, is_kicker, dist_to_goal, dist_to_ball] (6 features)
    B) Position+Velocity features: [x, y, vx, vy, team_flag, is_kicker, dist_to_goal, dist_to_ball] (8 features)
    
    Same model architecture, same training procedure, same data splits.
    Only difference: whether vx, vy are included.
    """
    # Condition A: zero out velocity columns
    dataset_no_vel = zero_out_velocity_features(corner_dataset)
    
    # Condition B: full features
    dataset_full = corner_dataset
    
    results_no_vel = finetune_on_corners(pretrained_model_A, dataset_no_vel)
    results_full = finetune_on_corners(pretrained_model_B, dataset_full)
    
    print("=== ABLATION RESULTS ===")
    print(f"Position-only AUC:     {np.mean([r['auc'] for r in results_no_vel]):.3f}")
    print(f"Position+Velocity AUC: {np.mean([r['auc'] for r in results_full]):.3f}")
    print(f"Delta:                 {np.mean([r['auc'] for r in results_full]) - np.mean([r['auc'] for r in results_no_vel]):.3f}")
    
    # Statistical test
    from scipy.stats import ttest_rel
    t_stat, p_val = ttest_rel(
        [r['auc'] for r in results_full],
        [r['auc'] for r in results_no_vel]
    )
    print(f"Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
```

**Expected outcome**: Position+Velocity achieves AUC > 0.55 while Position-only stays at AUC ≈ 0.50. This validates the thesis hypothesis.

**Phase 3 deliverable**: Ablation results table showing position-only vs position+velocity performance with statistical significance test.

---

## Phase 4 (Optional): Data Augmentation & Extensions

Only proceed here if Phase 3 produces meaningful results (AUC > 0.55).

### Step 4.1: Add Metrica + SoccerTrack data

If DFL alone doesn't have enough corners, supplement with:
- **Metrica Sports** (3 matches): `kloppy.metrica` loader
- **SoccerTrack v2** (10 matches, amateur level): manual loading

This is sim-to-real in reverse — training on multiple data sources with domain adaptation.

### Step 4.2: Gaussian Process baseline

If GNN doesn't work due to data size, fall back to:

```python
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern

# Handcrafted features per corner (extracted from tracking data at t=0):
# - mean attacker velocity magnitude
# - mean defender velocity magnitude  
# - num attackers moving toward goal
# - num defenders standing still
# - attacker-defender velocity differential
# - spatial density in 6-yard box
# - numerical advantage in box
# ... (~10-15 features total, but now INCLUDING velocity-derived features)

gp = GaussianProcessClassifier(kernel=1.0 * Matern(nu=2.5))
gp.fit(X_train, y_train)
print(f"GP AUC: {roc_auc_score(y_test, gp.predict_proba(X_test)[:, 1]):.3f}")
```

This is the methodologically sound fallback: GPs are designed for exactly this regime (10-100 samples).

---

## Success Criteria

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| Data loaded | 7 matches, ≥50 corners | Pipeline works |
| Position-only AUC | ≈ 0.50 | Replicates 7.5 ECTS finding |
| Position+Velocity AUC | > 0.55 | Velocity hypothesis validated |
| Delta significance | p < 0.10 | Effect is real, not noise |

**What "success" looks like for the thesis**: Demonstrating a statistically significant improvement from velocity features, even if absolute performance is modest (e.g., AUC 0.58). The contribution is the controlled experiment showing velocity is a necessary condition, not achieving TacticAI-level F1.

**What "failure" looks like**: If position+velocity also yields AUC ≈ 0.50 on DFL data, the conclusion is that 70 corners is insufficient regardless of features. Document this honestly and fall back to GP analysis.

---

## Appendix A: Fallback — Gaussian Process on Handcrafted Features

If Phase 1 reveals < 30 corners in DFL, skip the GNN entirely and build a GP classifier on velocity-enriched handcrafted features from whatever corners are available. This is a valid, publishable analysis for a thesis.

## Appendix B: Key Dependencies

```
kloppy>=3.0
torch>=2.0
torch-geometric>=2.4
numpy
pandas
matplotlib
scikit-learn
scipy
```

## Appendix C: Repository Structure

```
CornerTactics/
├── data/
│   ├── dfl/                    # DFL open tracking data
│   ├── processed/              # Processed corner datasets
│   └── corner_dataset.pkl      
├── src/
│   ├── data_loading.py         # Phase 1: kloppy data loading
│   ├── graph_construction.py   # Phase 2: frame -> graph conversion
│   ├── model.py                # Phase 3: ST-GNN architecture
│   ├── pretrain.py             # Phase 3: open-play pretraining
│   ├── train.py                # Phase 3: corner fine-tuning
│   └── ablation.py             # Phase 3: velocity ablation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_results_analysis.ipynb
├── results/
│   ├── ablation_results.json
│   └── figures/
└── README.md
```
