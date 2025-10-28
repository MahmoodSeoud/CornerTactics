#!/bin/bash
# Build graphs with team_with_ball adjacency strategy
# This includes ball-player connections like US Soccer Fed's approach

echo "Building graphs with team_with_ball adjacency (includes ball connections)..."
echo "============================================================"

cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Create output directory
mkdir -p data/graphs/adjacency_team_with_ball

# Build StatsBomb graphs with team_with_ball adjacency
echo ""
echo "Step 1: Building StatsBomb graphs with ball connections..."
python scripts/build_skillcorner_graphs.py \
    --strategy team_with_ball \
    --dataset statsbomb \
    --output-dir data/graphs/adjacency_team_with_ball

if [ $? -ne 0 ]; then
    echo "Failed to build StatsBomb graphs"
    exit 1
fi

# Build SkillCorner graphs with team_with_ball adjacency
echo ""
echo "Step 2: Building SkillCorner temporal graphs with ball connections..."
python scripts/build_skillcorner_graphs.py \
    --strategy team_with_ball \
    --dataset skillcorner \
    --output-dir data/graphs/adjacency_team_with_ball

if [ $? -ne 0 ]; then
    echo "Failed to build SkillCorner graphs"
    exit 1
fi

echo ""
echo "============================================================"
echo "✅ Graph building complete with team_with_ball adjacency!"
echo "============================================================"
echo ""
echo "Graphs saved in: data/graphs/adjacency_team_with_ball/"
echo ""
echo "Key improvement: Ball is now a node connected to all players"
echo "This matches US Soccer Fed's approach for counterattacks"
echo ""
echo "Next step: Train GAT model with these improved graphs:"
echo "python scripts/train_gnn.py --model gat --graph-path data/graphs/adjacency_team_with_ball/combined_temporal_graphs.pkl"