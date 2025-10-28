#!/usr/bin/env python3
"""
Quick script to inspect the structure of receiver-labeled graphs
"""

import pickle
from pathlib import Path

graph_path = Path("data/graphs/adjacency_team/combined_temporal_graphs_with_receiver.pkl")

print(f"Loading graphs from {graph_path}")
with open(graph_path, 'rb') as f:
    graphs = pickle.load(f)

print(f"Loaded {len(graphs)} graphs")
print()

# Inspect first graph
graph = graphs[0]
print("First graph attributes:")
print(f"  Type: {type(graph)}")
print(f"  Dir: {[attr for attr in dir(graph) if not attr.startswith('_')]}")
print()

print("Checking receiver-related attributes:")
print(f"  has 'receiver_player_id': {hasattr(graph, 'receiver_player_id')}")
print(f"  has 'receiver_index': {hasattr(graph, 'receiver_index')}")
print(f"  has 'player_ids': {hasattr(graph, 'player_ids')}")
print(f"  has 'receiver': {hasattr(graph, 'receiver')}")
print()

if hasattr(graph, 'receiver_player_id'):
    print(f"  receiver_player_id value: {graph.receiver_player_id}")
if hasattr(graph, 'receiver_index'):
    print(f"  receiver_index value: {graph.receiver_index}")
if hasattr(graph, 'player_ids'):
    print(f"  player_ids value: {graph.player_ids[:5] if graph.player_ids else None}")
if hasattr(graph, 'receiver'):
    print(f"  receiver value: {graph.receiver}")

# Check a few more graphs
print("\nChecking receiver_player_id in first 10 graphs:")
for i in range(min(10, len(graphs))):
    if hasattr(graphs[i], 'receiver_player_id'):
        print(f"  Graph {i}: {graphs[i].receiver_player_id}")
    else:
        print(f"  Graph {i}: NO ATTRIBUTE")
