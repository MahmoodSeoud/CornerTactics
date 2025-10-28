#!/usr/bin/env python3
"""
Inspect USSF counterattack dataset structure
"""
import pickle
import numpy as np
import sys

def inspect_ussf_data(filepath):
    """Load and inspect USSF data with torch compatibility"""
    print(f"Loading data from: {filepath}")
    print("=" * 80)

    try:
        # Try loading with torch first (in case it has torch tensors)
        try:
            import torch
            data = torch.load(filepath, map_location='cpu')
            print("✓ Loaded successfully with torch.load()")
        except:
            # Fall back to regular pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print("✓ Loaded successfully with pickle.load()")

        print(f"\n=== DATA STRUCTURE ===")
        print(f"Type: {type(data)}")

        if isinstance(data, dict):
            print(f"Top-level keys: {list(data.keys())}")

            for key in data.keys():
                print(f"\n--- Key: '{key}' ---")
                value = data[key]
                print(f"Type: {type(value)}")

                if isinstance(value, dict):
                    print(f"Sub-keys: {list(value.keys())}")
                    for subkey in value.keys():
                        subvalue = value[subkey]
                        if isinstance(subvalue, list) and len(subvalue) > 0:
                            print(f"  '{subkey}': List of {len(subvalue)} items")
                            first_item = subvalue[0]
                            print(f"    First item type: {type(first_item)}")
                            if isinstance(first_item, np.ndarray):
                                print(f"    First item shape: {first_item.shape}")
                                print(f"    First item dtype: {first_item.dtype}")
                        elif isinstance(subvalue, np.ndarray):
                            print(f"  '{subkey}': Array shape {subvalue.shape}")
                        else:
                            print(f"  '{subkey}': {type(subvalue)}")

                elif isinstance(value, list) and len(value) > 0:
                    print(f"List of {len(value)} items")
                    first_item = value[0]
                    print(f"First item type: {type(first_item)}")
                    if isinstance(first_item, (list, tuple)) and len(first_item) > 0:
                        print(f"First item length: {len(first_item)}")
                        print(f"First item sample: {first_item}")
                    elif isinstance(first_item, np.ndarray):
                        print(f"First item shape: {first_item.shape}")

            # Detailed analysis of the structure
            print("\n" + "=" * 80)
            print("=== DETAILED ANALYSIS ===")

            # Check for 'normal' adjacency matrix structure
            if 'normal' in data:
                normal_data = data['normal']
                if 'x' in normal_data and len(normal_data['x']) > 0:
                    print(f"\n✓ Node Features (x):")
                    print(f"  Total samples: {len(normal_data['x'])}")
                    print(f"  First sample shape: {normal_data['x'][0].shape}")
                    print(f"  (Players × Features)")
                    print(f"  Sample data:\n{normal_data['x'][0][:3]}")

                if 'a' in normal_data and len(normal_data['a']) > 0:
                    print(f"\n✓ Adjacency Matrix (a):")
                    print(f"  Total samples: {len(normal_data['a'])}")
                    print(f"  First sample shape: {normal_data['a'][0].shape}")
                    print(f"  Number of edges: {np.sum(normal_data['a'][0])}")
                    print(f"  Sparsity: {1 - (np.sum(normal_data['a'][0]) / normal_data['a'][0].size):.2%}")

                if 'e' in normal_data and len(normal_data['e']) > 0:
                    print(f"\n✓ Edge Features (e):")
                    print(f"  Total samples: {len(normal_data['e'])}")
                    print(f"  First sample shape: {normal_data['e'][0].shape}")

            if 'binary' in data:
                labels = data['binary']
                print(f"\n✓ Labels (binary):")
                print(f"  Total samples: {len(labels)}")
                positive = sum([x[0] for x in labels])
                negative = len(labels) - positive
                print(f"  Positive (success): {positive} ({positive/len(labels)*100:.1f}%)")
                print(f"  Negative (fail): {negative} ({negative/len(labels)*100:.1f}%)")
                print(f"  First 20 labels: {[x[0] for x in labels[:20]]}")

            # Check for imbalanced dataset structure
            if 'id' in data:
                print(f"\n✓ Sequence IDs (id):")
                print(f"  Total frames: {len(data['id'])}")
                unique_sequences = len(set(data['id']))
                print(f"  Unique sequences: {unique_sequences}")
                print(f"  Avg frames per sequence: {len(data['id']) / unique_sequences:.1f}")

            if 'node_feature_names' in data:
                print(f"\n✓ Node Feature Names:")
                print(f"  {data['node_feature_names']}")

            if 'edge_feature_names' in data:
                print(f"\n✓ Edge Feature Names:")
                print(f"  {data['edge_feature_names']}")

        print("\n" + "=" * 80)
        print("✓ Inspection complete!")

    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    filepath = "data/raw/ussf_data_sample.pkl"
    if len(sys.argv) > 1:
        filepath = sys.argv[1]

    inspect_ussf_data(filepath)
