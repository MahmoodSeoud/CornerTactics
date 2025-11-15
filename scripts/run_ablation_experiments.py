#!/usr/bin/env python3
"""
Run systematic ablation experiments based on exploration results.

Loads configs from results/exploration/ablation_plan.json and runs
each config sequentially, comparing clean vs leaky baselines.

Usage:
    python scripts/run_ablation_experiments.py [--config CONFIG_ID]
"""

import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import xgboost as xgb
from tqdm import tqdm

warnings.filterwarnings('ignore')
np.random.seed(42)


class AblationExperiment:
    """Run ablation experiments systematically."""

    def __init__(self,
                 data_path: str = "data/analysis/corner_sequences_full.json",
                 config_path: str = "results/exploration/ablation_plan.json",
                 output_dir: str = "results/ablation_experiments"):
        self.data_path = data_path
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data and configs
        print("Loading dataset...")
        with open(data_path, 'r') as f:
            self.corners = json.load(f)
        print(f"Loaded {len(self.corners)} corners")

        print("Loading ablation configs...")
        with open(config_path, 'r') as f:
            self.configs = json.load(f)
        print(f"Loaded {len(self.configs)} configs")

    def extract_features(self, corner: Dict, features_to_remove: List[str] = None) -> Dict:
        """Extract raw features, optionally removing specified ones."""
        if features_to_remove is None:
            features_to_remove = []

        features = {}
        event = corner['corner_event']

        # Basic event features
        feature_mapping = {
            'minute': event.get('minute'),
            'second': event.get('second'),
            'period': event.get('period'),
            'duration': event.get('duration'),
            'possession': event.get('possession'),
        }

        # Location
        location = event.get('location', [None, None])
        feature_mapping['location_x'] = location[0] if len(location) > 0 else None
        feature_mapping['location_y'] = location[1] if len(location) > 1 else None

        # Team/player IDs
        feature_mapping['team_id'] = event.get('team', {}).get('id')
        feature_mapping['player_id'] = event.get('player', {}).get('id')
        feature_mapping['possession_team_id'] = event.get('possession_team', {}).get('id')
        feature_mapping['position_id'] = event.get('position', {}).get('id')
        feature_mapping['play_pattern_id'] = event.get('play_pattern', {}).get('id')

        # Pass details (potentially leaky)
        pass_info = event.get('pass', {})
        if pass_info:
            end_loc = pass_info.get('end_location', [None, None])
            feature_mapping['end_location_x'] = end_loc[0] if len(end_loc) > 0 else None
            feature_mapping['end_location_y'] = end_loc[1] if len(end_loc) > 1 else None
            feature_mapping['pass_length'] = pass_info.get('length')
            feature_mapping['pass_angle'] = pass_info.get('angle')
            feature_mapping['shot_assist'] = 1 if pass_info.get('shot_assist') else 0
            feature_mapping['inswinging'] = 1 if pass_info.get('inswinging') else 0
            feature_mapping['switch'] = 1 if pass_info.get('switch') else 0

            # IDs
            feature_mapping['height_id'] = pass_info.get('height', {}).get('id')
            feature_mapping['body_part_id'] = pass_info.get('body_part', {}).get('id')
            feature_mapping['technique_id'] = pass_info.get('technique', {}).get('id')

        # Related events count
        related = event.get('related_events', [])
        feature_mapping['related_events_count'] = len(related) if isinstance(related, list) else 0

        # Remove specified features
        for feature_name in features_to_remove:
            if feature_name in feature_mapping:
                del feature_mapping[feature_name]

        return feature_mapping

    def prepare_dataset(self, features_to_remove: List[str] = None):
        """Prepare train/val/test splits with optional feature removal."""
        print(f"\nPreparing dataset (removing {len(features_to_remove or [])} features)...")

        # Extract features for all corners
        X_list = []
        y_list = []

        for corner in tqdm(self.corners, desc="Extracting features"):
            features = self.extract_features(corner, features_to_remove)

            # Get outcome label (next action type)
            next_events = corner.get('next_events', [])
            if next_events and len(next_events) > 0:
                next_action = next_events[0].get('type', {}).get('name', 'Unknown')
            else:
                next_action = 'Unknown'

            X_list.append(features)
            y_list.append(next_action)

        # Convert to DataFrame
        X_df = pd.DataFrame(X_list)
        y_series = pd.Series(y_list)

        # Fill NaN values
        X_df = X_df.fillna(0)

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_series)

        # Check class distribution
        from collections import Counter
        class_counts = Counter(y_encoded)
        min_samples = min(class_counts.values())

        print(f"Class distribution: {class_counts}")
        print(f"Min samples per class: {min_samples}")

        # Filter out classes with < 3 samples (need at least 3 for train/val/test split)
        if min_samples < 3:
            print(f"Filtering out classes with < 3 samples...")
            valid_mask = np.array([class_counts[y] >= 3 for y in y_encoded])
            X_df = X_df[valid_mask]
            y_encoded = y_encoded[valid_mask]

            # Re-encode to get consecutive labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_series[valid_mask])

            class_counts = Counter(y_encoded)
            print(f"After filtering: {class_counts}")

        # Use stratified split to ensure all classes in all splits
        # Split: 70/15/15
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_df, y_encoded, test_size=0.3, random_state=42,
            stratify=y_encoded
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42,
            stratify=y_temp
        )

        # Verify all splits have all classes
        train_classes = set(y_train)
        val_classes = set(y_val)
        test_classes = set(y_test)
        all_classes = set(y_encoded)

        print(f"Classes in train: {sorted(train_classes)}")
        print(f"Classes in val: {sorted(val_classes)}")
        print(f"Classes in test: {sorted(test_classes)}")

        if train_classes != all_classes:
            print(f"WARNING: Training set missing classes: {all_classes - train_classes}")
            # This shouldn't happen with stratified split, but if it does, we'll handle it

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"Features: {X_df.shape[1]}")
        print(f"Classes: {len(le.classes_)}")

        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': list(X_df.columns),
            'label_encoder': le
        }

    def train_xgboost(self, data: Dict) -> Dict:
        """Train XGBoost model."""
        print("\nTraining XGBoost...")

        # Verify labels are consecutive from 0
        y_all = np.concatenate([data['y_train'], data['y_val'], data['y_test']])
        unique_labels = np.unique(y_all)
        print(f"Unique labels: {unique_labels}")

        # Count actual classes
        num_classes = len(unique_labels)
        print(f"Number of classes: {num_classes}")

        # Best hyperparameters (from baseline)
        # Use softprob instead of softmax to avoid class inference issues
        params = {
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist',
            'eval_metric': 'mlogloss'
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            data['X_train'], data['y_train'],
            eval_set=[(data['X_val'], data['y_val'])],
            verbose=False
        )

        # Predictions (softprob returns probabilities, convert to classes)
        y_pred_proba = model.predict(data['X_test'])
        y_pred_test = np.argmax(y_pred_proba, axis=1) if len(y_pred_proba.shape) > 1 else y_pred_proba

        # Metrics
        accuracy = accuracy_score(data['y_test'], y_pred_test)
        macro_f1 = f1_score(data['y_test'], y_pred_test, average='macro')
        weighted_f1 = f1_score(data['y_test'], y_pred_test, average='weighted')

        return {
            'model': model,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'predictions': y_pred_test,
            'feature_importance': dict(zip(
                data['feature_names'],
                model.feature_importances_
            ))
        }

    def run_config(self, config: Dict) -> Dict:
        """Run a single ablation config."""
        print("\n" + "="*60)
        print(f"Running Config {config['config_id']}: {config['name']}")
        print("="*60)
        print(f"Description: {config['description']}")
        print(f"Purpose: {config['purpose']}")
        print(f"Features to remove: {len(config['features_to_remove'])}")
        if config['features_to_remove']:
            print(f"  - {', '.join(config['features_to_remove'])}")

        # Prepare data
        data = self.prepare_dataset(config['features_to_remove'])

        # Train model
        results = self.train_xgboost(data)

        # Summary
        print(f"\nResults:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Macro F1: {results['macro_f1']:.4f}")
        print(f"  Weighted F1: {results['weighted_f1']:.4f}")

        # Save results
        output = {
            'config_id': config['config_id'],
            'config_name': config['name'],
            'description': config['description'],
            'features_removed': config['features_to_remove'],
            'num_features': len(data['feature_names']),
            'accuracy': results['accuracy'],
            'macro_f1': results['macro_f1'],
            'weighted_f1': results['weighted_f1']
        }

        return output

    def run_all_configs(self):
        """Run all ablation configs."""
        print("\n" + "="*60)
        print("Starting Ablation Experiments")
        print("="*60)
        print(f"Total configs: {len(self.configs)}")

        all_results = []

        for config in self.configs:
            result = self.run_config(config)
            all_results.append(result)

            # Save incremental results
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(self.output_dir / "ablation_results.csv", index=False)

        # Final summary
        print("\n" + "="*60)
        print("ABLATION EXPERIMENTS COMPLETE")
        print("="*60)

        results_df = pd.DataFrame(all_results)
        print("\nResults Summary:")
        print(results_df[['config_id', 'config_name', 'num_features', 'accuracy', 'macro_f1']])

        # Compare to baseline
        baseline = results_df[results_df['config_id'] == 0].iloc[0]
        clean_baseline = results_df[results_df['config_id'] == 6].iloc[0]

        print(f"\n🔴 Baseline (all features): Acc={baseline['accuracy']:.4f}, F1={baseline['macro_f1']:.4f}")
        print(f"✅ Clean baseline (no leakage): Acc={clean_baseline['accuracy']:.4f}, F1={clean_baseline['macro_f1']:.4f}")
        print(f"📉 Leakage impact: {(baseline['accuracy'] - clean_baseline['accuracy'])*100:.2f}% accuracy drop")

        # Save final results
        results_df.to_csv(self.output_dir / "ablation_results_final.csv", index=False)
        print(f"\nResults saved to: {self.output_dir}")

        return results_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument('--config', type=int, default=None,
                       help='Run specific config ID only (default: all)')
    args = parser.parse_args()

    # Initialize experiment
    experiment = AblationExperiment()

    if args.config is not None:
        # Run single config
        config = [c for c in experiment.configs if c['config_id'] == args.config][0]
        result = experiment.run_config(config)
        print("\nResult:", json.dumps(result, indent=2))
    else:
        # Run all configs
        experiment.run_all_configs()


if __name__ == "__main__":
    main()
