#!/usr/bin/env python3
"""
Explore dataset to guide ablation design.

Phases:
  1. Train baseline (already done - Nov 13)
  2. Analyze feature importance
  3. Compute feature correlations
  4. Feature-receiver correlation
  5. Identify leakage suspects
  6. Generate ablation configs

Usage:
  python scripts/explore_dataset.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import pointbiserialr, pearsonr
import warnings

warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

class DatasetExplorer:
    """Systematic dataset exploration for ablation design."""

    def __init__(self,
                 dataset_path: str = "data/analysis/corner_sequences_full.json",
                 feature_importance_path: str = "results/feature_importance_event.csv",
                 output_dir: str = "results/exploration"):
        self.dataset_path = dataset_path
        self.feature_importance_path = feature_importance_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        print("Loading dataset...")
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        self.df = pd.DataFrame(data)

        # Load feature importance
        print("Loading feature importance...")
        self.feature_importance = pd.read_csv(feature_importance_path)

        print(f"Dataset: {len(self.df)} corners")
        print(f"Features: {len(self.feature_importance)} features")

    def analyze_importance(self) -> Dict:
        """Phase 2: Analyze feature importance tiers."""
        print("\n" + "="*60)
        print("Phase 2: Analyzing Feature Importance Tiers")
        print("="*60)

        fi = self.feature_importance

        # Define tiers
        tier1 = fi[fi['importance_percentage'] > 10]
        tier2 = fi[(fi['importance_percentage'] >= 5) & (fi['importance_percentage'] <= 10)]
        tier3 = fi[(fi['importance_percentage'] >= 1) & (fi['importance_percentage'] < 5)]
        tier4 = fi[fi['importance_percentage'] < 1]

        print(f"\nTier 1 (>10%): {len(tier1)} features")
        for _, row in tier1.iterrows():
            print(f"  - {row['feature_name']}: {row['importance_percentage']:.2f}%")

        print(f"\nTier 2 (5-10%): {len(tier2)} features")
        for _, row in tier2.iterrows():
            print(f"  - {row['feature_name']}: {row['importance_percentage']:.2f}%")

        print(f"\nTier 3 (1-5%): {len(tier3)} features")
        for _, row in tier3.iterrows():
            print(f"  - {row['feature_name']}: {row['importance_percentage']:.2f}%")

        print(f"\nTier 4 (<1%): {len(tier4)} features")
        for _, row in tier4.iterrows():
            print(f"  - {row['feature_name']}: {row['importance_percentage']:.2f}%")

        # Save tiers
        tiers_data = {
            "tier1_high": list(tier1['feature_name']),
            "tier2_medium": list(tier2['feature_name']),
            "tier3_low": list(tier3['feature_name']),
            "tier4_very_low": list(tier4['feature_name'])
        }

        output_path = self.output_dir / "feature_tiers.json"
        with open(output_path, 'w') as f:
            json.dump(tiers_data, f, indent=2)
        print(f"\nSaved: {output_path}")

        return tiers_data

    def compute_correlations(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """Phase 3: Feature correlation matrix."""
        print("\n" + "="*60)
        print("Phase 3: Computing Feature Correlations")
        print("="*60)

        # Get numeric features only
        numeric_features = []
        for col in self.df.columns:
            if col in ['receiver_name', 'receiver_player_id', 'next_action_type']:
                continue
            try:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                if self.df[col].notna().sum() > 0:
                    numeric_features.append(col)
            except:
                continue

        print(f"\nNumeric features: {len(numeric_features)}")

        # Compute correlation matrix
        df_numeric = self.df[numeric_features].copy()
        df_numeric = df_numeric.fillna(0)

        corr_matrix = df_numeric.corr()

        # Find high correlations (|r| > 0.8)
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.8:
                    high_corr.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(r)
                    })

        print(f"\nFound {len(high_corr)} highly correlated pairs (|r| > 0.8):")
        for pair in high_corr[:20]:  # Show top 20
            print(f"  {pair['feature1']} <-> {pair['feature2']}: r={pair['correlation']:.3f}")

        # Save correlation matrix
        corr_path = self.output_dir / "feature_correlation_matrix.csv"
        corr_matrix.to_csv(corr_path)
        print(f"\nSaved correlation matrix: {corr_path}")

        # Save high correlations
        high_corr_path = self.output_dir / "high_correlations.json"
        with open(high_corr_path, 'w') as f:
            json.dump(high_corr, f, indent=2)
        print(f"Saved high correlations: {high_corr_path}")

        return corr_matrix, high_corr

    def receiver_correlation(self) -> pd.DataFrame:
        """Phase 4: Feature-receiver correlation."""
        print("\n" + "="*60)
        print("Phase 4: Feature-Receiver Correlation Analysis")
        print("="*60)

        # Check if receiver exists
        if 'receiver_player_id' not in self.df.columns:
            print("WARNING: No receiver_player_id column found!")
            print("Skipping receiver correlation analysis.")
            return pd.DataFrame()

        # Encode receiver as numeric
        df_analysis = self.df.copy()
        df_analysis['receiver_encoded'] = pd.factorize(df_analysis['receiver_player_id'])[0]

        # Get numeric features
        numeric_features = []
        for col in self.df.columns:
            if col in ['receiver_name', 'receiver_player_id', 'next_action_type', 'receiver_encoded']:
                continue
            try:
                df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
                if df_analysis[col].notna().sum() > 0:
                    numeric_features.append(col)
            except:
                continue

        print(f"\nComputing correlation with receiver for {len(numeric_features)} features...")

        # Compute correlations
        receiver_corr = []
        for feature in numeric_features:
            # Fill NaN with 0
            feature_values = df_analysis[feature].fillna(0)
            receiver_values = df_analysis['receiver_encoded']

            # Remove rows where either is NaN
            mask = ~(feature_values.isna() | receiver_values.isna())
            feature_clean = feature_values[mask]
            receiver_clean = receiver_values[mask]

            if len(feature_clean) < 10:  # Skip if too few valid values
                continue

            try:
                r, p_value = pearsonr(feature_clean, receiver_clean)
                receiver_corr.append({
                    'feature': feature,
                    'correlation': r,
                    'abs_correlation': abs(r),
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
            except:
                continue

        # Sort by absolute correlation
        receiver_corr = sorted(receiver_corr, key=lambda x: x['abs_correlation'], reverse=True)

        print(f"\nTop 15 features correlated with receiver:")
        for i, item in enumerate(receiver_corr[:15], 1):
            sig = "***" if item['significant'] else ""
            print(f"  {i:2d}. {item['feature']:25s}: r={item['correlation']:+.3f} (p={item['p_value']:.4f}) {sig}")

        # Save results
        receiver_corr_df = pd.DataFrame(receiver_corr)
        output_path = self.output_dir / "feature_receiver_correlation.csv"
        receiver_corr_df.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")

        return receiver_corr_df

    def identify_leakage(self, receiver_corr_df: pd.DataFrame) -> List[Dict]:
        """Phase 5: Identify leakage suspects."""
        print("\n" + "="*60)
        print("Phase 5: Identifying Leakage Suspects")
        print("="*60)

        # Post-kick features (conceptual)
        post_kick_features = [
            'end_location_x', 'end_location_y',
            'pass_length', 'pass_angle',
            'shot_assist', 'switch',
            'second'  # time of next event might leak
        ]

        # Merge importance and receiver correlation
        fi = self.feature_importance.set_index('feature_name')

        leakage_suspects = []

        for _, row in self.feature_importance.iterrows():
            feature = row['feature_name']
            importance = row['importance_percentage']

            # Check importance
            high_importance = importance > 10
            medium_importance = importance > 5

            # Check if post-kick
            is_post_kick = feature in post_kick_features

            # Check receiver correlation
            high_correlation = False
            receiver_corr = 0.0
            if not receiver_corr_df.empty and feature in receiver_corr_df['feature'].values:
                receiver_corr = receiver_corr_df[receiver_corr_df['feature'] == feature]['abs_correlation'].values[0]
                high_correlation = receiver_corr > 0.5

            # Leakage criteria
            reason = None
            if high_importance and high_correlation:
                reason = f"High importance ({importance:.1f}%) + high receiver correlation ({receiver_corr:.3f})"
            elif high_importance and is_post_kick:
                reason = f"Post-kick information + high importance ({importance:.1f}%)"
            elif medium_importance and is_post_kick:
                reason = f"Post-kick information + medium importance ({importance:.1f}%)"
            elif is_post_kick and importance > 1:
                reason = f"Post-kick information (importance: {importance:.1f}%)"

            if reason:
                leakage_suspects.append({
                    'feature': feature,
                    'importance': importance,
                    'receiver_correlation': receiver_corr,
                    'is_post_kick': is_post_kick,
                    'reason': reason
                })

        # Sort by importance
        leakage_suspects = sorted(leakage_suspects, key=lambda x: x['importance'], reverse=True)

        print(f"\nFound {len(leakage_suspects)} suspected leakage features:\n")
        for i, suspect in enumerate(leakage_suspects, 1):
            print(f"{i:2d}. {suspect['feature']:25s}: {suspect['reason']}")

        # Save suspects
        output_path = self.output_dir / "leakage_suspects.json"
        with open(output_path, 'w') as f:
            json.dump(leakage_suspects, f, indent=2)
        print(f"\nSaved: {output_path}")

        return leakage_suspects

    def generate_configs(self, tiers_data: Dict, leakage_suspects: List[Dict],
                        high_corr: List[Dict]) -> List[Dict]:
        """Phase 6: Design ablation configs."""
        print("\n" + "="*60)
        print("Phase 6: Generating Ablation Config Plan")
        print("="*60)

        configs = []

        # Config 0: Baseline (all features)
        configs.append({
            "config_id": 0,
            "name": "baseline_all_features",
            "description": "All 23 features (reference)",
            "features_to_remove": [],
            "purpose": "Baseline performance with potential leakage"
        })

        # Configs 1-N: Remove leakage suspects one at a time
        for i, suspect in enumerate(leakage_suspects[:5], 1):  # Top 5 suspects
            configs.append({
                "config_id": i,
                "name": f"remove_{suspect['feature']}",
                "description": f"Remove {suspect['feature']} only",
                "features_to_remove": [suspect['feature']],
                "purpose": f"Test if {suspect['feature']} is leaking (importance: {suspect['importance']:.1f}%)"
            })

        # Config: Remove ALL suspected leakage
        config_id = len(configs)
        configs.append({
            "config_id": config_id,
            "name": "clean_baseline",
            "description": "Remove all suspected leakage features",
            "features_to_remove": [s['feature'] for s in leakage_suspects],
            "purpose": "Fair comparison baseline without leakage"
        })

        # Config: Remove Tier 4 (very low importance)
        config_id += 1
        configs.append({
            "config_id": config_id,
            "name": "remove_tier4",
            "description": "Remove all features with <1% importance",
            "features_to_remove": tiers_data['tier4_very_low'],
            "purpose": "Test if weak features add noise"
        })

        # Config: Keep only Tier 1+2
        config_id += 1
        configs.append({
            "config_id": config_id,
            "name": "keep_tier1_tier2",
            "description": "Keep only features with >5% importance",
            "features_to_remove": tiers_data['tier3_low'] + tiers_data['tier4_very_low'],
            "purpose": "Focus on most important features"
        })

        # Config: Keep only Tier 1
        config_id += 1
        configs.append({
            "config_id": config_id,
            "name": "keep_tier1_only",
            "description": "Keep only features with >10% importance",
            "features_to_remove": tiers_data['tier2_medium'] + tiers_data['tier3_low'] + tiers_data['tier4_very_low'],
            "purpose": "Most aggressive pruning"
        })

        print(f"\nGenerated {len(configs)} ablation configs:\n")
        for config in configs:
            print(f"Config {config['config_id']}: {config['name']}")
            print(f"  Description: {config['description']}")
            print(f"  Purpose: {config['purpose']}")
            print(f"  Features to remove: {len(config['features_to_remove'])}")
            print()

        # Save configs
        output_path = self.output_dir / "ablation_plan.json"
        with open(output_path, 'w') as f:
            json.dump(configs, f, indent=2)
        print(f"Saved ablation plan: {output_path}")

        return configs


def main():
    """Run all exploration phases."""
    print("="*60)
    print("Dataset Exploration for Ablation Design")
    print("="*60)

    # Initialize explorer
    explorer = DatasetExplorer()

    # Phase 2: Analyze importance
    print("\n" + "="*60)
    print("Running Phase 2: Feature Importance Analysis")
    print("="*60)
    tiers_data = explorer.analyze_importance()

    # Phase 3: Correlations
    print("\n" + "="*60)
    print("Running Phase 3: Feature Correlation Analysis")
    print("="*60)
    corr_matrix, high_corr = explorer.compute_correlations()

    # Phase 4: Receiver correlation
    print("\n" + "="*60)
    print("Running Phase 4: Receiver Correlation Analysis")
    print("="*60)
    receiver_corr_df = explorer.receiver_correlation()

    # Phase 5: Leakage suspects
    print("\n" + "="*60)
    print("Running Phase 5: Leakage Detection")
    print("="*60)
    leakage_suspects = explorer.identify_leakage(receiver_corr_df)

    # Phase 6: Generate configs
    print("\n" + "="*60)
    print("Running Phase 6: Ablation Config Generation")
    print("="*60)
    configs = explorer.generate_configs(tiers_data, leakage_suspects, high_corr)

    # Summary
    print("\n" + "="*60)
    print("EXPLORATION COMPLETE!")
    print("="*60)
    print(f"\nOutputs saved to: results/exploration/")
    print("\nFiles created:")
    print("  - feature_tiers.json")
    print("  - feature_correlation_matrix.csv")
    print("  - high_correlations.json")
    print("  - feature_receiver_correlation.csv")
    print("  - leakage_suspects.json")
    print("  - ablation_plan.json")
    print("\nNext steps:")
    print("  1. Review leakage_suspects.json")
    print("  2. Review ablation_plan.json")
    print("  3. Implement configs in feature registry")
    print("  4. Run ablation experiments")
    print("="*60)


if __name__ == "__main__":
    main()
