#!/usr/bin/env python3
"""
Comprehensive analysis of corner kick outcomes from existing dataset.
Proposes classification taxonomies for ML prediction.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def analyze_existing_outcomes():
    """Analyze the existing labeled corner dataset."""

    # Load the processed data
    df = pd.read_csv('data/raw/statsbomb/statsbomb/corners_360_with_outcomes.csv')

    print("="*80)
    print("CORNER KICK OUTCOME ANALYSIS - FULL DATASET")
    print("="*80)
    print(f"\nTotal corners analyzed: {len(df)}")

    # 1. Current 5-class distribution
    print("\n1. CURRENT OUTCOME CATEGORIES (5 classes):")
    print("-"*50)
    outcome_counts = df['outcome_category'].value_counts()
    for outcome, count in outcome_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {outcome:<15} {count:4d} ({pct:5.1f}%)")

    # 2. Detailed outcome types
    print("\n2. DETAILED OUTCOME TYPES:")
    print("-"*50)
    type_counts = df['outcome_type'].value_counts()
    for outcome_type, count in type_counts.head(15).items():
        pct = (count / len(df)) * 100
        print(f"  {outcome_type:<30} {count:4d} ({pct:5.1f}%)")

    # 3. Temporal analysis
    print("\n3. TIME TO OUTCOME ANALYSIS:")
    print("-"*50)
    time_stats = df['time_to_outcome'].describe()
    print(f"  Mean:   {time_stats['mean']:.2f} seconds")
    print(f"  Median: {time_stats['50%']:.2f} seconds")
    print(f"  Std:    {time_stats['std']:.2f} seconds")
    print(f"  Max:    {time_stats['max']:.2f} seconds")

    # 4. Team analysis (attacking vs defending outcomes)
    print("\n4. OUTCOME BY TEAM (Attacking vs Defending):")
    print("-"*50)
    same_team_outcomes = df.groupby(['outcome_category', 'same_team']).size().unstack(fill_value=0)
    print(same_team_outcomes)

    # 5. Shot outcome breakdown
    print("\n5. SHOT OUTCOME BREAKDOWN:")
    print("-"*50)
    shot_df = df[df['outcome_category'].isin(['Goal', 'Shot'])]
    shot_outcomes = shot_df['shot_outcome'].value_counts()
    for outcome, count in shot_outcomes.items():
        pct = (count / len(shot_df)) * 100
        print(f"  {outcome:<20} {count:4d} ({pct:5.1f}%)")

    return df

def propose_taxonomies(df):
    """Propose different classification taxonomies."""

    print("\n" + "="*80)
    print("PROPOSED CLASSIFICATION TAXONOMIES")
    print("="*80)

    # OPTION 1: Binary classification
    print("\n### OPTION 1: BINARY CLASSIFICATION")
    print("-"*50)
    print("Simple success/failure classification:")

    # Success = Goal or Shot
    df['binary_outcome'] = df['outcome_category'].apply(
        lambda x: 'Success' if x in ['Goal', 'Shot'] else 'Failure'
    )
    binary_counts = df['binary_outcome'].value_counts()
    for outcome, count in binary_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {outcome:<15} {count:4d} ({pct:5.1f}%)")

    print("\nJustification:")
    print("  - Simple and interpretable")
    print("  - Clear tactical meaning (created shooting chance vs not)")
    print("  - Balanced classes (~18% vs 82%)")
    print("  - Easy to evaluate model performance")

    # OPTION 2: 3-class (Shot/Clearance/Possession)
    print("\n### OPTION 2: 3-CLASS TAXONOMY")
    print("-"*50)
    print("Primary outcome classification:")

    def map_to_3_class(row):
        if row['outcome_category'] in ['Goal', 'Shot']:
            return 'Shot'
        elif row['outcome_category'] == 'Clearance':
            return 'Clearance'
        else:
            return 'Possession'

    df['three_class'] = df.apply(map_to_3_class, axis=1)
    three_counts = df['three_class'].value_counts()
    for outcome, count in three_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {outcome:<15} {count:4d} ({pct:5.1f}%)")

    print("\nJustification:")
    print("  - Captures key tactical outcomes")
    print("  - Shot: Attacking success (includes goals)")
    print("  - Clearance: Defensive success")
    print("  - Possession: Continued play")
    print("  - Reasonable class balance")
    print("  - Matches common football analysis")

    # OPTION 3: 4-class (Goal/Shot/Clearance/Other)
    print("\n### OPTION 3: 4-CLASS TAXONOMY")
    print("-"*50)
    print("Separating goals from shots:")

    def map_to_4_class(row):
        if row['outcome_category'] == 'Goal':
            return 'Goal'
        elif row['outcome_category'] == 'Shot':
            return 'Shot'
        elif row['outcome_category'] == 'Clearance':
            return 'Clearance'
        else:
            return 'Other'

    df['four_class'] = df.apply(map_to_4_class, axis=1)
    four_counts = df['four_class'].value_counts()
    for outcome, count in four_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {outcome:<15} {count:4d} ({pct:5.1f}%)")

    print("\nJustification:")
    print("  - Most granular for high-value outcomes")
    print("  - Separates goals (ultimate success)")
    print("  - BUT: Very imbalanced (1.3% goals)")
    print("  - May be difficult for ML models")
    print("  - Consider only if goal prediction is critical")

    # OPTION 4: Attacking focus (3-class)
    print("\n### OPTION 4: ATTACKING-FOCUSED TAXONOMY")
    print("-"*50)
    print("From attacking team perspective:")

    def map_attacking_focus(row):
        if row['outcome_category'] in ['Goal', 'Shot']:
            return 'Attacking_Success'
        elif row['outcome_category'] == 'Clearance':
            return 'Defensive_Win'
        else:
            # Loss, Possession, Duel, etc.
            if row['same_team'] == True:
                return 'Maintained_Attack'
            else:
                return 'Defensive_Win'

    df['attacking_focus'] = df.apply(map_attacking_focus, axis=1)
    attacking_counts = df['attacking_focus'].value_counts()
    for outcome, count in attacking_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {outcome:<20} {count:4d} ({pct:5.1f}%)")

    print("\nJustification:")
    print("  - Clear tactical interpretation")
    print("  - Useful for coaching decisions")
    print("  - Focuses on attacking team's perspective")
    print("  - Better class balance than 4-class")

    # OPTION 5: Time-based urgency
    print("\n### OPTION 5: TIME-BASED URGENCY")
    print("-"*50)
    print("Based on how quickly outcome occurs:")

    def map_time_urgency(row):
        time = row['time_to_outcome']
        category = row['outcome_category']

        if pd.isna(time):
            return 'Extended_Play'
        elif time <= 2.0:  # Within 2 seconds
            if category in ['Goal', 'Shot']:
                return 'Quick_Shot'
            elif category == 'Clearance':
                return 'Quick_Clear'
            else:
                return 'Quick_Transition'
        else:  # After 2 seconds
            if category in ['Goal', 'Shot']:
                return 'Delayed_Shot'
            else:
                return 'Extended_Play'

    df['time_based'] = df.apply(map_time_urgency, axis=1)
    time_counts = df['time_based'].value_counts()
    for outcome, count in time_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {outcome:<20} {count:4d} ({pct:5.1f}%)")

    print("\nJustification:")
    print("  - Captures tempo and urgency")
    print("  - Distinguishes direct vs worked corners")
    print("  - Useful for tactical preparation")
    print("  - Novel approach for corner analysis")

    return df

def recommend_taxonomy():
    """Provide final recommendation."""

    print("\n" + "="*80)
    print("RECOMMENDATION FOR ML PREDICTION")
    print("="*80)

    print("""
### PRIMARY RECOMMENDATION: 3-Class Taxonomy (Shot/Clearance/Possession)

**Classes:**
1. **Shot** (18.2%): Goal or shot attempt - attacking success
2. **Clearance** (51.8%): Defensive clearance - defensive success
3. **Possession** (30.0%): Continued possession by either team

**Reasoning:**
- Balanced enough for ML training (no class < 15%)
- Clear tactical interpretation
- Matches how coaches analyze corners
- Can be easily extended to 4-class later if needed
- Aligns with paper methodology (action classification)

**Implementation:**
```python
def label_corner_outcome(outcome_category, outcome_type):
    if outcome_category in ['Goal', 'Shot']:
        return 0  # Shot class
    elif outcome_category == 'Clearance':
        return 1  # Clearance class
    else:
        return 2  # Possession class
```

### ALTERNATIVE: Binary Classification (Success/Failure)

Use this if:
- You need maximum model performance
- Focus is purely on "did we create a chance?"
- Interpretability is paramount

### ADVANCED: Multi-task Learning

Train separate models for:
1. Primary outcome (3-class)
2. Time to outcome (regression)
3. Shot outcome (if shot occurs)

This provides richer predictions for tactical analysis.
""")

if __name__ == "__main__":
    # Analyze existing outcomes
    df = analyze_existing_outcomes()

    # Propose taxonomies
    df_with_taxonomies = propose_taxonomies(df)

    # Final recommendation
    recommend_taxonomy()

    # Save augmented dataset
    output_path = 'data/raw/statsbomb/statsbomb/corners_with_taxonomies.csv'
    df_with_taxonomies.to_csv(output_path, index=False)
    print(f"\nAugmented dataset saved to: {output_path}")