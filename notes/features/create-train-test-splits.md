# Task 4: Create Train/Test Splits

## Objective
Create match-based stratified train/test splits (80/20) to prevent data leakage while maintaining similar outcome distributions.

## Requirements
1. Load `data/processed/corners_with_features.csv` (1,933 corners with 27 features)
2. Create **match-based** stratified split:
   - Group by match_id to ensure no match appears in both train and test
   - 80/20 split ratio
   - Stratify to maintain similar outcome class distributions
3. Save split indices to CSV files:
   - `data/processed/train_indices.csv`
   - `data/processed/test_indices.csv`

## Key Constraints
- **Match-based splitting is mandatory**: Prevents data leakage (corners from same match should not appear in both train and test)
- **Stratification needed**: Maintain class balance given 5.4:1 imbalance
- Class distribution should be similar in train and test sets

## Implementation Strategy
- Use match-level grouping
- Implement custom stratified group split (sklearn's StratifiedGroupKFold or manual)
- Validate distributions match expected ratios:
  - Ball Receipt: ~54%
  - Clearance: ~23%
  - Goalkeeper: ~10%
  - Other: ~12%

## Test-Driven Development Plan
1. Test loading features CSV correctly
2. Test match-based split (no match overlap between train/test)
3. Test split ratio is approximately 80/20
4. Test class distributions are similar in train and test
5. Test output CSV files are created with correct format

## Success Criteria
- Train/test sets have no overlapping match_ids
- Split ratio close to 80/20
- Class distributions in train and test are within ±2% of overall distribution
- CSV files contain correct row indices
