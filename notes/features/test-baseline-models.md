# Test Baseline Models Feature

## Purpose
Write comprehensive tests for `scripts/05_train_baseline_models.py` following TDD principles.

## Script Overview
The script trains three baseline models for corner kick outcome prediction:
1. Random Forest (with `class_weight='balanced'`)
2. XGBoost (with sample weights)
3. MLP (with StandardScaler)

## Key Functions to Test

### Data Loading
- `load_data()`: Loads features CSV and split indices
- `prepare_features(df)`: Extracts features and encodes labels

### Model Training
- `train_random_forest()`: Trains RF with balanced weights
- `train_xgboost()`: Trains XGB with sample weights
- `train_mlp()`: Trains MLP with StandardScaler

### Helper Functions
- `calculate_class_weights()`: Computes balanced class weights
- `evaluate_on_test()`: Evaluates all models on test set
- `save_models()`: Persists models and results
- `print_comparison_table()`: Displays results

## Testing Strategy

### Unit Tests
1. Test data loading functions with mock data
2. Test feature preparation and label encoding
3. Test class weight calculation
4. Test model training functions return correct types
5. Test evaluation metrics computation
6. Test model saving/loading

### Integration Tests
1. Test full pipeline with small synthetic dataset
2. Verify all models are trained successfully
3. Verify results JSON has correct structure
4. Verify saved models can be loaded and used for prediction

## Edge Cases to Cover
- Empty datasets
- Missing columns in features
- Invalid split indices
- File I/O errors

## Current Implementation Status
- Script exists and runs successfully
- Produces expected outputs (models, metrics)
- Need to add comprehensive test coverage

## Notes
- Tests should use small synthetic datasets to run quickly
- Use pytest fixtures for reusable test data
- Mock file I/O where appropriate to avoid dependency on actual data files
