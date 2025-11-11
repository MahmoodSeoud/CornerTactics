# StatsBomb Raw Data Analysis Feature

## Objective
Create a comprehensive script to analyze StatsBomb raw data to:
1. Calculate transition probabilities P(Event at t+1 | Event at t) for all events, especially corners
2. Build a complete transition matrix
3. Document all available features in the raw data

## Requirements
- **UPDATED**: Use StatsBomb SDK (statsbombpy) for data access
- Calculate transition probabilities for all event types
- Special focus on corner kicks
- Document all features available in raw data
- Output clear reports and visualizations

## Implementation Plan
1. ✅ Create test suite for data fetching and analysis
2. ✅ Implement StatsBomb data fetcher using SDK
3. ✅ Build transition matrix calculator
4. ✅ Create feature extractor and documenter
5. ✅ Generate comprehensive reports
6. ⏳ Update all tests for SDK format

## Key Classes/Modules
- `StatsBombRawAnalyzer`: Main analyzer class (uses statsbombpy SDK)
- `TransitionMatrixBuilder`: Builds probability matrices
- `FeatureExtractor`: Documents all available features
- `ReportGenerator`: Creates analysis reports

## Data Sources
- StatsBomb SDK (`statsbombpy`): Provides structured access to StatsBomb open data
- SDK flattens nested JSON structures (e.g., `type.name` → `type`, `pass.length` → `pass_length`)

## Key Changes Made
- Replaced direct GitHub API calls with statsbombpy SDK
- Updated all methods to handle SDK's flattened data structure
- Modified tests to match SDK format (no nested dicts)