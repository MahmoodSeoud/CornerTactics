# StatsBomb Raw Data Analysis Feature

## Objective
Create a comprehensive script to analyze StatsBomb raw data to:
1. Calculate transition probabilities P(Event at t+1 | Event at t) for all events, especially corners
2. Build a complete transition matrix
3. Document all available features in the raw data

## Requirements
- Analyze raw StatsBomb data directly from GitHub
- No preprocessing or modifications
- Calculate transition probabilities for all event types
- Special focus on corner kicks
- Document all features available in raw data
- Output clear reports and visualizations

## Implementation Plan
1. Create test suite for data fetching and analysis
2. Implement StatsBomb data fetcher
3. Build transition matrix calculator
4. Create feature extractor and documenter
5. Generate comprehensive reports

## Key Classes/Modules
- `StatsBombRawAnalyzer`: Main analyzer class
- `TransitionMatrixBuilder`: Builds probability matrices
- `FeatureExtractor`: Documents all available features
- `ReportGenerator`: Creates analysis reports

## Data Sources
- StatsBomb Open Data: https://github.com/statsbomb/open-data
- Raw JSON events, no modifications