# Separate Download and Data Access Functionality

## Problem
Currently `SoccerNetDataLoader` class has mixed responsibilities:
1. **Download** functions (`download_broadcast_videos`, `download_tracklets`) 
2. **Data access** functions (`load_annotations`, `list_games`, `get_corner_events`)

This violates Single Responsibility Principle.

## Current Usage Analysis
- `main.py` only uses data access methods: `list_games()` and `get_corner_events()`
- Shell scripts use `download_soccernet.py` which is a CLI wrapper around download methods
- Download methods in `SoccerNetDataLoader` are never called by main pipeline

## Desired Design
- `SoccerNetDataLoader` → Only data access (loading, listing, extracting)
- `download_soccernet.py` → Handle downloads directly, not just as CLI wrapper

## Implementation Plan
1. Create tests for data access functionality first (TDD Red phase)
2. Refactor `SoccerNetDataLoader` to remove download methods (Green phase)
3. Update `download_soccernet.py` to handle downloads directly (Green phase)
4. Ensure `main.py` still works (integration test)
5. Clean up and refactor for better design (Refactor phase)

## Files to Modify
- `src/data_loader.py` - Remove download methods, keep data access
- `src/download_soccernet.py` - Add download functionality directly

## Tests Needed
- Test `SoccerNetDataLoader` data access methods work independently
- Test download functionality still works via CLI
- Integration test that main.py pipeline works end-to-end