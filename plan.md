# Corner Kick Training Data Pipeline: Execution Plan

## Objective
Build a training dataset for a corner kick outcome prediction model. Each row represents one player present during a corner kick event. The final dataset contains: player x,y position, player height, team affiliation, and who received the ball.

## Output Schema

| Column | Type | Source |
|--------|------|--------|
| corner_event_id | string | StatsBomb event ID |
| match_id | int | StatsBomb match ID |
| competition | string | StatsBomb competition name |
| delivery_player_id | int | StatsBomb pass event |
| delivery_player_name | string | StatsBomb pass event |
| delivery_x | float | StatsBomb pass end_location |
| delivery_y | float | StatsBomb pass end_location |
| delivery_type | string | Inswinger/Outswinger/Straight/Short |
| freeze_player_id | int | StatsBomb 360 freeze frame |
| freeze_player_name | string | StatsBomb 360 freeze frame |
| freeze_x | float | 360 freeze frame location[0] |
| freeze_y | float | 360 freeze frame location[1] |
| is_teammate | bool | 360 freeze frame teammate field |
| is_keeper | bool | 360 freeze frame keeper field |
| is_actor | bool | 360 freeze frame actor field |
| player_height_cm | int | Transfermarkt or FBref |
| outcome_player_id | int | Next event in chain (ball receipt, header, clearance, shot) |
| outcome_type | string | Header/Shot/Clearance/Goal Kick/Ball Receipt |
| outcome_body_part | string | Head/Right Foot/Left Foot |
| resulted_in_shot | bool | Derived from event chain |
| resulted_in_goal | bool | Derived from event chain |

---

## Phase 1: Identify Competitions with 360 Data

### Step 1.1: Load competitions
```
pip install statsbombpy
```

Use statsbombpy to load all competitions. Filter for competitions where match_available_360 is not null. The free open data includes 360 data for:
- Euro 2020 (competition_id=55, season_id=43)
- Euro 2024 (competition_id=55, season_id=282)
- 2022 FIFA World Cup (competition_id=43, season_id=106)
- Select La Liga, Premier League, Bundesliga seasons (check each)

### Step 1.2: Collect all match IDs with 360 data
For each competition+season pair with 360 availability, load matches. Store match_id, competition_name, season_name, home_team, away_team.

Expected output: a list of 150-300+ match IDs.

---

## Phase 2: Extract Corner Kick Events

### Step 2.1: Load events for each match
For each match_id, call sb.events(match_id=match_id, split=True, flatten_attrs=True).

### Step 2.2: Filter for corner kick deliveries
Corner kicks are pass events where:
- play_pattern_name == "From Corner"
- type_name == "Pass"
- pass_type_name == "Corner"

Store: event_id, match_id, player_id, player_name, team_name, location (x,y of delivery start), pass_end_location (x,y of delivery target), pass_outcome_name, pass_technique_name (Inswinging/Outswinging/Straight), pass_body_part_name.

### Step 2.3: Build the event chain for each corner
For each corner kick pass event, find the subsequent events in the same possession:
1. Look at events with the same possession number and a higher index.
2. The first ball-touching event after the delivery tells you who received the ball. This is typically: Ball Receipt, Clearance, Miscontrol, Shot, or an aerial duel (Duel where duel_type_name == "Aerial Lost" or duel_outcome_name includes aerial).
3. Record: outcome_player_id, outcome_player_name, outcome_type_name, outcome_body_part_name.
4. Scan the full possession for any Shot event. If found, set resulted_in_shot = True. If shot_outcome_name == "Goal", set resulted_in_goal = True.

Expected output: 1,500-2,000 corner kick events with outcome chains.

---

## Phase 3: Extract 360 Freeze Frames

### Step 3.1: Load 360 data for each match
For each match_id, call sb.frames(match_id=match_id, fmt='dataframe').

### Step 3.2: Join freeze frames to corner events
Filter the frames dataframe where the event_id matches a corner kick pass event from Phase 2. Each matching frame contains multiple rows (one per visible player). Fields per row:
- id (event_id)
- teammate (bool)
- actor (bool)
- keeper (bool)
- location (list of [x, y])

### Step 3.3: Resolve player identity in freeze frames
The open data 360 freeze frames do NOT always include player_id for every player in the frame. The teammate/actor/keeper flags are present.

Workaround for identity:
- The actor (the player performing the event, the corner taker) is identified by the parent event's player_id.
- For shot freeze frames, all players have IDs. For pass freeze frames (including corner deliveries), player_id is included in the paid API but may be missing in open data.

CHECK: Load a sample frame and inspect whether player_id and player_name fields are populated. If they are null for non-actor players, you need an alternative approach.

Alternative if player IDs are missing from freeze frames:
- Use the freeze frame positions (x, y, teammate flag) without individual identity.
- Join height data at the team level instead: compute average team height, or assign positional height proxies (center backs get team's tallest player height, etc.).
- OR: use only shot freeze frames (which do include player IDs) and filter for shots that originated from corners.

---

## Phase 4: Collect Player Height Data

### Step 4.1: Build a unique player list
From Phase 2 event data, collect all unique player_id + player_name combinations. This includes: corner takers, ball recipients, and (if available from freeze frames) all players visible during corners.

### Step 4.2: Scrape Transfermarkt for height

Option A (recommended): Use the transfermarkt Python package.
```
pip install transfermarkt
```

For each player name, search Transfermarkt. Extract height_cm from the player profile.

Option B: Scrape FBref player pages. FBref provides height on player bio pages.

Option C: Use a pre-built dataset. Kaggle has several FIFA/football player datasets with height. Search for "football player height dataset kaggle" and download.

### Step 4.3: Handle missing heights
Some players will not have height data available. For these:
1. Try alternate name spellings (accented characters, transliterations).
2. If still missing, impute by position: use the average height for that position from the rest of the dataset. Typical values: GK ~188cm, CB ~185cm, FB ~178cm, CM ~180cm, FW ~180cm.
3. Flag imputed rows with a boolean column: height_imputed = True.

### Step 4.4: Join height to the dataset
Merge player_height_cm into the freeze frame rows on player_id. If player IDs are missing from freeze frames, merge on team + is_keeper flag (keepers are identifiable; outfield players get team-level positional estimates).

---

## Phase 5: Assemble Final Dataset

### Step 5.1: Merge all data
For each corner kick event:
- One row per visible player in the freeze frame.
- Columns from the event (delivery info, outcome info).
- Columns from the freeze frame (x, y, teammate, keeper, actor).
- Columns from height join (player_height_cm, height_imputed).

### Step 5.2: Add derived features
For each player row, compute:
- distance_to_goal: Euclidean distance from (freeze_x, freeze_y) to center of goal (120, 40).
- distance_to_delivery_target: Euclidean distance from player position to pass_end_location.
- is_near_post: freeze_y < 36 (near post side, assuming standard orientation).
- is_far_post: freeze_y > 44.
- is_in_box: freeze_x > 102 and 18 < freeze_y < 62.
- is_in_6yard: freeze_x > 114 and 30 < freeze_y < 50.

### Step 5.3: Validate
- Check: every corner event has at least 5 players in the freeze frame (discard if fewer, likely bad data).
- Check: each corner has exactly one actor == True.
- Check: delivery coordinates are near a corner flag (x near 120, y near 0 or 80).
- Check: no duplicate player rows per corner event.
- Print summary statistics: total corners, corners per competition, average players per frame, height coverage percentage, shot rate, goal rate.

### Step 5.4: Export
Save as CSV and Parquet. Include a metadata JSON file documenting column definitions, data sources, and imputation rates.

---

## Phase 6: Validation Checks Before Training

Run these sanity checks on the final dataset:

1. Corner kick count by competition. Expect roughly 10-12 per match.
2. Shot conversion rate from corners. Professional average is 3-4%. If your dataset shows >10% or <1%, something is wrong with the event chain logic.
3. Goal conversion rate from corners. Professional average is roughly 3% of corners result in goals. 
4. Height distribution. Should be roughly normal, centered around 180cm with std of 6-7cm.
5. Position distribution. Most freeze frame players should be in or near the penalty box (x > 90).
6. Teammate/opponent split. Expect roughly 50/50 per corner (both teams commit players to the box).

---

## Estimated Scale

| Metric | Estimate |
|--------|----------|
| Matches with 360 data | 150-300 |
| Corner kicks extracted | 1,500-2,500 |
| Player-corner rows (final dataset) | 20,000-35,000 |
| Unique players needing height lookup | 500-1,000 |
| Height coverage (before imputation) | 85-95% |

---

## Dependencies

```
pip install statsbombpy pandas numpy
pip install transfermarkt  # or use web scraping
```

No GPU required. No video processing. Pure structured data extraction and joining.

---

## Known Risks

1. Open data 360 freeze frames may not include player_id for all visible players. This is the biggest risk. Check Phase 3.3 immediately. If IDs are missing, the height join becomes approximate (team-level, not player-level).

2. Transfermarkt scraping may be rate-limited or blocked. Use delays between requests. Have FBref or Kaggle datasets as fallback.

3. StatsBomb coordinate system uses 120x80 with origin at bottom-left. Make sure the corner flag positions match expectations (120,0), (120,80), (0,0), (0,80) before computing derived features.

4. The pass_technique_name field (Inswinger/Outswinger) may be null for some corners. These are still usable but that feature will need null handling.

5. Short corners exist in the data. These are corners where the ball is played short to a nearby teammate instead of delivered into the box. Filter or flag these separately. Identify them by: pass_end_location x < 105 or distance from corner flag to end location < 15 yards.
