# SkillCorner Corner Kick Deep Data Audit

**Date**: 2026-02-15
**Dataset**: SkillCorner Open Data — 10 A-League 2024/2025 matches
**Location**: `/home/mseo/CornerTactics/data/skillcorner/data/matches/`
**Audit CSV**: `/home/mseo/CornerTactics/data/skillcorner/corner_audit_results.csv`

---

## Task 1: Corner Count

**86 corners across 10 A-League 2024/2025 matches.**

| Match ID | Home vs Away | Corners |
|----------|-------------|---------|
| 1886347 | Auckland FC vs Newcastle | 8 |
| 1899585 | Auckland FC vs Wellington P FC | 7 |
| 1925299 | Perth Glory vs Brisbane FC | 4 |
| 1953632 | Melbourne City vs CC Mariners | 8 |
| 1996435 | Adelaide United vs Sydney FC | 14 |
| 2006229 | Melbourne City vs Macarthur FC | 4 |
| 2011166 | Melbourne V FC vs Wellington P FC | 12 |
| 2013725 | Sydney FC vs Western United | 13 |
| 2015213 | Western United vs ? | 4 |
| 2017461 | Melbourne V FC vs Auckland FC | 12 |
| **Total** | | **86** |

**Average**: 8.6 corners per match.

### How corners are identified

- **Source column**: `game_interruption_before` in `dynamic_events.csv`
- **Values**: `corner_for` (36 occurrences) and `corner_against` (50 occurrences)
- **Deduplication**: Events within 20 frames (2s at 10Hz) of each other in the same period are grouped as one corner
- **Raw rows**: 117 event rows → 86 unique corners after dedup
- No corner indicator exists in `event_type` or `event_subtype` columns
- `phases_of_play.csv` has `set_play` phase type (174 total), of which 82 (47%) contain a corner. The remaining 92 are free kicks (54) and throw-ins (7).

### Per-corner frame ranges

Full per-corner listing with match ID, period, and frame number is saved in the audit CSV.

---

## Task 2: What One Corner Kick Looks Like

**Match 2011166 (Melbourne Victory vs Wellington Phoenix), frame 29851 — GOAL from corner**

### From dynamic_events.csv

Single event row at the delivery:

| Field | Value |
|-------|-------|
| event_type | player_possession |
| player_name | Z. Machach |
| team_shortname | Melbourne V FC |
| game_interruption_before | corner_for |
| game_interruption_after | goal_for |
| lead_to_shot | True |
| lead_to_goal | True |
| x_start | 51.26 |
| y_start | 11.03 |
| pass_outcome | (empty) |
| player_targeted_name | (empty) |

This is a sparse corner — only one event row, no passing options, no off-ball runs logged. The ball went from corner to goal with no intermediate events captured.

### From tracking data (frame 29851)

```
timestamp: 00:45:55.10
period: 1
ball: x=57.02, y=34.36, z=0.2, is_detected=False
possession: group="away team"
camera: x=[27.5, 85.5], y=[-22.4, 39.0]
players: 22 total (12 detected, 10 extrapolated)
```

### Player positions at delivery (t=0)

| Player | # | Team | Position | x | y | Detected |
|--------|---|------|----------|---|---|----------|
| M. Langerak | 1 | AWAY (MEL) | GK | -16.2 | 0.4 | No |
| R. Miranda | 21 | AWAY | RCB | 45.1 | -4.1 | Yes |
| L. Jackson | 4 | AWAY | LCB | 44.9 | -2.4 | No |
| J. Rawlins | 22 | AWAY | RB | 27.7 | 0.9 | No |
| K. Bos | 28 | AWAY | LB | 33.3 | -4.9 | Yes |
| J. Valadon | 14 | AWAY | LDM | 30.6 | 5.9 | No |
| R. Teague | 6 | AWAY | RDM | 48.0 | 3.9 | No |
| Z. Machach | 8 | AWAY | AM | 51.3 | 11.0 | No |
| D. Arzani | 7 | AWAY | RW | 45.0 | 3.9 | No |
| N. Velupillay | 17 | AWAY | LW | 50.2 | 6.4 | Yes |
| N. Vergos | 9 | AWAY | CF | 47.9 | 8.2 | Yes |
| J. Oluwayemi | 1 | HOME (WEL) | GK | 51.3 | -0.2 | No |
| I. Hughes | 15 | HOME | RCB | 45.9 | -2.9 | Yes |
| S. Wootton | 4 | HOME | LCB | 49.6 | 2.9 | Yes |
| T. Payne | 6 | HOME | RB | 47.8 | 5.2 | Yes |
| S. Sutton | 19 | HOME | LB | 49.7 | 1.9 | No |
| A. Rufer | 14 | HOME | DM | 47.1 | -4.5 | Yes |
| M. Sheridan | 27 | HOME | LM | 46.8 | 3.8 | Yes |
| C. Piper | 3 | HOME | RM | 50.2 | -1.5 | Yes |
| H. Ishige | 9 | HOME | AM | 40.7 | 5.9 | Yes |
| K. Nagasawa | 25 | HOME | AM | 48.8 | 3.6 | No |
| K. Barbarouses | 7 | HOME | CF | 46.6 | 14.5 | Yes |

**Observations**:
- Corner taker (Z. Machach) at (51.3, 11.0) — near corner flag, but **extrapolated** (not detected)
- Attacking GK (M. Langerak) at x=-16.2 — staying in own half, way off-screen, extrapolated
- Most players clustered in x=[40, 52], y=[-5, 15] — inside/near the penalty box
- Ball extrapolated at (57.02, 34.36) — near corner flag area

### From match.json

Player metadata available per player:
- `id` (maps to `player_id` in tracking)
- `team_id` (maps to home_team.id or away_team.id)
- `player_role.name` (e.g., "Goalkeeper", "Right Center Back", "Left Winger")
- `player_role.position_group` (e.g., "Central Defender", "Midfield", "Wide Attacker")
- `number` (shirt number)
- `short_name`, `first_name`, `last_name`
- `birthday` (DOB available for 357/360 players across all matches)
- `trackable_object` (different from `id` — NOT used in tracking JSONL)
- **No height or weight fields**

---

## Task 3: Feature Completeness for GNN Construction

| Feature | TacticAI had it | SkillCorner has it? | How to extract |
|---------|----------------|---------------------|----------------|
| Player (x, y) at t=0 | Yes | **Yes** | `player_data[].x, .y` in tracking JSONL at delivery frame |
| Velocity (vx, vy) at t=0 | Yes (25Hz) | **Yes** (10Hz) | `(x[t]-x[t-1])/0.1` — raw is clean, no smoothing needed |
| Player identity | Yes | **Yes** | `player_data[].player_id` → match.json `players[].id` |
| Team assignment | Yes | **Yes** | `player_id` → match.json `team_id` → compare to corner team |
| Player role/position | Yes | **Yes** | match.json `player_role.name` (GK, CB, LW, AM, CF, etc.) |
| Ball position | Yes | **Yes (92%)** | `ball_data.x, .y, .z` — detected in 79/86 corners |
| Corner taker identity | Yes | **Partial (74%)** | `player_name` when `event_type=player_possession` at delivery |
| Receiver identity | Yes | **Partial (42-86%)** | `player_targeted_name` or passing_option `targeted=True` |
| Shot outcome label | Yes | **Yes (100%)** | `lead_to_shot` / `lead_to_goal` in event chain |
| Height/weight | Yes | **No** | Not in match.json — only `birthday` available |
| Detection flag | N/A (optical) | **Yes** | `is_detected` per player per frame |

### Notes on receiver identification

Three methods to identify the receiver, with different coverage:

1. **Delivery event `player_targeted_name`**: Available on 73% of corners (63/86) — the person the corner taker aimed for
2. **Passing_option `targeted=True` + `received=True`**: Available on 65% of corners (56/86) within 3 seconds — includes xpass_completion, xthreat, dangerous flags
3. **Next player_possession event**: Available on 42% (36/86) — who actually gained possession after the corner
4. **Combined (any method)**: 86% of corners (74/86) have at least one receiver indicator

---

## Task 4: Data Quality

### Detection rates at delivery frame (n=86)

| Metric | Value |
|--------|-------|
| Players per frame | Always 22/22 (extrapolation fills gaps) |
| Mean detection rate | **69.8%** (15.3 detected, 6.7 extrapolated) |
| Std detection rate | 17.2% |
| Min detection rate | 0.0% (3 corners) |
| Max detection rate | 90.9% (2 corners) |
| Ball detected | **79/86 (91.9%)** |

### Detection rate distribution

| Bucket | Count | % |
|--------|-------|---|
| 0-30% | 3 | 3.5% |
| 30-50% | 3 | 3.5% |
| 50-70% | 28 | 32.6% |
| 70-90% | 50 | 58.1% |
| 90-100% | 2 | 2.3% |

### Detection in ±5 second window

| Metric | Value |
|--------|-------|
| Mean window detection rate | 64.8% |
| Corners with <50% window detection | 16/86 (18.6%) |
| Corners with <30% window detection | 1/86 (1.2%) |
| Frames found per window | Always 101 (50 before + delivery + 50 after) |

### Camera coverage (image_corners_projection)

- Camera typically covers one half of the pitch (~60m x-range)
- Corner flag area is always visible when camera projection exists
- 3 corners have NaN camera projection (same as 0% detection — likely replay/cutaway frames)
- Players staying back (attacking GK, holding midfielders) are consistently off-screen and extrapolated

### Worst-quality corners (should be excluded)

| Match | Frame | Detection Rate | Ball Detected | Camera | Issue |
|-------|-------|---------------|---------------|--------|-------|
| 1899585 | 60306 | 0.0% | No | NaN | Likely replay/cutaway |
| 2006229 | 1126 | 0.0% | No | NaN | Very early in match |
| 2017461 | 57102 | 0.0% | No | NaN | Ball position mid-pitch (misaligned) |

### Usable corners after quality filtering

- Excluding 3 corners with 0% detection: **83 usable**
- Excluding corners with <50% delivery detection: **80 usable**
- Excluding corners with <50% window detection: **70 usable**

---

## Task 5: Corner Kick Event Chain Reconstruction

### Event coverage within 10 seconds post-corner (n=86)

| Metric | Value |
|--------|-------|
| Mean events per corner | 11.9 |
| Min events | 1 |
| Max events | 29 |
| Corners with 0 events | 0 (every corner has at least 1) |

### Event types in post-corner window

| Event Type | Total Count | Corners with this type |
|-----------|-------------|----------------------|
| passing_option | 484 | 71/86 (82.6%) |
| player_possession | 248 | 86/86 (100%) |
| on_ball_engagement | 163 | 68/86 (79.1%) |
| off_ball_run | 132 | 56/86 (65.1%) |

### Event subtypes (off-ball runs and engagements)

| Subtype | Count |
|---------|-------|
| pressure | 90 |
| other | 39 |
| run_ahead_of_the_ball | 37 |
| dropping_off | 35 |
| recovery_press | 21 |
| coming_short | 18 |
| support | 13 |
| pulling_wide | 9 |
| counter_press | 8 |
| behind | 6 |
| cross_receiver | 5 |
| pressing | 5 |

### Outcome labels

| Outcome | Count | % |
|---------|-------|---|
| lead_to_shot = True | 29/86 | 33.7% |
| lead_to_goal = True | 3/86 | 3.5% |
| No shot | 57/86 | 66.3% |

### Pass outcome at delivery event

| Metric | Count |
|--------|-------|
| Has pass_outcome | 49/86 (57%) |
| Successful | 17 |
| Unsuccessful | 32 |
| No pass_outcome | 37 |

### High pass indicator

- Only 13/86 corners have `high_pass` populated (True: 4, False: 9)
- No explicit inswinger/outswinger indicator exists
- Delivery type would need to be derived from ball trajectory in tracking data

### Sample event chains

**Goal corner (match 2011166, frame 29851):**
```
f=29851 player_possession by Z. Machach (Melbourne V FC) → SHOT → GOAL
```
(Single event — no intermediate events captured)

**Shot corner (match 1886347, frame 38111):**
```
f=38111 player_possession by F. Gallegos (Auckland FC) targeted=G. May pass=successful
f=38111 passing_option by L. Verstraete (Auckland FC)
f=38111 passing_option by J. Brimmer (Auckland FC)
f=38111 passing_option by L. Gillion (Auckland FC)
f=38115 off_ball_run/dropping_off by J. Brimmer
f=38122 passing_option by G. May (cross_receiver run)
f=38125 off_ball_run/run_ahead_of_the_ball by J. Brimmer
f=38139 player_possession by G. May → SHOT
```

**No-shot corner (match 1886347, frame 7939):**
```
f=7939  player_possession by J. Brimmer targeted=L. Verstraete pass=unsuccessful
f=7941  on_ball_engagement/other by M. Natta (Newcastle)
f=7956  player_possession by N. Pijnaker targeted=A. Paulsen pass=successful
f=8011  player_possession by E. Adams (Newcastle) pass=unsuccessful
f=8037  player_possession by L. Gillion targeted=F. Gallegos pass=successful
```

---

## Task 6: Velocity Derivation Feasibility

**Analysis**: 5 corners across 4 matches, 2,926 velocity samples (1,918 detected, 1,008 extrapolated).

### Raw velocity distributions

| Metric | All | Detected | Extrapolated |
|--------|-----|----------|-------------|
| n | 2,926 | 1,918 | 1,008 |
| Mean speed | 1.69 m/s | 1.95 m/s | 1.20 m/s |
| Std speed | 1.34 | 1.48 | 0.83 |
| Median | 1.34 m/s | 1.53 m/s | 1.08 m/s |
| P95 | 4.61 m/s | 5.19 m/s | 2.75 m/s |
| Max | 7.56 m/s | 7.56 m/s | 5.99 m/s |

### Teleportation events

**Zero** across all 2,926 samples (threshold: >50 m/s = 5m displacement in 0.1s).

All player speeds below 12 m/s (physical sprint maximum). No velocity clipping needed.

### Physical plausibility

| Threshold | Count | % | Assessment |
|-----------|-------|---|-----------|
| > 5 m/s | 115 | 3.93% | OK (fast runs) |
| > 8 m/s | 0 | 0.00% | OK |
| > 10 m/s | 0 | 0.00% | OK |
| > 12 m/s | 0 | 0.00% | OK |

### Smoothing effectiveness

| Method | Accel Mean | Accel Std | Noise Reduction |
|--------|-----------|-----------|-----------------|
| Raw | 1.51 m/s² | 1.43 | baseline |
| SMA-3 | 1.23 m/s² | 1.13 | 19.0% |
| SMA-5 | 1.13 m/s² | 1.02 | 25.3% |

### Detected vs extrapolated comparison

| Metric | Detected | Extrapolated | Ratio |
|--------|----------|-------------|-------|
| Speed mean | 1.95 m/s | 1.20 m/s | 0.61x |
| Speed std | 1.48 | 0.83 | — |
| Accel mean | 1.48 m/s² | 1.58 m/s² | 1.07x |

Extrapolated velocities are **attenuated** (0.61x speed ratio) because SkillCorner's extrapolation model uses smooth predictions. Noise levels are nearly identical (1.07x ratio).

### Ball velocity during delivery

| Match | Ball Detected | Peak Speed | Pattern |
|-------|--------------|-----------|---------|
| 1996435 | Yes | 21.3 m/s | Textbook delivery: 21→13→5 m/s deceleration |
| 2013725 | Yes | 9.3 m/s | Consistent 8-9 m/s — inswinging delivery |
| 1886347 | Yes | 4.2 m/s | Slow (2-4 m/s) — likely short corner |
| 2017461 | Yes | 13.9 m/s | Clear delivery arc visible |
| 2011166 | **No** | 10.2 m/s | All extrapolated — dampened, unreliable |

### Velocity recommendation

1. **Use raw frame-to-frame velocity** — no smoothing needed. `vx = (x[t] - x[t-1]) / 0.1`
2. **If smoothing desired**: SMA-3 is the best tradeoff (19% noise reduction, 0.2s latency cost)
3. **Include `is_detected` as a GNN node feature** — extrapolated velocities are systematically dampened
4. **Ball velocity**: Only trust when `ball_data.is_detected=True`. Extrapolated ball speeds are unreliable.
5. **For GNN features**: Raw `vx, vy` as node features + `speed = sqrt(vx²+vy²)` as derived feature

---

## Task 7: TacticAI Comparison Matrix

| Requirement | TacticAI Spec | SkillCorner Capability | Gap | Severity |
|------------|--------------|----------------------|-----|----------|
| Sample size | 7,176 corners | **86 corners** | 98.8% fewer | **CRITICAL** |
| Tracking frequency | 25 Hz | 10 Hz | 2.5x lower | Moderate |
| Position accuracy | Optical (sub-cm) | Broadcast CV | Noisier + 30% extrapolated | Moderate |
| All 22 tracked | Always (100% detected) | Always (100% via extrapolation) | ~30% are guesses | Significant |
| Detection rate | 100% | **70% at delivery** | 30% extrapolated | Significant |
| Velocity vectors | Derived from 25Hz | Derivable from 10Hz, clean | Lower temporal resolution | Low |
| Player identity | Full | **Full** | None | None |
| Team assignment | Full | **Full** (via match.json) | None | None |
| Player role | Full | **Full** (GK, CB, LW, etc.) | None | None |
| Ball position | Always | **92% at delivery** | 8% extrapolated | Low |
| Corner taker ID | From club data | **74%** from events | 26% missing | Moderate |
| Receiver ID | From tracking | **42-86%** from events | Partial coverage | Significant |
| Shot outcome | From club data | **100%** (lead_to_shot) | None | None |
| Height/weight | Yes | **No** | Missing entirely | Moderate |

---

## Feasibility Assessment

### Can you build receiver prediction?

**Technically yes, but severely underpowered.** You have 86 corners, of which ~74 have usable receiver labels. TacticAI had 7,176. With 86 samples and a GNN that needs to predict which of ~20 candidate receivers gets the ball, you have roughly 4 samples per class. Cross-validation will be wildly unstable.

### Can you build shot prediction?

**Marginally better** because it's binary: 29 shot vs 57 no-shot. But the transfer learning experiments already showed that corner-specific shot prediction hits ~0.55 AUC with 57 DFL samples. Adding 86 more gives ~143 total, which still won't produce statistically significant results.

### What this dataset IS good for

1. **Proof-of-concept pipeline** — Build the full extraction → graph construction → GNN pipeline on real broadcast tracking data
2. **Feature engineering validation** — Test whether SkillCorner's rich event features (passing_options, off_ball_runs, xThreat) add signal beyond position-only
3. **Multi-source integration** — Combine these 86 corners with 57 DFL corners for ~143 total
4. **Data quality benchmarking** — Quantify the accuracy gap between broadcast CV tracking and optical tracking
5. **Methodology demonstration** — Show for the thesis that the approach generalizes across data sources

### What you CANNOT do

- Train a reliable receiver prediction model (too few samples, partial labels)
- Achieve statistically significant results for shot prediction
- Match TacticAI's results in any meaningful comparison
- Use height/weight as features (not available)

### Bottom line

This dataset adds 86 corners to your pool. Combined with DFL (57) and potentially SoccerNet GSR, you're building toward a multi-source dataset. But you'd need hundreds of matches — not 10 — to approach statistical power for corner outcome prediction. The value is in demonstrating the pipeline works across data sources, not in the raw predictive performance.

---

## Data File Reference

### Per-match files

```
data/skillcorner/data/matches/{match_id}/
├── {id}_match.json                    # Lineups, player metadata, pitch dimensions
├── {id}_tracking_extrapolated.jsonl   # Frame-by-frame tracking at 10Hz
├── {id}_dynamic_events.csv            # Events: possessions, passes, runs, pressing
└── {id}_phases_of_play.csv            # Phase labels per frame range
```

### Match IDs

```
2017461, 2015213, 2013725, 2011166, 2006229,
1996435, 1953632, 1925299, 1899585, 1886347
```

### Key field mappings

- **Tracking `player_id`** → match.json `players[].id` (NOT `trackable_object`)
- **Team assignment**: `player_id` → match.json `players[].team_id` → compare to `home_team.id` / `away_team.id`
- **Corner detection**: `dynamic_events.csv` → `game_interruption_before` contains `corner_for` or `corner_against`
- **Shot outcome**: `lead_to_shot` / `lead_to_goal` columns in `dynamic_events.csv`
- **Receiver**: `player_targeted_name` in delivery event or passing_option events with `targeted=True`
- **Coordinate system**: Center-origin meters, x ∈ [-52.5, 52.5], y ∈ [-34, 34]

### Audit outputs

- Script: `/home/mseo/CornerTactics/scripts/audit_skillcorner_corners.py`
- Results CSV: `/home/mseo/CornerTactics/data/skillcorner/corner_audit_results.csv`
- Velocity analysis: `/home/mseo/CornerTactics/scripts/velocity_feasibility_analysis.py`
