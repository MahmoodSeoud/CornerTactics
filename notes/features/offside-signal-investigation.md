# OFFSIDE Signal Investigation

## Feature Overview

Investigate why OFFSIDE class achieved 38.36% AP in FAANTRA video model (well above random for 1.6% prevalence class). Extract and analyze spatial features that could predict offside outcomes.

## Data Availability

| Dataset | OFFSIDE Count | Has Positions | Has 8-Class Labels |
|---------|---------------|---------------|-------------------|
| FAANTRA (video) | 77 (1.6%) | No | Yes |
| StatsBomb (freeze-frame) | 1 (0.05%) | Yes | No (binary only) |

**Limitation**: Cannot directly analyze OFFSIDE freeze-frame positions due to minimal overlap.

## Adapted Approach

Since we lack OFFSIDE freeze-frame data, we'll:

1. **Compute offside-predictive features** for all corners:
   - Attackers beyond last defender (x-position)
   - Number of attackers in "offside zone"
   - Compactness of defensive line
   - Distance of furthest attacker from defensive line

2. **Hypothesize**: These features should theoretically predict offside
   - If attackers are near/beyond the defensive line, offside is more likely
   - The video model might learn this visual pattern

3. **Test transfer**: Do these features improve shot prediction?
   - If not, confirms they're specific to offside (procedural outcomes)

## Key Spatial Features to Extract

### Defensive Line Features
- `last_defender_x`: X-position of last defender (excluding goalkeeper)
- `defensive_line_y_spread`: Y-spread of defensive line
- `defensive_line_compactness`: How tight the defensive line is

### Attacker Position Features
- `attackers_beyond_last_defender`: Count of attackers past defensive line
- `furthest_attacker_x`: X-position of furthest forward attacker
- `attacker_to_defender_x_gap`: Distance from furthest attacker to last defender
- `attackers_in_offside_zone`: Attackers within 2m of goal line beyond defenders

### Hypothesis
- High `attackers_beyond_last_defender` → Higher offside probability
- Compact `defensive_line` + Spread attackers → Clear offsides more likely
- These features visible in video pre-kick

## Expected Results

1. **Offside-predictive features** correlate with certain spatial patterns
2. **Transfer to shot prediction** likely fails (offside is procedural, not related to shot probability)
3. **Visualization** shows distinct positioning patterns

## Progress Log

- [2026-01-05] Feature branch created
- [2026-01-05] Discovered data limitation: only 1 OFFSIDE in freeze-frame data
- [2026-01-05] Adapted approach to compute theoretical offside-predictive features
