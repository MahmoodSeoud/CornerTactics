# Corner Kick Action Taxonomy Report

## Dataset Overview
- **Source**: StatsBomb Open Data (Raw GitHub)
- **Competition**: La Liga - 2019/2020
- **Match Analyzed**: Barcelona vs Eibar
- **Total Events**: 3690
- **Corner Kicks Found**: 10

## Raw Event Sequences After Corners

### Complete Event Type Distribution
Events occurring within 5 events after a corner kick:

| Event Type | Count | Percentage |
|------------|-------|------------|
| Ball Receipt* | 13 | 36.1% |
| Carry | 10 | 27.8% |
| Pass | 9 | 25.0% |
| Clearance | 3 | 8.3% |
| Goal Keeper | 1 | 2.8% |

## Outcome Patterns

### Primary Outcomes (First Significant Event)
Significant events are: Shot, Clearance, Goal Keeper, Interception

| Outcome | Count | Team |
|---------|-------|------|
| Clearance | 4 | opp team |
| Goal Keeper | 1 | opp team |
| Shot | 1 | same team |


## Detailed Sequence Examples

### Example Corner Sequences

#### Corner 1: 21:14 by Eibar
- **Location**: [120.0, 80.0]
- **Target**: [112.8, 43.8]
- **Next 5 Events**:
  1. Clearance (Barcelona)

#### Corner 2: 47:50 by Eibar
- **Location**: [120.0, 0.1]
- **Target**: [114.7, 39.1]
- **Next 5 Events**:
  1. Goal Keeper (Barcelona)
  2. Carry (Barcelona)
  3. Pass (Barcelona)
  4. Ball Receipt* (Barcelona)
  5. Carry (Barcelona)

#### Corner 3: 50:58 by Barcelona
- **Location**: [120.0, 0.1]
- **Target**: [109.3, 5.7]
- **Next 5 Events**:
  1. Ball Receipt* (Barcelona)
  2. Carry (Barcelona)
  3. Pass (Barcelona)
  4. Ball Receipt* (Barcelona)
  5. Pass (Barcelona)


## Proposed Classification Taxonomies

### Option 1: Binary Classification
- **Success**: Shot attempts (including goals)
- **Failure**: All other outcomes

### Option 2: 3-Class System (Recommended)
- **Class 0 - Shot**: Goal or shot attempt
- **Class 1 - Clearance**: Defensive clearance
- **Class 2 - Possession**: Continued play

### Option 3: 4-Class System
- **Class 0 - Goal**: Successful goal
- **Class 1 - Shot**: Shot attempt (no goal)
- **Class 2 - Clearance**: Defensive clearance
- **Class 3 - Possession**: Continued play

## Implementation Considerations

1. **Time Window**: Most outcomes occur within 5 events or 10 seconds
2. **Event Chains**: Track event sequences, not just single outcomes
3. **Team Context**: Consider whether outcome is by attacking or defending team
4. **Spatial Context**: Use location data to assess danger/quality

## Raw Data Advantages

Using raw StatsBomb data provides:
- Complete event sequences
- Precise timestamps
- Related events linkage
- Unprocessed spatial coordinates
- Full event metadata

This enables more sophisticated analysis than processed/aggregated data.
