#!/usr/bin/env python3
"""
Velocity Derivation Feasibility Analysis for SkillCorner 10Hz Tracking Data.

Reads tracking data around corner kick frames, computes raw and smoothed
velocities, and reports noise statistics to determine the best approach
for deriving velocity features from position-only tracking data.
"""

import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_ROOT = Path("/home/mseo/CornerTactics/data/skillcorner/data/matches")
DT = 0.1  # 10 Hz

CORNERS = [
    (2011166, 29851, "goal-scoring corner"),
    (1996435, 16293, "corner kick"),
    (2017461, 43897, "corner kick"),
    (1886347, 38111, "corner kick"),
    (2013725, 13729, "corner kick"),
]

FRAME_WINDOW_BEFORE = 20
FRAME_WINDOW_AFTER = 10
