# Supervisor Meeting Prep — Everything You Need to Know

**Mahmood Mohammed Seoud | ITU Copenhagen | Defense: March 16, 2026**

---

## The Thesis in One Sentence

Aggregate features from static freeze-frames contain no signal for corner kick prediction (AUC ~ 0.50, all p > 0.6), but adding per-player velocity vectors from continuous tracking data unlocks statistically significant prediction (AUC = 0.730, p = 0.010) — even with only 143 corners.

---

## The Full Story

**Photos don't work.** In the 7.5 ECTS project, we had 1,933 corners from StatsBomb with freeze-frame snapshots — one photo of where everyone stands when the corner is taken. We tried Random Forest, XGBoost, and MLP on aggregate features (zone counts, density measures, distances). AUC ~ 0.50-0.52 across every model and feature configuration. GNN experiments on the same data returned p-values from 0.61 to 0.97 — no model beat the random permutations. A snapshot of player positions tells you nothing about what happens next.

**The pre-trained football brain works.** We took a GNN built by US Soccer Federation researchers. They trained it on 20,863 open-play football situations (counterattacks, build-ups — not corners) to predict whether a shot would happen within 5 seconds. We froze that trained model and tested it on 1,796 DFL Bundesliga open-play shots with just a linear probe on top: 0.86 AUC. The brain genuinely understands football and transfers across datasets and leagues.

**But 57 corners isn't enough.** We applied that brain to 57 DFL corners with a linear probe. AUC = 0.57 +/- 0.24. Directional but not statistically significant. Too few samples to tell if it's real or noise. This motivated building a better pipeline and collecting more data.

**More corners, add velocity.** SkillCorner Open Data gave us 86 more corners from 10 A-League matches with 10Hz broadcast tracking. Combined with DFL's 57 corners = 143 total. Now we have velocity features — not just where players stand, but how fast they're moving and in which direction. We built a TacticAI-style two-stage pipeline: first predict who receives the corner, then predict whether a shot results.

**It works.** The cleanest proof is the velocity ablation on SkillCorner-only (same pipeline, same data, same evaluation — only the feature mask changes): position-only gives 0.583, adding velocity gives 0.747, a +0.164 jump. Combined 143 corners: shot AUC = 0.730, p = 0.010 (only 1 out of 100 shuffled models matched). Same permutation test methodology that returned "no signal" on the static data now returns "real signal" on velocity data.

**What doesn't work yet.** Receiver prediction on 86 SkillCorner corners alone: p = 0.406 (not significant). On 143 combined: Top-3 accuracy = 0.458, p = 0.050 (borderline). Needs more labeled data.

---

## Key Concepts Explained

**AUC** measures how good the model is at telling shots from non-shots. 0.50 = coin flip, useless. 0.75 = picks the right one 75% of the time when shown one shot-corner and one non-shot-corner. 1.00 = perfect. We use AUC instead of accuracy because 67% of corners don't lead to shots — a model that always says "no shot" gets 67% accuracy without learning anything.

**Permutation test** is how we verify the model actually learned something. Scramble all the labels randomly so "shot" and "no shot" are assigned to random corners. Train the model on this garbage. Repeat 100 times. If the real model isn't better than the garbage models, there's no signal. Crucially, if the model STILL scores well on shuffled labels, that's BAD — it means it can fit noise. What you WANT is the model to FAIL on shuffled labels. The gap between real and shuffled performance is your evidence. This works the same way for multi-class problems — just shuffle all labels.

**p-value** is the chance your result is a fluke. Below 0.05 = significant, below 0.01 = very significant. Our static data: all p > 0.6 (no signal). Our velocity data: p = 0.010 (1% fluke chance, real signal). Receiver prediction combined: p = 0.050 (borderline).

**GNN (Graph Neural Network)** treats players as nodes in a graph, with edges connecting nearby players. Each player has features (position, velocity, team, role). The GNN passes messages between connected players — each player "looks at" its neighbors and updates its understanding. After several rounds, each player's embedding encodes information about its local context (who's nearby, who's moving where). Then we make a prediction from the whole graph.

**Transfer learning** means taking knowledge from a big task (20,863 open-play situations) to help a small task (143 corners). The model doesn't start from zero.

**LOMO (Leave-One-Match-Out)** is how we split data for testing. We have 17 matches (combined). Each round: train on all except one, test on the held-out match. Rotate so every match gets a turn. This guarantees corners from the same match never appear in both training and testing — preventing the model from memorizing team-specific patterns. This IS a form of K-fold cross-validation where K = number of matches and folds are defined by match boundaries rather than random assignment. Not to be confused with cross-entropy, which is a loss function (how the model measures mistakes during training) and has nothing to do with data splitting.

**Velocity** is how fast and in which direction a player moves. Computed from tracking data: `vx = (x_now - x_previous_frame) / 0.1 seconds`. Two corners can look identical in a photograph but in one the forward is sprinting at 7 m/s toward the near post and in the other he's standing still. Completely different threat. This is THE key feature.

**Oracle vs predicted receiver:** Oracle means we tell the model who actually received the ball (cheating, shows upper bound). Predicted means Stage 1 guesses the receiver, Stage 2 uses that guess. Oracle AUC: 0.730. Predicted AUC: 0.715. Very close — Stage 1 isn't hurting much.

**Detected vs extrapolated:** SkillCorner tracks players from the TV broadcast camera, which doesn't show the whole pitch. Detected (70%) = actually seen on screen, accurate. Extrapolated (30%) = off-screen, position estimated, velocities dampened (0.61x real speed). We include `is_detected` as a feature so the model knows which velocities to trust.

---

## What Every Model Sees

### GNN: 13 features per player x 22 players

Each player node has 13 numbers: **position** (2) — x_norm and y_norm, pitch coordinates normalized to [-1, 1], all corners flipped so attacking team attacks toward +x. **Velocity** (3) — vx, vy (directional speed in m/s), and speed (total = sqrt(vx^2 + vy^2)), raw frame-to-frame difference, no smoothing. **Team and role** (7) — is_attacking, is_corner_taker, is_goalkeeper (binary flags), plus position group one-hot [GK, DEF, MID, FWD]. **Detection quality** (1) — is_detected, binary flag for tracking reliability.

A 14th feature (receiver_indicator) is appended during the forward pass — 0 for everyone in Stage 1, set to 1.0 for the receiver in Stage 2.

Edges connect each player to 6 nearest neighbors (KNN k=6, 132 edges). Each edge has 4 features: dx, dy, Euclidean distance, same_team.

### MLP baseline: same 13 features, no graph

Concatenates everything flat: 22 x 13 = 286 numbers -> dense layers -> shot probability. No edges, no message passing. Downside: player order is arbitrary, which hurts at larger scale.

### XGBoost baseline: 27 aggregate features

Doesn't see individual players. Sees summary statistics: spatial features per team (8 attacker + 8 defender), velocity aggregates per team (4 attacker + 4 defender), speed differential, corner side, detection rate.

### StatsBomb features (7.5 ECTS — the ones that failed)

Aggregate features from freeze-frames with NO velocity: counts of players in zones, density measures, distances, tactical ratios. All WHERE players stand, none HOW they move. AUC ~ 0.50-0.52.

### Feature comparison

| Approach | Velocity? | Individual players? | Relationships? | Shot AUC |
|----------|----------|-------------------|---------------|----------|
| StatsBomb 7.5 ECTS | No | No | No | ~0.50 |
| GNN position-only | No | Yes | Yes | 0.583 |
| XGBoost baseline | Yes (averaged) | No | No | 0.743 |
| GNN pretrained | Yes (per player) | Yes | Yes | 0.751 |
| MLP baseline | Yes (per player) | Yes (flat) | No | 0.802 |

No velocity -> fails. Add velocity -> works. Graph structure doesn't help yet. Per-player beats aggregates.

---

## The USSF Backbone

A CrystalConv GNN (CGConv) with 3 CGConv message-passing layers and 1 linear dimension-lifting layer. Architecture:

```
Input [N, 14] -> node_proj Linear(14, 12) [TRAINABLE]
             -> edge_proj Linear(4, 6) [TRAINABLE]
             -> conv1: CGConv(12, dim=6) + ReLU [FROZEN]
             -> lin_in: Linear(12, 128) [FROZEN]     <-- dimension lift, not message-passing
             -> convs[0]: CGConv(128, dim=6) + ReLU [FROZEN]
             -> convs[1]: CGConv(128, dim=6) + ReLU [FROZEN]
             -> Output [N, 128]
```

Output: 128-dim embedding per player. Trained on 20,863 open-play graphs to predict shot within 5 seconds, achieving AUC = 0.693 on its original task.

USSF used **12 features per player**: x, y, vx (unit direction), vy (unit direction), velocity_mag (normalized), velocity_angle (normalized), dist_goal, angle_goal, dist_ball, angle_ball, attacking_team_flag, potential_receiver. Plus **6 edge features**.

Our corners have 14 features (13 + receiver indicator) and 4 edge features — different from USSF's schema. The trainable adapter layers bridge this: `node_proj: Linear(14->12)` maps our features into USSF's input space, `edge_proj: Linear(4->6)` does the same for edges. These adapters are the ONLY trainable parts of the backbone path. Everything after (conv1, lin_in, both convs) is FROZEN.

Why freeze? With 143 corners, unfreezing would cause overfitting. Freezing forces the model to use existing football knowledge and only learn the translation + prediction heads.

Pretrained beats from-scratch on every metric: receiver Top-1 0.238 vs 0.091, shot AUC 0.751 vs 0.706 on SkillCorner-only. But it doesn't beat the MLP (0.802) because the MLP gets the same velocity features directly and the signal is simple enough to exploit without relational reasoning.

---

## Why the MLP Beats the GNN on SkillCorner-Only (and why that reverses on combined)

On SkillCorner-only (86 corners, 10 folds): MLP 0.802 > XGBoost 0.743 > GNN 0.751. Three reasons the MLP wins: fewer parameters generalize better with 86 samples; the velocity signal is simple ("anyone sprinting?") and doesn't need spatial reasoning; KNN edge construction may not be tactically meaningful with so few examples.

**But on the combined dataset (143 corners, 17 folds), the ordering flips:**

| Model | SK-only (86, 10 folds) | Combined (143, 17 folds) | Combined SK folds | Combined DFL folds |
|-------|:---:|:---:|:---:|:---:|
| **GNN pretrained** | 0.751 | **0.730** | **0.813** | **0.612** |
| XGBoost | 0.743 | 0.695 | 0.785 | 0.567 |
| MLP | 0.802 | 0.665 | 0.763 | 0.525 |

The GNN has the highest mean AUC on combined (0.730 vs 0.695 vs 0.665), though the standard deviations overlap (~0.20-0.25) and there is no formal paired significance test between models — the ordering is suggestive, not proven. What IS clear: all models degrade on DFL folds (harder data), but the GNN degrades least. The MLP drops the most — from best on SK-only to worst on combined.

**Key insight:** The GNN's frozen USSF backbone acts as a regularizer that forces a league-agnostic representation. The MLP, with no such constraint, is more sensitive to distribution shifts between SkillCorner and DFL data (different tracking systems, leagues, and sampling rates). This isn't purely "overfitting" — DFL folds are genuinely harder for all models — but the GNN handles the heterogeneity better.

---

## All Results (verified against result files)

### Combined dataset (143 corners, 17 LOMO folds)

| Metric | Value | p-value |
|--------|-------|---------|
| Shot AUC (oracle) | 0.730 +/- 0.202 | **0.010** |
| Shot AUC (predicted) | 0.715 +/- 0.193 | -- |
| Shot AUC (unconditional) | 0.730 +/- 0.204 | -- |
| Receiver Top-1 | 0.289 +/- 0.226 | -- |
| Receiver Top-3 | 0.458 +/- 0.341 | **0.050** |

### SkillCorner-only (86 corners, 10 LOMO folds)

| Metric | Value | p-value |
|--------|-------|---------|
| Shot AUC (oracle) | 0.751 +/- 0.213 | **0.020** |
| Receiver Top-3 | 0.308 +/- 0.279 | 0.406 |

### Velocity ablation (SkillCorner-only) — THE clean one-variable test

| Config | Features | Shot AUC |
|--------|----------|----------|
| Position only | x, y + team/role (9) | 0.583 +/- 0.280 |
| + Velocity | add vx, vy, speed (12) | **0.747 +/- 0.238** |
| + Detection | add is_detected (13) | 0.751 +/- 0.213 |
| Full KNN | all 13, KNN k=6 | 0.748 +/- 0.218 |
| Full FC | all 13, fully connected | 0.713 +/- 0.198 |

### Baselines — SK-only vs Combined (NO permutation tests)

| Model | SK-only (86, 10 folds) | Combined (143, 17 folds) |
|-------|:---:|:---:|
| MLP (286 flat) | 0.802 +/- 0.214 | 0.665 +/- 0.251 |
| XGBoost (27 aggregate) | 0.743 +/- 0.232 | 0.695 +/- 0.257 |
| GNN pretrained | 0.751 +/- 0.213 | **0.730 +/- 0.202** |
| Random | 0.500 | 0.500 |

### Source breakdown within combined pipeline (17 folds)

| Model | SK folds (10) | DFL folds (7) | All (17) |
|-------|:---:|:---:|:---:|
| GNN pretrained | 0.813 +/- 0.180 | 0.612 +/- 0.168 | 0.730 +/- 0.202 |
| XGBoost | 0.785 +/- 0.180 | 0.567 +/- 0.294 | 0.695 +/- 0.257 |
| MLP | 0.763 +/- 0.231 | 0.525 +/- 0.211 | 0.665 +/- 0.251 |

DFL folds are consistently harder across all models (different tracking system, league, sampling rate). The GNN degrades least, suggesting the pretrained backbone provides some cross-dataset robustness. However, no formal paired significance test was run between models — the ordering is suggestive.

### Permutation tests (100 permutations each)

| Dataset | Metric | Real | Null mean +/- std | Null range | p |
|---------|--------|------|-------------------|-----------|---|
| Combined (143) | Shot AUC | 0.715 | 0.508 +/- 0.059 | [0.341, 0.633] | **0.010** |
| Combined (143) | Receiver Top-3 | 0.458 | 0.300 +/- 0.086 | [0.058, 0.535] | **0.050** |
| SC-only (86) | Shot AUC | 0.751 | 0.502 +/- 0.087 | -- | **0.020** |
| SC-only (86) | Receiver Top-3 | 0.308 | 0.284 +/- 0.087 | -- | 0.406 |

### Per-fold breakdown (combined, shot AUC oracle)

| Fold | Match | Source | Corners | AUC |
|------|-------|--------|---------|-----|
| 1 | 1886347 | SK | 8 | 1.000 |
| 2 | 1899585 | SK | 7 | 0.600 |
| 3 | 1925299 | SK | 4 | 1.000 |
| 4 | 1953632 | SK | 8 | 0.875 |
| 5 | 1996435 | SK | 14 | 0.825 |
| 6 | 2006229 | SK | 4 | 1.000 |
| 7 | 2011166 | SK | 12 | 0.600 |
| 8 | 2013725 | SK | 13 | 0.533 |
| 9 | 2015213 | SK | 4 | 1.000 |
| 10 | 2017461 | SK | 12 | 0.700 |
| 11 | J03WMX | DFL | 10 | 0.571 |
| 12 | J03WN1 | DFL | 7 | 0.700 |
| 13 | J03WOH | DFL | 6 | 0.750 |
| 14 | J03WOY | DFL | 2 | 0.500 |
| 15 | J03WPY | DFL | 11 | 0.900 |
| 16 | J03WQQ | DFL | 15 | 0.364 |
| 17 | J03WR9 | DFL | 6 | 0.500 |

Four folds hit 1.0 (all small SK matches, 4-8 corners — easy to get perfect separation by chance). Fold 16 worst at 0.364. The permutation test accounts for this variance.

---

## Disentangling Pipeline vs Data Contributions

The jump from the transfer learning experiment (DFL-only, 0.57) to the combined two-stage GNN (0.730) confounds four changes: more data, different pipeline, different features, and a different data source. We cannot run a clean single-variable ablation because DFL corners lack receiver labels (Stage 1 can't train on DFL-only). However, the existing results allow partial disentanglement:

**1. The velocity ablation IS the clean proof (same pipeline, same data, one variable):**
- Position-only: 0.583 -> +velocity: 0.747 (+0.164). This is the headline finding.

**2. Adding DFL training data helps the GNN on SK test folds:**
- SK-only pipeline: 0.751 (trained on ~77 SK corners)
- Combined pipeline, same SK test folds: 0.813 (trained on ~135 corners including DFL)
- Delta: +0.062. The DFL corners provide generalizable signal.

**3. DFL corners are genuinely harder to predict:**
- Within the combined pipeline: SK folds = 0.813, DFL folds = 0.612
- The 0.730 combined result is a weighted average — mostly driven by SK performance.

**4. The GNN's pretrained backbone provides cross-dataset robustness:**
- All models degrade on DFL folds (harder data from different tracking system, league, and sampling rate).
- But the GNN degrades least: 0.751 to 0.730 overall (-0.021), vs MLP 0.802 to 0.665 (-0.137).
- GNN actually improves on the SK folds (+0.062) when DFL training data is added.
- On combined, GNN has the highest mean AUC (no formal paired test between models, but ordering is consistent across both SK and DFL fold subsets).

**5. The transfer learning 0.57 is not a reliable baseline:**
- Per-seed test AUCs: 0.70, 0.96, 0.54, 0.25, 0.40. Wild variance from a completely different pipeline (linear probe, USSF features, DFL-only training). Don't compare it to 0.730.

**Bottom line for the thesis:** Frame the velocity ablation as the primary evidence (one variable, clean test). Frame the combined p=0.010 as the statistical validation. Frame the cross-dataset analysis as evidence for the GNN's robustness advantage. Don't frame it as a progression from 0.57 to 0.730 — those are apples-to-oranges.

---

## Technical Nuances

**The permutation test "real" value (0.715) doesn't exactly match the main oracle AUC (0.730).** The permutation test re-runs the entire LOMO pipeline from scratch each time. Neural network training is stochastic, so numbers vary slightly between runs. 0.715 is closer to the predicted-receiver AUC (0.715) than the oracle (0.730). The significance finding is robust — the null distribution is centered at 0.508, far below either number.

**Two different velocity feature schemas exist in this project.** The `corner_prediction/` pipeline uses 3 velocity features per player: raw vx, vy (m/s), and speed (scalar magnitude). The `transfer_learning/` Phase 4 permutation importance test uses the USSF schema with 4 velocity features: vx (unit direction), vy (unit direction), velocity_mag (normalized), velocity_angle (normalized). These are different feature representations of the same underlying motion data. The headline ablation result (position-only 0.583 -> +velocity 0.747) comes from `corner_prediction/` and adds 3 features.

**DFL has zero receiver labels.** Shot prediction uses all 17 folds. Receiver prediction uses only 10 SkillCorner folds. Only 66/143 corners (46.2%) have receiver labels.

**Single seed (42).** Config defines 5 seeds but multi-seed not implemented. Permutation test compensates partially (reruns 100 times). p=0.010 is robust.

**No permutation tests on baselines.** MLP and XGBoost lack formal significance testing on both SK-only and combined.

---

## Corrections from Original Draft

This section documents changes from the original meeting prep document, with explanations.

### 1. Removed "p = 0.52" (SERIOUS)

**Original:** "Permutation test: p = 0.52"

**Problem:** No permutation test in the entire codebase returns p = 0.52. The StatsBomb GNN baseline experiments have p-values ranging from 0.608 to 0.970 (file: `experiments/gnn_baseline/results/results_20260105_053142.json`). The number 0.52 appears only as the classical ML MLP's AUC (from `docs/RESULTS.md`), which was likely confused with a p-value.

**Correction:** Changed to "p-values from 0.61 to 0.97" which reflects the actual GNN baseline permutation tests. The narrative is the same — no signal — but the number is now traceable to a result file.

### 2. Changed "22 aggregate features" to unspecified (MODERATE)

**Original:** "22 aggregate features (zone counts, density measures, distances)"

**Problem:** The codebase documents ~9-11 features for the 7.5 ECTS StatsBomb work (in `docs/METHODOLOGY.md`). The number "22" appears only in one planning document (`two-stage-pipeline.md`) and has no code or result file backing. The current XGBoost baseline uses 27 features (verified in code and result JSON), which includes velocity — unavailable in the 7.5 ECTS work.

**Correction:** Changed to "aggregate features" without a specific count. The exact number from the 7.5 ECTS project should be verified against that project's codebase.

### 3. Changed "AUC = 0.50" to "AUC ~ 0.50-0.52" (MINOR)

**Original:** "AUC = 0.50 across every model and feature configuration"

**Problem:** GNN models achieved 0.44-0.49 AUC. Classical ML models achieved 0.51-0.52 AUC. All effectively random, but not all exactly 0.50. Note: CLAUDE.md's "StatsBomb (events): AUC = 0.43" is also potentially inconsistent — the per-model breakdown in `docs/RESULTS.md` shows 0.51-0.52 for all models after removing data leakage, but the 0.43 figure has no backing result file and may be from a pre-fix evaluation.

**Correction:** Changed to "AUC ~ 0.50-0.52" which accurately covers the range of all models tested.

### 4. Fixed backbone architecture description (MINOR)

**Original:** "3 message-passing layers: conv1 (input->12), lin_in (12->128), two more CGConv layers (128->128)"

**Problem:** `lin_in` is `nn.Linear`, not a CGConv message-passing layer. There are 3 CGConv layers (conv1, convs[0], convs[1]) plus 1 linear layer (lin_in). The data flow description was correct but the layer count/types were wrong.

**Correction:** Changed to "3 CGConv message-passing layers and 1 linear dimension-lifting layer" with the architecture shown as a clear forward-pass diagram.

### 5. Fixed USSF feature names (MINOR)

**Original:** "x, y, vx, vy, speed, dist_goal, angle_goal, dist_ball, angle_ball, is_attacking, is_gk, is_ball_carrier"

**Problem:** The actual USSF schema (from `transfer_learning/phase2_engineer_dfl_features.py` and the pretrained weight config) uses: vx/vy as unit direction vectors (not raw m/s), velocity_mag (normalized [0,1], not raw speed), velocity_angle (missing from original list entirely), attacking_team_flag (not is_attacking), potential_receiver (not is_ball_carrier). There is no is_gk feature in the USSF schema.

**Correction:** Listed the correct 12 USSF features with their actual names and normalizations.

### 6. Removed "StatsBomb features: 22 aggregate features" from feature list

Replaced with the accurate description from `docs/METHODOLOGY.md`.

### 7. Added combined baseline results (NEW DATA)

**Change:** Ran MLP and XGBoost baselines on the combined 143-corner dataset (previously only existed for SK-only 86 corners). This was a known gap in the results.

**Finding:** The GNN becomes the best model on combined data. MLP drops from 0.802 to 0.665. The frozen USSF backbone provides cross-dataset robustness that flat models lack.

### 8. Added "Disentangling Pipeline vs Data Contributions" section (NEW)

**Change:** Added analysis addressing the confounding between the transfer learning 0.57 result and the combined 0.730 result. Uses per-fold source breakdown and cross-dataset baseline comparison to partially disentangle contributions.

### 9. Reframed "MLP beats GNN" section

**Original:** Presented as a limitation — MLP beats GNN on SK-only.

**Updated:** Now tells the full story — MLP wins on SK-only but loses on combined. The GNN's advantage is cross-dataset robustness from the pretrained backbone, not relational reasoning (yet).

---

## Questions Stella Might Ask

**"Main finding?"** -> Velocity is necessary. Static = zero signal. +0.164 AUC from velocity. p = 0.010.

**"Why trust small-sample p-value?"** -> Small n makes signal harder to detect, not easier to fake. Same test returned no signal (all p > 0.6) on 1,933 static corners. Works both directions.

**"MLP beats GNN?"** -> Only on SkillCorner-only (86 corners). On combined (143), GNN has highest mean: 0.730 vs 0.665 (no formal paired test, but GNN degrades least across datasets). The frozen backbone acts as a regularizer for cross-dataset generalization.

**"Compare to TacticAI?"** -> Different metrics, different scale. Contribution is identifying necessary conditions (velocity), not matching numbers.

**"Receiver prediction?"** -> Weak on SC-only (p=0.406). Improves combined (p=0.050). DFL has no labels so only 10 folds contribute.

**"What drove the improvement from 0.57 to 0.730?"** -> Those aren't comparable (different pipeline, different data, different features). The clean evidence is the velocity ablation: 0.583 -> 0.747, same everything except the feature mask. The per-fold breakdown shows DFL corners are harder (0.612 vs 0.813 SK) — the 0.730 is mostly driven by SK performance.

**"Known gaps?"** -> No baseline permutation tests. Single seed. DFL has no receiver labels. Transfer learning and two-stage pipeline changes were confounded (can't isolate contributions).

**"Experiment phase done?"** -> Core results complete. Baselines now on combined dataset too. Optional: multi-seed, baseline permutation tests. None change the story.

---

## Open Items for Stella

1. ~~Run baselines on combined dataset~~ DONE — GNN wins on combined, MLP drops to 0.665
2. Commit to writing only from here?
3. Thesis structure: chronological or thematic?
4. Run multi-seed before defense? (Probably not — p=0.010 is robust)
5. Should the "GNN wins on combined" finding be a headline result?

**If she asks something you don't know:** "Let me check the code and follow up." Don't guess.
