#!/usr/bin/env python3
"""
Comprehensive data quality audit across all corners in the SkillCorner dataset.
"""

import json
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

BASE = "/home/mseo/CornerTactics/data/skillcorner/data/matches"
MATCH_IDS = [1886347, 1899585, 1925299, 1953632, 1996435,
             2006229, 2011166, 2013725, 2015213, 2017461]
DEDUP_WINDOW = 20
WINDOW_FRAMES = 50
EVENT_CHAIN_WINDOW = 100


def load_tracking_index(mid):
    path = os.path.join(BASE, str(mid), f"{mid}_tracking_extrapolated.jsonl")
    index = {}
    with open(path) as f:
        for line in f:
            frame = json.loads(line)
            index[frame["frame"]] = frame
    return index


def load_events(mid):
    path = os.path.join(BASE, str(mid), f"{mid}_dynamic_events.csv")
    return pd.read_csv(path, low_memory=False)


def find_corners(events_df):
    mask = events_df["game_interruption_before"].str.contains(
        "corner", case=False, na=False
    )
    return events_df[mask].copy()


def deduplicate_corners(corner_rows):
    if corner_rows.empty:
        return corner_rows
    corner_rows = corner_rows.sort_values(["period", "frame_start"]).reset_index(drop=True)
    keep = []
    last_frame = {}
    for _, row in corner_rows.iterrows():
        p = row["period"]
        f = row["frame_start"]
        if p not in last_frame or (f - last_frame[p]) > DEDUP_WINDOW:
            keep.append(True)
            last_frame[p] = f
        else:
            keep.append(False)
    return corner_rows[keep].reset_index(drop=True)


def frame_stats(tracking_index, frame_num):
    frame = tracking_index.get(frame_num)
    if frame is None:
        return None
    players = frame.get("player_data", [])
    n_players = len(players)
    n_detected = sum(1 for p in players if p.get("is_detected"))
    n_extrapolated = n_players - n_detected

    ball = frame.get("ball_data", {})
    ball_detected = ball.get("is_detected")
    ball_x = ball.get("x")
    ball_y = ball.get("y")
    ball_z = ball.get("z")

    icp = frame.get("image_corners_projection", {})
    xs = [icp.get(k) for k in ["x_top_left", "x_bottom_left", "x_bottom_right", "x_top_right"] if icp.get(k) is not None]
    ys = [icp.get(k) for k in ["y_top_left", "y_bottom_left", "y_bottom_right", "y_top_right"] if icp.get(k) is not None]

    cam_x_min = min(xs) if xs else None
    cam_x_max = max(xs) if xs else None
    cam_y_min = min(ys) if ys else None
    cam_y_max = max(ys) if ys else None

    return {
        "n_players": n_players,
        "n_detected": n_detected,
        "n_extrapolated": n_extrapolated,
        "detection_rate": n_detected / n_players if n_players > 0 else 0.0,
        "ball_detected": ball_detected,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_z": ball_z,
        "cam_x_min": cam_x_min,
        "cam_x_max": cam_x_max,
        "cam_y_min": cam_y_min,
        "cam_y_max": cam_y_max,
    }


def window_stats(tracking_index, center_frame, half_window=WINDOW_FRAMES):
    detection_rates = []
    player_counts = []
    for f in range(center_frame - half_window, center_frame + half_window + 1):
        frame = tracking_index.get(f)
        if frame is None:
            continue
        players = frame.get("player_data", [])
        n = len(players)
        player_counts.append(n)
        if n > 0:
            det = sum(1 for p in players if p.get("is_detected"))
            detection_rates.append(det / n)
    return {
        "win_avg_det_rate": np.mean(detection_rates) if detection_rates else None,
        "win_min_players": min(player_counts) if player_counts else None,
        "win_max_players": max(player_counts) if player_counts else None,
        "win_frames_found": len(player_counts),
    }


def event_chain(events_df, corner_frame, period):
    mask = (
        (events_df["frame_start"] >= corner_frame)
        & (events_df["frame_start"] <= corner_frame + EVENT_CHAIN_WINDOW)
        & (events_df["period"] == period)
    )
    window = events_df[mask].copy()
    event_types = window["event_type"].dropna().unique().tolist()
    event_subtypes = window["event_subtype"].dropna().unique().tolist()
    pass_outcomes = window["pass_outcome"].dropna().unique().tolist()
    targeted_names = window["player_targeted_name"].dropna().unique().tolist()
    lead_to_shot = bool(window["lead_to_shot"].any()) if "lead_to_shot" in window.columns else None
    lead_to_goal = bool(window["lead_to_goal"].any()) if "lead_to_goal" in window.columns else None
    n_events = len(window)
    return {
        "chain_n_events": n_events,
        "chain_event_types": event_types,
        "chain_event_subtypes": event_subtypes,
        "chain_pass_outcomes": pass_outcomes,
        "chain_targeted_names": targeted_names,
        "chain_lead_to_shot": lead_to_shot,
        "chain_lead_to_goal": lead_to_goal,
    }


def main():
    all_rows = []
    total_corners_before_dedup = 0
    total_corners_after_dedup = 0

    for mid in MATCH_IDS:
        print(f"\n{'='*60}")
        print(f"Processing match {mid}...")
        events_df = load_events(mid)
        print(f"  Loading tracking data...")
        tracking = load_tracking_index(mid)
        total_frames = len(tracking)
        print(f"  Loaded {total_frames} tracking frames, {len(events_df)} event rows")

        corner_rows = find_corners(events_df)
        n_before = len(corner_rows)
        total_corners_before_dedup += n_before

        deduped = deduplicate_corners(corner_rows)
        n_after = len(deduped)
        total_corners_after_dedup += n_after
        print(f"  Corners: {n_before} rows -> {n_after} unique corners (deduped)")

        for idx, (_, crow) in enumerate(deduped.iterrows()):
            corner_frame = int(crow["frame_start"])
            period = int(crow["period"])
            interruption = crow["game_interruption_before"]
            corner_side = "for" if "corner_for" in str(interruption) else "against"
            player = crow.get("player_name", "")

            fs = frame_stats(tracking, corner_frame)
            if fs is None:
                print(f"  WARNING: frame {corner_frame} not found in tracking!")
                fs = {k: None for k in [
                    "n_players", "n_detected", "n_extrapolated", "detection_rate",
                    "ball_detected", "ball_x", "ball_y", "ball_z",
                    "cam_x_min", "cam_x_max", "cam_y_min", "cam_y_max"
                ]}

            ws = window_stats(tracking, corner_frame, WINDOW_FRAMES)
            ec = event_chain(events_df, corner_frame, period)

            row = {
                "match_id": mid,
                "corner_idx": idx,
                "frame": corner_frame,
                "period": period,
                "corner_side": corner_side,
                "player": player,
                **fs,
                **ws,
                **ec,
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    print(f"\n{'='*80}")
    print(f"DATA QUALITY AUDIT - SkillCorner Corners")
    print(f"{'='*80}")
    print(f"\nTotal corners before dedup: {total_corners_before_dedup}")
    print(f"Total corners after dedup:  {total_corners_after_dedup}")
    print(f"Matches processed: {len(MATCH_IDS)}")

    print(f"\n{'_'*80}")
    print("PER-CORNER SUMMARY")
    print(f"{'_'*80}")

    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.max_rows", 200)

    core = ["match_id", "corner_idx", "frame", "period", "corner_side", "player",
            "n_players", "n_detected", "n_extrapolated",
            "ball_detected", "ball_x", "ball_y", "ball_z"]
    core = [c for c in core if c in df.columns]
    print("\n--- Delivery Frame: Player & Ball ---")
    print(df[core].to_string(index=False))

    cam_cols = ["match_id", "corner_idx", "frame",
                "cam_x_min", "cam_x_max", "cam_y_min", "cam_y_max"]
    cam_cols = [c for c in cam_cols if c in df.columns]
    print("\n--- Delivery Frame: Camera Coverage (image_corners_projection) ---")
    print(df[cam_cols].to_string(index=False))

    det_cols = ["match_id", "corner_idx", "frame", "detection_rate",
                "win_avg_det_rate", "win_min_players", "win_max_players", "win_frames_found"]
    det_cols = [c for c in det_cols if c in df.columns]
    print("\n--- Detection Rate (delivery frame & +/-5s window) ---")
    print(df[det_cols].to_string(index=False))

    chain_cols = ["match_id", "corner_idx", "frame",
                  "chain_n_events", "chain_pass_outcomes", "chain_targeted_names",
                  "chain_lead_to_shot", "chain_lead_to_goal",
                  "chain_event_types", "chain_event_subtypes"]
    chain_cols = [c for c in chain_cols if c in df.columns]
    print("\n--- Event Chain (corner_frame to +100 frames) ---")
    print(df[chain_cols].to_string(index=False))

    print(f"\n{'_'*80}")
    print("AGGREGATE STATISTICS")
    print(f"{'_'*80}")

    numeric_cols = ["n_players", "n_detected", "n_extrapolated", "detection_rate",
                    "ball_x", "ball_y", "ball_z",
                    "cam_x_min", "cam_x_max", "cam_y_min", "cam_y_max",
                    "win_avg_det_rate", "win_min_players", "win_max_players",
                    "chain_n_events"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    print("\n--- Numeric Column Statistics ---")
    print(df[numeric_cols].describe().round(3).to_string())

    ball_det_count = df["ball_detected"].sum()
    ball_det_total = df["ball_detected"].notna().sum()
    print(f"\n--- Ball Detection at Delivery Frame ---")
    print(f"  Detected: {ball_det_count}/{ball_det_total} ({100*ball_det_count/ball_det_total:.1f}%)")

    print(f"\n--- Corner Side Distribution ---")
    print(df["corner_side"].value_counts().to_string())

    print(f"\n--- Period Distribution ---")
    print(df["period"].value_counts().sort_index().to_string())

    print(f"\n--- Corners per Match ---")
    print(df.groupby("match_id").size().to_string())

    all_outcomes = []
    for outcomes in df["chain_pass_outcomes"]:
        if isinstance(outcomes, list):
            all_outcomes.extend(outcomes)
    print(f"\n--- Pass Outcomes in Event Chain (all corners) ---")
    outcome_counts = pd.Series(all_outcomes).value_counts()
    print(outcome_counts.to_string())

    shot_count = df["chain_lead_to_shot"].sum()
    goal_count = df["chain_lead_to_goal"].sum()
    print(f"\n--- Lead to Shot/Goal ---")
    print(f"  Lead to shot: {shot_count}/{len(df)} ({100*shot_count/len(df):.1f}%)")
    print(f"  Lead to goal: {goal_count}/{len(df)} ({100*goal_count/len(df):.1f}%)")

    all_etypes = []
    for etypes in df["chain_event_types"]:
        if isinstance(etypes, list):
            all_etypes.extend(etypes)
    print(f"\n--- Event Types in Chain Window (all corners) ---")
    etype_counts = pd.Series(all_etypes).value_counts()
    print(etype_counts.to_string())

    all_stypes = []
    for stypes in df["chain_event_subtypes"]:
        if isinstance(stypes, list):
            all_stypes.extend(stypes)
    print(f"\n--- Event Subtypes in Chain Window (all corners) ---")
    stype_counts = pd.Series(all_stypes).value_counts()
    print(stype_counts.to_string())

    print(f"\n{'_'*80}")
    print("QUALITY FLAGS")
    print(f"{'_'*80}")

    low_players = df[df["n_players"] < 15]
    print(f"\n  Corners with <15 players at delivery: {len(low_players)}/{len(df)}")
    if len(low_players) > 0:
        print(low_players[["match_id", "corner_idx", "frame", "n_players"]].to_string(index=False))

    low_det = df[df["detection_rate"] < 0.5]
    print(f"\n  Corners with <50% detection rate at delivery: {len(low_det)}/{len(df)}")
    if len(low_det) > 0:
        print(low_det[["match_id", "corner_idx", "frame", "detection_rate", "n_players", "n_detected"]].to_string(index=False))

    no_ball = df[df["ball_detected"] == False]
    print(f"\n  Corners with ball NOT detected at delivery: {len(no_ball)}/{len(df)}")
    if len(no_ball) > 0:
        print(no_ball[["match_id", "corner_idx", "frame", "ball_x", "ball_y"]].to_string(index=False))

    no_events = df[df["chain_n_events"] == 0]
    print(f"\n  Corners with 0 events in chain window: {len(no_events)}/{len(df)}")

    win_low_det = df[df["win_avg_det_rate"] < 0.5]
    print(f"\n  Corners with <50% avg detection in +/-5s window: {len(win_low_det)}/{len(df)}")

    print(f"\n{'='*80}")
    print("AUDIT COMPLETE")
    print(f"{'='*80}")

    out_path = "/home/mseo/CornerTactics/data/skillcorner/corner_audit_results.csv"
    for col in ["chain_event_types", "chain_event_subtypes", "chain_pass_outcomes", "chain_targeted_names"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: "|".join(x) if isinstance(x, list) else str(x))
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
