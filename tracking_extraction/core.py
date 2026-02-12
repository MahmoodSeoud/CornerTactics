"""Core data structures, coordinate transforms, velocity computation, and I/O.

All tracking sources are normalized to a unified format:
- Pitch: 105m x 68m, origin at (0, 0) corner
- Velocities: m/s via central difference
- Teams: "attacking" / "defending" / "unknown"
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


# --- Data Structures ---

@dataclass
class PlayerFrame:
    player_id: str
    team: str           # "attacking" | "defending" | "unknown"
    role: str           # "player" | "goalkeeper"
    x: float            # pitch coords in meters (0-105)
    y: float            # pitch coords in meters (0-68)
    vx: Optional[float] = None  # m/s
    vy: Optional[float] = None  # m/s
    is_visible: bool = True     # True if detected, False if extrapolated/missing


@dataclass
class Frame:
    frame_idx: int
    timestamp_ms: float
    players: List[PlayerFrame]
    ball_x: Optional[float] = None  # meters (0-105)
    ball_y: Optional[float] = None  # meters (0-68)


@dataclass
class CornerTrackingData:
    corner_id: str      # e.g. "skillcorner_1886347_corner_3"
    source: str         # "skillcorner" | "soccernet_gsr" | "dfl"
    match_id: str
    delivery_frame: int
    fps: float          # 10.0 for SkillCorner, 25.0 for DFL/GSR
    frames: List[Frame]
    outcome: Optional[str] = None  # "shot" | "no_shot" | None
    metadata: Dict[str, Any] = field(default_factory=dict)


# --- Coordinate Transformation ---

# Standard pitch dimensions
PITCH_LENGTH = 105.0  # meters
PITCH_WIDTH = 68.0    # meters


def normalize_to_pitch(x: float, y: float, source: str) -> tuple[float, float]:
    """Transform source-native coordinates to standard pitch (0-105, 0-68).

    Args:
        x: x coordinate in source's native system
        y: y coordinate in source's native system
        source: "skillcorner" | "dfl" | "soccernet_gsr"

    Returns:
        (x_m, y_m) in meters on standard pitch
    """
    if source == "skillcorner":
        # SkillCorner: center-origin, meters. Pitch varies per match
        # (typically 104x68). We offset to corner-origin using the
        # standard half-dimensions. Any residual error (<1m) is within
        # the tolerance of broadcast tracking.
        x_m = x + (PITCH_LENGTH / 2.0)
        y_m = y + (PITCH_WIDTH / 2.0)
    elif source == "dfl":
        # DFL: already in (0-105, 0-68) after kloppy loading
        x_m = x
        y_m = y
    elif source == "soccernet_gsr":
        # GSR: depends on calibration, assume already pitch-relative meters
        x_m = x
        y_m = y
    else:
        raise ValueError(f"Unknown source: {source}")

    # Clamp to pitch bounds
    x_m = max(0.0, min(PITCH_LENGTH, x_m))
    y_m = max(0.0, min(PITCH_WIDTH, y_m))
    return x_m, y_m


# --- Velocity Computation ---

def compute_velocities_central_diff(frames: List[Frame], fps: float) -> List[Frame]:
    """Compute velocities via central difference. Modifies frames in-place.

    v(t) = (pos(t+1) - pos(t-1)) / (2 * dt)
    Edge frames use forward/backward difference.
    Units: m/s.

    Args:
        frames: List of Frame objects (sorted by time)
        fps: Frame rate in Hz

    Returns:
        Same list of frames with vx/vy populated on each PlayerFrame
    """
    if len(frames) < 2:
        return frames

    dt = 1.0 / fps

    # Build position lookup: frame_idx -> player_id -> (x, y)
    n = len(frames)
    for i in range(n):
        frame = frames[i]
        for pf in frame.players:
            vx, vy = 0.0, 0.0

            if i == 0:
                # Forward difference
                next_pf = _find_player(frames[i + 1], pf.player_id)
                if next_pf is not None:
                    vx = (next_pf.x - pf.x) / dt
                    vy = (next_pf.y - pf.y) / dt
            elif i == n - 1:
                # Backward difference
                prev_pf = _find_player(frames[i - 1], pf.player_id)
                if prev_pf is not None:
                    vx = (pf.x - prev_pf.x) / dt
                    vy = (pf.y - prev_pf.y) / dt
            else:
                # Central difference
                prev_pf = _find_player(frames[i - 1], pf.player_id)
                next_pf = _find_player(frames[i + 1], pf.player_id)
                if prev_pf is not None and next_pf is not None:
                    vx = (next_pf.x - prev_pf.x) / (2 * dt)
                    vy = (next_pf.y - prev_pf.y) / (2 * dt)

            pf.vx = vx
            pf.vy = vy

    return frames


def _find_player(frame: Frame, player_id: str) -> Optional[PlayerFrame]:
    """Find a player by ID in a frame."""
    for pf in frame.players:
        if pf.player_id == player_id:
            return pf
    return None


# --- Serialization ---

def _corner_to_dict(corner: CornerTrackingData) -> dict:
    """Convert CornerTrackingData to a JSON-serializable dict."""
    return {
        "corner_id": corner.corner_id,
        "source": corner.source,
        "match_id": corner.match_id,
        "delivery_frame": corner.delivery_frame,
        "fps": corner.fps,
        "outcome": corner.outcome,
        "metadata": corner.metadata,
        "frames": [
            {
                "frame_idx": f.frame_idx,
                "timestamp_ms": f.timestamp_ms,
                "ball_x": f.ball_x,
                "ball_y": f.ball_y,
                "players": [
                    {
                        "player_id": p.player_id,
                        "team": p.team,
                        "role": p.role,
                        "x": p.x,
                        "y": p.y,
                        "vx": p.vx,
                        "vy": p.vy,
                        "is_visible": p.is_visible,
                    }
                    for p in f.players
                ],
            }
            for f in corner.frames
        ],
    }


def _dict_to_corner(d: dict) -> CornerTrackingData:
    """Reconstruct CornerTrackingData from a dict."""
    frames = []
    for fd in d["frames"]:
        players = [
            PlayerFrame(
                player_id=p["player_id"],
                team=p["team"],
                role=p["role"],
                x=p["x"],
                y=p["y"],
                vx=p.get("vx"),
                vy=p.get("vy"),
                is_visible=p.get("is_visible", True),
            )
            for p in fd["players"]
        ]
        frames.append(
            Frame(
                frame_idx=fd["frame_idx"],
                timestamp_ms=fd["timestamp_ms"],
                players=players,
                ball_x=fd.get("ball_x"),
                ball_y=fd.get("ball_y"),
            )
        )

    return CornerTrackingData(
        corner_id=d["corner_id"],
        source=d["source"],
        match_id=d["match_id"],
        delivery_frame=d["delivery_frame"],
        fps=d["fps"],
        outcome=d.get("outcome"),
        metadata=d.get("metadata", {}),
        frames=frames,
    )


def save_corner(corner: CornerTrackingData, path: Path) -> None:
    """Save single corner to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_corner_to_dict(corner), f, indent=2)


def load_corner(path: Path) -> CornerTrackingData:
    """Load single corner from JSON file."""
    with open(path) as f:
        return _dict_to_corner(json.load(f))


def save_dataset(corners: List[CornerTrackingData], output_dir: Path) -> None:
    """Save all corners as individual JSON files + manifest.json index."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    for corner in corners:
        filename = f"{corner.corner_id}.json"
        save_corner(corner, output_dir / filename)

        # Compute summary stats for manifest
        n_players = [len(f.players) for f in corner.frames]
        mean_players = sum(n_players) / len(n_players) if n_players else 0

        manifest_entries.append({
            "corner_id": corner.corner_id,
            "source": corner.source,
            "match_id": corner.match_id,
            "outcome": corner.outcome,
            "file_path": filename,
            "n_frames": len(corner.frames),
            "n_players_mean": round(mean_players, 1),
        })

    manifest = {"corners": manifest_entries}
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def load_dataset(output_dir: Path) -> List[CornerTrackingData]:
    """Load all corners from directory using manifest."""
    output_dir = Path(output_dir)
    with open(output_dir / "manifest.json") as f:
        manifest = json.load(f)

    corners = []
    for entry in manifest["corners"]:
        corner = load_corner(output_dir / entry["file_path"])
        corners.append(corner)
    return corners
