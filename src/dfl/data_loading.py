"""Data loading utilities for DFL and Metrica tracking data.

Uses kloppy library to load tracking and event data from different providers
into a unified format.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

from kloppy import metrica
from kloppy.domain import EventType


def load_tracking_data(provider: str, data_dir: Path, match_id: Optional[str] = None):
    """
    Load tracking data from the specified provider.

    Args:
        provider: Data provider name ("metrica" or "dfl")
        data_dir: Path to the data directory for the match
        match_id: Optional match identifier (for DFL, e.g., "DFL-MAT-J03WMX")

    Returns:
        kloppy TrackingDataset object
    """
    data_dir = Path(data_dir)

    if provider == "metrica":
        return _load_metrica_tracking(data_dir)
    elif provider == "dfl":
        return _load_dfl_tracking(data_dir, match_id=match_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _load_metrica_tracking(data_dir: Path):
    """Load Metrica tracking data using kloppy."""
    # Find tracking file (Sample_Game_3 format)
    tracking_file = data_dir / "Sample_Game_3_tracking.txt"
    metadata_file = data_dir / "Sample_Game_3_metadata.xml"

    if tracking_file.exists() and metadata_file.exists():
        return metrica.load_tracking_epts(
            raw_data=str(tracking_file),
            meta_data=str(metadata_file),
        )

    # Try older format (Sample_Game_1/2)
    home_file = list(data_dir.glob("*_RawTrackingData_Home_Team.csv"))
    away_file = list(data_dir.glob("*_RawTrackingData_Away_Team.csv"))

    if home_file and away_file:
        return metrica.load_tracking_csv(
            home_data=str(home_file[0]),
            away_data=str(away_file[0]),
        )

    raise FileNotFoundError(f"No Metrica tracking data found in {data_dir}")


def _load_dfl_tracking(data_dir: Path, match_id: Optional[str] = None):
    """Load DFL tracking data using kloppy sportec loader."""
    # Find tracking and metadata files
    position_files = sorted(data_dir.glob("*positions_raw*.xml"))
    metadata_files = sorted(data_dir.glob("*matchinformation*.xml"))

    if not position_files:
        raise FileNotFoundError(f"No DFL position files found in {data_dir}")
    if not metadata_files:
        raise FileNotFoundError(f"No DFL metadata files found in {data_dir}")

    # Select specific match if match_id provided
    if match_id:
        position_files = [f for f in position_files if match_id in str(f)]
        metadata_files = [f for f in metadata_files if match_id in str(f)]

        if not position_files or not metadata_files:
            raise FileNotFoundError(f"No files found for match_id: {match_id}")

    # Use kloppy's sportec loader (DFL uses Sportec format)
    from kloppy import sportec

    return sportec.load_tracking(
        raw_data=str(position_files[0]),
        meta_data=str(metadata_files[0]),
    )


def load_event_data(provider: str, data_dir: Path, match_id: Optional[str] = None):
    """
    Load event data from the specified provider.

    Args:
        provider: Data provider name ("metrica" or "dfl")
        data_dir: Path to the data directory for the match
        match_id: Optional match identifier (for DFL, e.g., "DFL-MAT-J03WMX")

    Returns:
        kloppy EventDataset object
    """
    data_dir = Path(data_dir)

    if provider == "metrica":
        return _load_metrica_events(data_dir)
    elif provider == "dfl":
        return _load_dfl_events(data_dir, match_id=match_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _load_metrica_events(data_dir: Path):
    """Load Metrica event data using kloppy."""
    # Try JSON format first (Sample_Game_3)
    event_file = data_dir / "Sample_Game_3_events.json"
    if event_file.exists():
        return metrica.load_event(
            event_data=str(event_file),
            meta_data=str(data_dir / "Sample_Game_3_metadata.xml"),
        )

    # Try CSV format (Sample_Game_1/2)
    event_csv = list(data_dir.glob("*_RawEventsData.csv"))
    if event_csv:
        return metrica.load_event(
            event_data=str(event_csv[0]),
        )

    raise FileNotFoundError(f"No Metrica event data found in {data_dir}")


def _load_dfl_events(data_dir: Path, match_id: Optional[str] = None):
    """Load DFL event data using kloppy sportec loader."""
    event_files = sorted(data_dir.glob("*events_raw*.xml"))
    metadata_files = sorted(data_dir.glob("*matchinformation*.xml"))

    if not event_files:
        raise FileNotFoundError(f"No DFL event files found in {data_dir}")

    # Select specific match if match_id provided
    if match_id:
        event_files = [f for f in event_files if match_id in str(f)]
        metadata_files = [f for f in metadata_files if match_id in str(f)]

        if not event_files:
            raise FileNotFoundError(f"No event files found for match_id: {match_id}")

    from kloppy import sportec

    return sportec.load_event(
        event_data=str(event_files[0]),
        meta_data=str(metadata_files[0]) if metadata_files else None,
    )


def find_corner_events(event_dataset) -> List:
    """
    Find all corner kick events in an event dataset.

    Args:
        event_dataset: kloppy EventDataset

    Returns:
        List of corner kick events
    """
    corners = []

    for event in event_dataset.events:
        # Check event type
        event_type_str = str(event.event_type).lower()

        # Check if it's a corner kick directly
        if "corner" in event_type_str:
            corners.append(event)
            continue

        # Check qualifiers for corner kick
        if hasattr(event, "qualifiers") and event.qualifiers:
            for q in event.qualifiers:
                q_str = str(q).lower()
                if "corner" in q_str:
                    corners.append(event)
                    break

    return corners


def extract_corner_sequence(
    tracking_dataset,
    corner_event,
    pre_seconds: float = 2.0,
    post_seconds: float = 6.0,
) -> List:
    """
    Extract tracking frames around a corner kick event.

    Args:
        tracking_dataset: kloppy TrackingDataset
        corner_event: The corner kick event
        pre_seconds: Seconds before corner to include
        post_seconds: Seconds after corner to include

    Returns:
        List of tracking frames within the time window
    """
    from datetime import timedelta

    corner_time = corner_event.timestamp
    corner_period = corner_event.period if hasattr(corner_event, "period") else None

    # Convert to timedelta if needed
    if hasattr(corner_time, "total_seconds"):
        pre_delta = timedelta(seconds=pre_seconds)
        post_delta = timedelta(seconds=post_seconds)
        start_time = corner_time - pre_delta
        end_time = corner_time + post_delta
    else:
        # Assume numeric seconds
        start_time = corner_time - pre_seconds
        end_time = corner_time + post_seconds

    window_frames = []
    for frame in tracking_dataset.records:
        # Match period if available (timestamps can be relative to period start)
        if corner_period is not None and hasattr(frame, "period"):
            if frame.period != corner_period:
                continue

        if start_time <= frame.timestamp <= end_time:
            window_frames.append(frame)

    return window_frames


def compute_velocities(
    frames: List,
    fps: int = 25,
) -> List[Dict[str, Dict[str, float]]]:
    """
    Compute velocity vectors using central difference.

    v(t) = (pos(t+1) - pos(t-1)) / (2 * dt)

    Args:
        frames: List of tracking frames
        fps: Frame rate

    Returns:
        List of dicts, one per frame, mapping player_id -> {"vx": float, "vy": float}
    """
    dt = 1.0 / fps
    velocities = []

    for i in range(len(frames)):
        frame_vel = {}

        if i == 0 or i == len(frames) - 1:
            # Edge frames: use forward/backward difference
            if i == 0 and len(frames) > 1:
                for player_id, pdata in frames[i].players_data.items():
                    if player_id in frames[i + 1].players_data:
                        curr = pdata
                        next_p = frames[i + 1].players_data[player_id]
                        if curr.coordinates and next_p.coordinates:
                            vx = (next_p.coordinates.x - curr.coordinates.x) / dt
                            vy = (next_p.coordinates.y - curr.coordinates.y) / dt
                        else:
                            vx, vy = 0.0, 0.0
                    else:
                        vx, vy = 0.0, 0.0
                    frame_vel[player_id] = {"vx": vx, "vy": vy}
            elif i == len(frames) - 1 and len(frames) > 1:
                for player_id, pdata in frames[i].players_data.items():
                    if player_id in frames[i - 1].players_data:
                        curr = pdata
                        prev_p = frames[i - 1].players_data[player_id]
                        if curr.coordinates and prev_p.coordinates:
                            vx = (curr.coordinates.x - prev_p.coordinates.x) / dt
                            vy = (curr.coordinates.y - prev_p.coordinates.y) / dt
                        else:
                            vx, vy = 0.0, 0.0
                    else:
                        vx, vy = 0.0, 0.0
                    frame_vel[player_id] = {"vx": vx, "vy": vy}
            else:
                for player_id in frames[i].players_data:
                    frame_vel[player_id] = {"vx": 0.0, "vy": 0.0}
        else:
            # Central difference
            for player_id, pdata in frames[i].players_data.items():
                prev_p = frames[i - 1].players_data.get(player_id)
                next_p = frames[i + 1].players_data.get(player_id)

                if prev_p and next_p and prev_p.coordinates and next_p.coordinates:
                    vx = (next_p.coordinates.x - prev_p.coordinates.x) / (2 * dt)
                    vy = (next_p.coordinates.y - prev_p.coordinates.y) / (2 * dt)
                else:
                    vx, vy = 0.0, 0.0

                frame_vel[player_id] = {"vx": vx, "vy": vy}

        velocities.append(frame_vel)

    return velocities
