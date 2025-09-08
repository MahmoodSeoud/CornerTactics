import os
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.ActionSpotting import getListGames

@dataclass
class CornerEvent:
    game_id: str
    half: int
    timestamp: float
    confidence: float
    team: str
    
@dataclass 
class TrackingFrame:
    frame_id: int
    timestamp: float
    players: List[Dict[str, float]]
    ball_position: Optional[Tuple[float, float]]

class SoccerNetCornerLoader:
    def __init__(self, data_path: str = "./data/soccernet"):
        self.data_path = data_path
        self.downloader = SoccerNetDownloader(LocalDirectory=data_path)
        self.corner_events = []
        self.tracking_data = {}
        
    def download_datasets(self, password: str, tracking_only: bool = False):
        self.downloader.password = password
        
        if not tracking_only:
            print("Downloading SoccerNet-v2 annotations...")
            self.downloader.downloadGames(files=["Labels-v2.json"], split=["train"])
        
        print("Downloading SoccerNet-Tracking data (train split only)...")
        self.downloader.downloadDataTask(task="tracking", split=["train"])
        
    def load_corner_timestamps(self, split: str = "train") -> List[CornerEvent]:
        games = getListGames(split, task="spotting")
        corners = []
        
        for game in games:
            labels_path = os.path.join(self.data_path, game, "Labels-v2.json")
            if not os.path.exists(labels_path):
                continue
                
            with open(labels_path, 'r') as f:
                labels = json.load(f)
                
            for annotation in labels.get("annotations", []):
                if annotation["label"] == "Corner":
                    corner = CornerEvent(
                        game_id=game,
                        half=annotation["gameTime"].split(" - ")[0],
                        timestamp=self._parse_time(annotation["gameTime"]),
                        confidence=annotation.get("confidence", 1.0),
                        team=annotation.get("team", "unknown")
                    )
                    corners.append(corner)
                    
        self.corner_events = corners
        return corners
    
    def load_tracking_for_corner(self, corner: CornerEvent, window_seconds: float = 30) -> List[TrackingFrame]:
        tracking_path = os.path.join(
            self.data_path, 
            corner.game_id, 
            f"{corner.half}_tracking.json"
        )
        
        if not os.path.exists(tracking_path):
            return []
            
        with open(tracking_path, 'r') as f:
            tracking = json.load(f)
            
        frames = []
        start_time = corner.timestamp - window_seconds / 2
        end_time = corner.timestamp + window_seconds / 2
        
        for frame_data in tracking.get("frames", []):
            frame_time = frame_data["timestamp"]
            if start_time <= frame_time <= end_time:
                players = []
                for detection in frame_data.get("detections", []):
                    if detection["class"] == "player":
                        players.append({
                            "id": detection["track_id"],
                            "x": detection["bbox"][0] + detection["bbox"][2]/2,
                            "y": detection["bbox"][1] + detection["bbox"][3]/2,
                            "team": detection.get("team", "unknown"),
                            "confidence": detection["confidence"]
                        })
                
                ball_det = next((d for d in frame_data.get("detections", []) if d["class"] == "ball"), None)
                ball_pos = None
                if ball_det:
                    ball_pos = (
                        ball_det["bbox"][0] + ball_det["bbox"][2]/2,
                        ball_det["bbox"][1] + ball_det["bbox"][3]/2
                    )
                    
                frames.append(TrackingFrame(
                    frame_id=frame_data["frame_id"],
                    timestamp=frame_time,
                    players=players,
                    ball_position=ball_pos
                ))
                
        return frames
    
    def _parse_time(self, time_str: str) -> float:
        parts = time_str.split(" - ")[1].split(":")
        return float(parts[0]) * 60 + float(parts[1])
    
    def get_corner_sequences(self, min_tracking_frames: int = 150) -> List[Dict]:
        sequences = []
        
        for corner in self.corner_events:
            tracking = self.load_tracking_for_corner(corner)
            if len(tracking) >= min_tracking_frames:
                sequences.append({
                    "corner": corner,
                    "tracking": tracking
                })
                
        return sequences