import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import cv2
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

@dataclass
class DefensiveFormation:
    positions: np.ndarray
    formation_type: str
    compactness: float
    marking_distances: np.ndarray
    vulnerable_zones: List[Tuple[float, float]]

class TrackingProcessor:
    def __init__(self, pitch_width: float = 68.0, pitch_height: float = 105.0):
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        self.defensive_third_x = pitch_width / 3
        
    def normalize_positions(self, positions: List[Dict], frame_width: int, frame_height: int) -> np.ndarray:
        normalized = []
        for pos in positions:
            x_norm = (pos['x'] / frame_width) * self.pitch_width
            y_norm = (pos['y'] / frame_height) * self.pitch_height
            normalized.append([x_norm, y_norm])
        return np.array(normalized)
    
    def classify_defensive_formation(self, defender_positions: np.ndarray) -> str:
        if len(defender_positions) < 4:
            return "incomplete"
            
        sorted_by_y = defender_positions[defender_positions[:, 1].argsort()]
        
        defensive_lines = []
        current_line = [sorted_by_y[0]]
        
        for pos in sorted_by_y[1:]:
            if abs(pos[1] - current_line[-1][1]) < 5.0:
                current_line.append(pos)
            else:
                defensive_lines.append(current_line)
                current_line = [pos]
        defensive_lines.append(current_line)
        
        line_counts = [len(line) for line in defensive_lines]
        line_counts.sort(reverse=True)
        
        formations = {
            (4, 4): "4-4-2",
            (5, 3): "5-3-2", 
            (4, 3): "4-3-3",
            (3, 5): "3-5-2",
            (4, 2): "4-2-3-1"
        }
        
        key = tuple(line_counts[:2]) if len(line_counts) >= 2 else (line_counts[0], 0)
        return formations.get(key, "custom")
    
    def calculate_compactness(self, positions: np.ndarray) -> float:
        if len(positions) < 2:
            return 0.0
            
        hull = cv2.convexHull(positions.astype(np.float32))
        area = cv2.contourArea(hull)
        
        max_area = (self.pitch_width / 2) * (self.pitch_height / 2)
        return 1.0 - (area / max_area)
    
    def identify_marking_assignments(self, defenders: np.ndarray, attackers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(defenders) == 0 or len(attackers) == 0:
            return np.array([]), np.array([])
            
        dist_matrix = distance_matrix(defenders, attackers)
        
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        marking_distances = dist_matrix[row_ind, col_ind]
        
        return row_ind, marking_distances
    
    def find_vulnerable_zones(self, defenders: np.ndarray, goal_position: Tuple[float, float] = (0, 52.5)) -> List[Tuple[float, float]]:
        vulnerable = []
        
        danger_zone = {
            'x_min': 0,
            'x_max': 16.5,
            'y_min': goal_position[1] - 20,
            'y_max': goal_position[1] + 20
        }
        
        grid_size = 2.0
        for x in np.arange(danger_zone['x_min'], danger_zone['x_max'], grid_size):
            for y in np.arange(danger_zone['y_min'], danger_zone['y_max'], grid_size):
                point = np.array([x, y])
                
                if len(defenders) > 0:
                    min_dist = np.min(np.linalg.norm(defenders - point, axis=1))
                    if min_dist > 5.0:
                        vulnerable.append((x, y))
                        
        return vulnerable
    
    def analyze_corner_defense(self, tracking_frames: List, defending_team: str) -> List[DefensiveFormation]:
        formations = []
        
        for frame in tracking_frames:
            defenders = [p for p in frame.players if p['team'] == defending_team]
            attackers = [p for p in frame.players if p['team'] != defending_team]
            
            if len(defenders) < 4:
                continue
                
            def_positions = self.normalize_positions(defenders, 1920, 1080)
            att_positions = self.normalize_positions(attackers, 1920, 1080)
            
            formation_type = self.classify_defensive_formation(def_positions)
            compactness = self.calculate_compactness(def_positions)
            
            _, marking_distances = self.identify_marking_assignments(def_positions, att_positions)
            
            vulnerable = self.find_vulnerable_zones(def_positions)
            
            formations.append(DefensiveFormation(
                positions=def_positions,
                formation_type=formation_type,
                compactness=compactness,
                marking_distances=marking_distances,
                vulnerable_zones=vulnerable
            ))
            
        return formations
    
    def calculate_defensive_metrics(self, formations: List[DefensiveFormation]) -> Dict:
        if not formations:
            return {}
            
        metrics = {
            'avg_compactness': np.mean([f.compactness for f in formations]),
            'formation_consistency': self._calculate_formation_consistency(formations),
            'avg_marking_distance': np.mean([np.mean(f.marking_distances) if len(f.marking_distances) > 0 else 0 for f in formations]),
            'vulnerable_zone_coverage': self._calculate_zone_coverage(formations),
            'most_common_formation': self._most_common_formation(formations)
        }
        
        return metrics
    
    def _calculate_formation_consistency(self, formations: List[DefensiveFormation]) -> float:
        if len(formations) < 2:
            return 1.0
            
        formation_types = [f.formation_type for f in formations]
        changes = sum(1 for i in range(1, len(formation_types)) if formation_types[i] != formation_types[i-1])
        
        return 1.0 - (changes / len(formations))
    
    def _calculate_zone_coverage(self, formations: List[DefensiveFormation]) -> float:
        if not formations:
            return 0.0
            
        total_vulnerable = sum(len(f.vulnerable_zones) for f in formations)
        max_zones = len(formations) * 100
        
        return 1.0 - (total_vulnerable / max_zones)
    
    def _most_common_formation(self, formations: List[DefensiveFormation]) -> str:
        if not formations:
            return "unknown"
            
        formation_counts = {}
        for f in formations:
            formation_counts[f.formation_type] = formation_counts.get(f.formation_type, 0) + 1
            
        return max(formation_counts, key=formation_counts.get)