import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import DBSCAN
from collections import Counter
import json

class FormationAnalyzer:
    def __init__(self):
        self.defensive_patterns = {}
        self.successful_defenses = []
        self.failed_defenses = []
        
    def cluster_defensive_shapes(self, all_formations: List[np.ndarray], eps: float = 3.0) -> Dict:
        if not all_formations:
            return {}
            
        flattened = []
        for formation in all_formations:
            if len(formation) >= 10:
                sorted_pos = formation[formation[:, 0].argsort()][:10]
                flattened.append(sorted_pos.flatten())
                
        if not flattened:
            return {}
            
        X = np.array(flattened)
        clustering = DBSCAN(eps=eps, min_samples=5).fit(X)
        
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label != -1:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(all_formations[i])
                
        pattern_stats = {}
        for label, formations in clusters.items():
            pattern_stats[f"pattern_{label}"] = {
                "count": len(formations),
                "avg_shape": np.mean(formations, axis=0),
                "std_dev": np.std(formations, axis=0)
            }
            
        return pattern_stats
    
    def analyze_marking_effectiveness(self, marking_distances: List[np.ndarray], outcomes: List[bool]) -> Dict:
        effective_markings = []
        ineffective_markings = []
        
        for distances, success in zip(marking_distances, outcomes):
            if success:
                effective_markings.extend(distances)
            else:
                ineffective_markings.extend(distances)
                
        stats = {
            "effective_avg_distance": np.mean(effective_markings) if effective_markings else 0,
            "ineffective_avg_distance": np.mean(ineffective_markings) if ineffective_markings else 0,
            "optimal_marking_distance": np.percentile(effective_markings, 25) if effective_markings else 0,
            "critical_distance_threshold": np.percentile(ineffective_markings, 75) if ineffective_markings else 0
        }
        
        return stats
    
    def identify_key_positions(self, successful_formations: List[np.ndarray]) -> List[Tuple[float, float]]:
        if not successful_formations:
            return []
            
        all_positions = np.vstack(successful_formations)
        
        clustering = DBSCAN(eps=2.0, min_samples=10).fit(all_positions)
        
        key_positions = []
        for label in set(clustering.labels_):
            if label != -1:
                cluster_points = all_positions[clustering.labels_ == label]
                centroid = np.mean(cluster_points, axis=0)
                key_positions.append(tuple(centroid))
                
        return key_positions
    
    def calculate_formation_transitions(self, formation_sequence: List[str]) -> Dict:
        if len(formation_sequence) < 2:
            return {}
            
        transitions = []
        for i in range(len(formation_sequence) - 1):
            transitions.append((formation_sequence[i], formation_sequence[i+1]))
            
        transition_counts = Counter(transitions)
        
        transition_matrix = {}
        for (from_f, to_f), count in transition_counts.items():
            if from_f not in transition_matrix:
                transition_matrix[from_f] = {}
            transition_matrix[from_f][to_f] = count
            
        for from_f in transition_matrix:
            total = sum(transition_matrix[from_f].values())
            for to_f in transition_matrix[from_f]:
                transition_matrix[from_f][to_f] /= total
                
        return transition_matrix
    
    def generate_defensive_recommendations(self, analysis_results: Dict) -> List[str]:
        recommendations = []
        
        if analysis_results.get('avg_compactness', 0) < 0.6:
            recommendations.append("Increase defensive compactness - players are too spread out")
            
        if analysis_results.get('avg_marking_distance', float('inf')) > 3.0:
            recommendations.append("Tighten marking - defenders are too far from attackers")
            
        if analysis_results.get('formation_consistency', 0) < 0.7:
            recommendations.append("Maintain formation shape throughout the corner sequence")
            
        if analysis_results.get('vulnerable_zone_coverage', 0) < 0.8:
            recommendations.append("Improve coverage of near-post and far-post zones")
            
        common_formation = analysis_results.get('most_common_formation', '')
        if common_formation in ['3-5-2', '5-3-2']:
            recommendations.append(f"Consider more defenders in the box for {common_formation} formation")
            
        return recommendations
    
    def export_analysis(self, results: Dict, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
    def compare_teams(self, team_a_metrics: Dict, team_b_metrics: Dict) -> Dict:
        comparison = {}
        
        for metric in team_a_metrics:
            if metric in team_b_metrics:
                comparison[metric] = {
                    'team_a': team_a_metrics[metric],
                    'team_b': team_b_metrics[metric],
                    'difference': team_a_metrics[metric] - team_b_metrics[metric]
                }
                
        return comparison