#!/usr/bin/env python3

import argparse
import os
from corner_tactics.data_loader import SoccerNetCornerLoader
from corner_tactics.tracking_processor import TrackingProcessor
from corner_tactics.formation_analyzer import FormationAnalyzer
from corner_tactics.visualizer import TacticalVisualizer

def main():
    parser = argparse.ArgumentParser(description='Analyze corner kick defensive positioning')
    parser.add_argument('--data-path', default='./data/soccernet', help='Path to SoccerNet data')
    parser.add_argument('--download', action='store_true', help='Download SoccerNet datasets')
    parser.add_argument('--password', help='SoccerNet password for download')
    parser.add_argument('--split', default='train', choices=['train', 'valid', 'test'])
    parser.add_argument('--output', default='./results', help='Output directory for results')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    loader = SoccerNetCornerLoader(args.data_path)
    
    if args.download:
        if not args.password:
            print("Password required for downloading SoccerNet data")
            return
        loader.download_datasets(args.password)
    
    print(f"Loading corner timestamps from {args.split} split...")
    corners = loader.load_corner_timestamps(args.split)
    print(f"Found {len(corners)} corner kicks")
    
    processor = TrackingProcessor()
    analyzer = FormationAnalyzer()
    
    print("Analyzing corner sequences...")
    sequences = loader.get_corner_sequences(min_tracking_frames=150)
    
    all_metrics = []
    all_formations = []
    
    for i, sequence in enumerate(sequences[:10]):
        corner = sequence['corner']
        tracking = sequence['tracking']
        
        print(f"Processing corner {i+1}/{min(10, len(sequences))} - {corner.game_id}")
        
        formations = processor.analyze_corner_defense(tracking, defending_team='home')
        
        if formations:
            metrics = processor.calculate_defensive_metrics(formations)
            all_metrics.append(metrics)
            all_formations.extend([f.positions for f in formations])
            
            if args.visualize and i < 3:
                visualizer = TacticalVisualizer()
                
                fig = visualizer.plot_defensive_formation(
                    formations[0].positions,
                    vulnerable_zones=formations[0].vulnerable_zones
                )
                fig.savefig(f"{args.output}/formation_{i}.png", dpi=150, bbox_inches='tight')
                
                positions_over_time = [f.positions for f in formations]
                heatmap_fig = visualizer.create_heatmap(
                    positions_over_time,
                    title=f"Corner Defense Heatmap - {corner.game_id}"
                )
                heatmap_fig.savefig(f"{args.output}/heatmap_{i}.png", dpi=150, bbox_inches='tight')
    
    if all_metrics:
        avg_metrics = {
            'avg_compactness': sum(m.get('avg_compactness', 0) for m in all_metrics) / len(all_metrics),
            'avg_marking_distance': sum(m.get('avg_marking_distance', 0) for m in all_metrics) / len(all_metrics),
            'formation_consistency': sum(m.get('formation_consistency', 0) for m in all_metrics) / len(all_metrics),
            'vulnerable_zone_coverage': sum(m.get('vulnerable_zone_coverage', 0) for m in all_metrics) / len(all_metrics)
        }
        
        print("\n=== Overall Defensive Metrics ===")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.3f}")
        
        recommendations = analyzer.generate_defensive_recommendations(avg_metrics)
        
        print("\n=== Recommendations ===")
        for rec in recommendations:
            print(f"• {rec}")
        
        results = {
            'metrics': avg_metrics,
            'recommendations': recommendations,
            'num_corners_analyzed': len(all_metrics)
        }
        
        analyzer.export_analysis(results, f"{args.output}/analysis_results.json")
        print(f"\nResults saved to {args.output}/analysis_results.json")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()