import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import cv2
from typing import List, Dict, Tuple, Optional

class TacticalVisualizer:
    def __init__(self):
        self.pitch_color = '#2e7d32'
        self.line_color = 'white'
        self.fig = None
        self.ax = None
        
    def setup_pitch(self, figsize: Tuple[int, int] = (12, 8)) -> Tuple:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 68)
        ax.set_ylim(0, 105)
        ax.set_aspect('equal')
        ax.set_facecolor(self.pitch_color)
        
        pitch = patches.Rectangle((0, 0), 68, 105, linewidth=2, 
                                 edgecolor=self.line_color, facecolor=self.pitch_color)
        ax.add_patch(pitch)
        
        penalty_area = patches.Rectangle((0, 21.5), 16.5, 62, linewidth=2,
                                        edgecolor=self.line_color, fill=False)
        ax.add_patch(penalty_area)
        
        six_yard = patches.Rectangle((0, 36.5), 5.5, 32, linewidth=2,
                                    edgecolor=self.line_color, fill=False)
        ax.add_patch(six_yard)
        
        goal = patches.Rectangle((0, 47.5), 2, 10, linewidth=3,
                                edgecolor='black', fill=False)
        ax.add_patch(goal)
        
        center_circle = patches.Circle((34, 52.5), 9.15, linewidth=2,
                                      edgecolor=self.line_color, fill=False)
        ax.add_patch(center_circle)
        
        ax.plot([34, 34], [0, 105], color=self.line_color, linewidth=2)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        self.fig = fig
        self.ax = ax
        return fig, ax
    
    def plot_defensive_formation(self, positions: np.ndarray, 
                                attackers: Optional[np.ndarray] = None,
                                ball_pos: Optional[Tuple[float, float]] = None,
                                vulnerable_zones: Optional[List] = None):
        
        fig, ax = self.setup_pitch()
        
        ax.scatter(positions[:, 0], positions[:, 1], 
                  c='blue', s=200, edgecolors='white', linewidth=2, 
                  label='Defenders', zorder=5)
        
        for i, pos in enumerate(positions):
            ax.text(pos[0], pos[1], str(i+1), 
                   color='white', fontsize=10, ha='center', va='center')
        
        if attackers is not None:
            ax.scatter(attackers[:, 0], attackers[:, 1],
                      c='red', s=200, edgecolors='white', linewidth=2,
                      label='Attackers', zorder=5)
        
        if ball_pos is not None:
            ax.scatter(ball_pos[0], ball_pos[1], 
                      c='white', s=100, edgecolors='black', linewidth=2,
                      label='Ball', zorder=6)
        
        if vulnerable_zones:
            for zone in vulnerable_zones:
                circle = patches.Circle(zone, 2.0, alpha=0.3, 
                                       facecolor='red', edgecolor='red')
                ax.add_patch(circle)
        
        ax.legend(loc='upper right')
        plt.title('Defensive Formation Analysis', fontsize=14, fontweight='bold')
        
        return fig
    
    def create_heatmap(self, positions_over_time: List[np.ndarray], 
                      title: str = "Defensive Position Heatmap"):
        
        heatmap = np.zeros((105, 68))
        
        for positions in positions_over_time:
            for pos in positions:
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < 68 and 0 <= y < 105:
                    cv2.circle(heatmap, (x, y), 3, 1, -1)
        
        heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)
        
        fig, ax = self.setup_pitch()
        
        im = ax.imshow(heatmap.T, extent=[0, 68, 0, 105], 
                      origin='lower', cmap='hot', alpha=0.6, zorder=2)
        
        plt.colorbar(im, ax=ax, label='Position Frequency')
        plt.title(title, fontsize=14, fontweight='bold')
        
        return fig
    
    def animate_corner_sequence(self, tracking_frames: List, 
                               defending_team: str,
                               save_path: Optional[str] = None):
        
        fig, ax = self.setup_pitch()
        
        defenders_scatter = ax.scatter([], [], c='blue', s=200, 
                                     edgecolors='white', linewidth=2)
        attackers_scatter = ax.scatter([], [], c='red', s=200,
                                     edgecolors='white', linewidth=2)
        ball_scatter = ax.scatter([], [], c='white', s=100,
                                 edgecolors='black', linewidth=2)
        
        def update(frame_idx):
            frame = tracking_frames[frame_idx]
            
            defenders = np.array([[p['x'], p['y']] for p in frame.players 
                                 if p['team'] == defending_team])
            attackers = np.array([[p['x'], p['y']] for p in frame.players
                                if p['team'] != defending_team])
            
            defenders_scatter.set_offsets(defenders * [68/1920, 105/1080])
            attackers_scatter.set_offsets(attackers * [68/1920, 105/1080])
            
            if frame.ball_position:
                ball_pos = np.array([[frame.ball_position[0] * 68/1920,
                                    frame.ball_position[1] * 105/1080]])
                ball_scatter.set_offsets(ball_pos)
            
            ax.set_title(f'Corner Defense - Frame {frame_idx}/{len(tracking_frames)}')
            
            return defenders_scatter, attackers_scatter, ball_scatter
        
        anim = FuncAnimation(fig, update, frames=len(tracking_frames),
                           interval=50, blit=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
            
        return anim
    
    def plot_3d_formation(self, positions: np.ndarray, time_dimension: np.ndarray):
        
        fig = go.Figure(data=[go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=time_dimension,
            mode='markers+lines',
            marker=dict(
                size=8,
                color=time_dimension,
                colorscale='Viridis',
                showscale=True
            ),
            line=dict(
                color='darkblue',
                width=2
            )
        )])
        
        fig.update_layout(
            title='Defensive Formation Evolution',
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Time',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=False
        )
        
        return fig
    
    def create_comparison_plot(self, metrics_a: Dict, metrics_b: Dict,
                             team_a_name: str = "Team A",
                             team_b_name: str = "Team B"):
        
        metrics = list(metrics_a.keys())
        values_a = list(metrics_a.values())
        values_b = list(metrics_b.values())
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, values_a, width, label=team_a_name, color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, values_b, width, label=team_b_name, color='red', alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Defensive Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontsize=8)
        
        plt.tight_layout()
        return fig