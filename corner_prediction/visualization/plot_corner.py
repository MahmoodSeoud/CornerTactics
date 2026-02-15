"""Figure 1: Receiver prediction example on a pitch.

Plots one corner kick showing player positions, receiver probabilities,
true receiver, and predicted receiver on a half-pitch diagram.
"""

import pickle
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mplsoccer import Pitch

from corner_prediction.config import DATA_DIR, PRETRAINED_PATH


def _find_example_corner(records: list) -> dict:
    """Pick a good example corner for visualization.

    Prefers: has_receiver_label, lead_to_shot, high detection_rate.
    """
    # Best: shot + receiver + high detection
    candidates = [
        r for r in records
        if r["has_receiver_label"] and r["lead_to_shot"]
    ]
    if not candidates:
        candidates = [r for r in records if r["has_receiver_label"]]
    if not candidates:
        candidates = records

    return max(candidates, key=lambda r: r["detection_rate"])


def _get_receiver_probs(corner_record: dict) -> Optional[np.ndarray]:
    """Run a forward pass to get receiver probabilities for one corner.

    Returns array of shape [22] with probabilities (0 for masked-out players),
    or None if the model cannot be loaded.
    """
    try:
        import torch
        from corner_prediction.data.build_graphs import corner_record_to_graph
        from corner_prediction.training.train import build_model

        graph = corner_record_to_graph(corner_record, edge_type="knn", k=6)

        pretrained_path = PRETRAINED_PATH if PRETRAINED_PATH.exists() else None
        backbone_mode = "pretrained" if pretrained_path else "scratch"
        model = build_model(
            backbone_mode=backbone_mode,
            pretrained_path=str(pretrained_path) if pretrained_path else None,
            freeze=(backbone_mode == "pretrained"),
        )
        model.eval()

        with torch.no_grad():
            batch_idx = torch.zeros(graph.x.shape[0], dtype=torch.long)
            probs = model.predict_receiver(
                graph.x, graph.edge_index, graph.edge_attr,
                graph.receiver_mask, batch_idx,
            )
            return probs.cpu().numpy()
    except Exception as e:
        print(f"  Warning: Could not compute receiver probs: {e}")
        return None


def plot_receiver_example(
    corner_record: dict,
    receiver_probs: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot a corner kick with receiver prediction on a half-pitch.

    Args:
        corner_record: Raw corner record from extracted_corners.pkl.
        receiver_probs: Optional [22] array of receiver probabilities.
        output_path: Path to save PDF.
        show: Whether to display interactively.

    Returns:
        matplotlib Figure.
    """
    players = corner_record["players"]

    # Separate attackers and defenders
    atk_players = [p for p in players if p["is_attacking"]]
    def_players = [p for p in players if not p["is_attacking"]]

    # Build pitch (horizontal, half-pitch = attacking half x>0)
    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=105,
        pitch_width=68,
        half=True,
        goal_type="box",
        linewidth=1.5,
        line_color="#888888",
        pitch_color="white",
    )
    fig, ax = pitch.draw(figsize=(10, 7))

    # --- Defenders: blue circles ---
    def_x = [p["x"] for p in def_players]
    def_y = [p["y"] for p in def_players]
    ax.scatter(def_x, def_y, s=200, c="#4393C3", edgecolors="black",
               linewidth=0.8, zorder=4, alpha=0.8)

    # --- Attackers: colored by receiver probability ---
    atk_x = [p["x"] for p in atk_players]
    atk_y = [p["y"] for p in atk_players]

    if receiver_probs is not None:
        # Map player indices to probs
        atk_probs = []
        for p in atk_players:
            idx = players.index(p)
            atk_probs.append(receiver_probs[idx])
        atk_probs = np.array(atk_probs)

        norm = Normalize(vmin=0, vmax=max(atk_probs.max(), 0.01))
        cmap = plt.cm.Reds
        colors = cmap(norm(atk_probs))
        sc = ax.scatter(atk_x, atk_y, s=250, c=atk_probs, cmap=cmap,
                        norm=norm, edgecolors="black", linewidth=0.8,
                        zorder=5)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Receiver Probability", fontsize=11)
    else:
        ax.scatter(atk_x, atk_y, s=250, c="#D6604D", edgecolors="black",
                   linewidth=0.8, zorder=5, alpha=0.8)

    # --- True receiver: star marker ---
    receiver_id = corner_record.get("receiver_id")
    if receiver_id:
        recv_player = next((p for p in players if p["player_id"] == receiver_id), None)
        if recv_player:
            ax.scatter([recv_player["x"]], [recv_player["y"]],
                       s=400, marker="*", c="gold", edgecolors="black",
                       linewidth=1.0, zorder=7, label="True receiver")

    # --- Predicted receiver: diamond marker ---
    if receiver_probs is not None:
        # Mask: only attacking outfield
        mask = np.array([
            p["is_attacking"] and not p["is_goalkeeper"] for p in players
        ])
        masked_probs = receiver_probs.copy()
        masked_probs[~mask] = -1
        pred_idx = masked_probs.argmax()
        pred_player = players[pred_idx]
        ax.scatter([pred_player["x"]], [pred_player["y"]],
                   s=350, marker="D", c="lime", edgecolors="black",
                   linewidth=1.2, zorder=6, label="Predicted receiver")

    # --- Corner taker: annotate ---
    taker = next((p for p in players if p["is_corner_taker"]), None)
    if taker:
        ax.annotate("Taker", (taker["x"], taker["y"]),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=9, ha="center", fontweight="bold", zorder=8)

    # --- Ball position ---
    ball_x = corner_record.get("ball_x")
    ball_y = corner_record.get("ball_y")
    if ball_x is not None and ball_y is not None:
        ax.scatter([ball_x], [ball_y], s=120, c="white",
                   edgecolors="black", linewidth=1.5, zorder=8,
                   marker="o", label="Ball")

    # --- Arrow from taker to predicted receiver ---
    if taker and receiver_probs is not None:
        ax.annotate(
            "", xy=(pred_player["x"], pred_player["y"]),
            xytext=(taker["x"], taker["y"]),
            arrowprops=dict(
                arrowstyle="->", color="gray", lw=1.5,
                connectionstyle="arc3,rad=0.2",
            ),
            zorder=3,
        )

    # --- Velocity arrows for all players ---
    for p in players:
        if p["speed"] > 0.5:  # only show meaningful movement
            scale = 2.0  # arrow length scaling
            ax.annotate(
                "", xy=(p["x"] + p["vx"] * scale, p["y"] + p["vy"] * scale),
                xytext=(p["x"], p["y"]),
                arrowprops=dict(arrowstyle="-|>", color="gray",
                                lw=0.8, alpha=0.5),
                zorder=3,
            )

    # --- Legend ---
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D6604D",
               markeredgecolor="black", markersize=10, label="Attacker"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4393C3",
               markeredgecolor="black", markersize=10, label="Defender"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="black", markersize=14, label="True receiver"),
    ]
    if receiver_probs is not None:
        legend_elements.append(
            Line2D([0], [0], marker="D", color="w", markerfacecolor="lime",
                   markeredgecolor="black", markersize=10,
                   label="Predicted receiver"),
        )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=8, label="Ball"),
    )
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              framealpha=0.9)

    # --- Title ---
    side = corner_record.get("corner_side", "?")
    shot_str = "Shot" if corner_record["lead_to_shot"] else "No shot"
    det = corner_record["detection_rate"]
    ax.set_title(
        f"Corner Kick: {side.capitalize()} side | {shot_str} | "
        f"Detection: {det:.0%}",
        fontsize=13, fontweight="bold", pad=10,
    )

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {output_path}")

    if show:
        plt.show()

    return fig


def generate(
    output_dir: Path,
    records: Optional[list] = None,
    show: bool = False,
) -> Optional[Path]:
    """Generate Figure 1 and return the output path."""
    if records is None:
        records_path = DATA_DIR / "extracted_corners.pkl"
        if not records_path.exists():
            print("  Skipping Figure 1: extracted_corners.pkl not found")
            return None
        with open(records_path, "rb") as f:
            records = pickle.load(f)

    corner = _find_example_corner(records)
    print(f"  Example corner: {corner['corner_id']} "
          f"(shot={corner['lead_to_shot']}, det={corner['detection_rate']:.0%})")

    probs = _get_receiver_probs(corner)

    output_path = output_dir / "fig1_receiver_example.pdf"
    plot_receiver_example(corner, receiver_probs=probs,
                          output_path=str(output_path), show=show)
    plt.close("all")
    return output_path
