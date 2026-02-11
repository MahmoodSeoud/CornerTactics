"""DFL tracking data loading, graph construction, and ST-GNN for corner kick prediction."""

from .data_loading import (
    load_tracking_data,
    load_event_data,
    find_corner_events,
    extract_corner_sequence,
    compute_velocities,
)
from .graph_construction import (
    frame_to_graph,
    corner_to_temporal_graphs,
    label_corner,
    build_corner_dataset_from_match,
    save_corner_dataset,
    load_corner_dataset,
    get_dataset_summary,
)
from .visualization import plot_corner_frame
from .model import (
    SpatialGNN,
    TemporalAggregator,
    CornerKickPredictor,
)
from .train import (
    extract_open_play_sequences,
    pretrain_spatial_gnn,
    leave_one_match_out_split,
    compute_multi_task_loss,
    finetune_on_corners,
    zero_out_velocity_features,
    run_ablation,
)
from .evaluate import (
    compute_auc,
    compute_f1,
    aggregate_fold_results,
    paired_t_test,
    analyze_ablation_results,
    format_ablation_report,
)

__all__ = [
    # Data loading
    "load_tracking_data",
    "load_event_data",
    "find_corner_events",
    "extract_corner_sequence",
    "compute_velocities",
    # Graph construction
    "frame_to_graph",
    "corner_to_temporal_graphs",
    "label_corner",
    "build_corner_dataset_from_match",
    "save_corner_dataset",
    "load_corner_dataset",
    "get_dataset_summary",
    # Visualization
    "plot_corner_frame",
    # Model
    "SpatialGNN",
    "TemporalAggregator",
    "CornerKickPredictor",
    # Training
    "extract_open_play_sequences",
    "pretrain_spatial_gnn",
    "leave_one_match_out_split",
    "compute_multi_task_loss",
    "finetune_on_corners",
    "zero_out_velocity_features",
    "run_ablation",
    # Evaluation
    "compute_auc",
    "compute_f1",
    "aggregate_fold_results",
    "paired_t_test",
    "analyze_ablation_results",
    "format_ablation_report",
]
