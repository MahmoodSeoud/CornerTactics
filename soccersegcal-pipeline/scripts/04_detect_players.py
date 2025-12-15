#!/usr/bin/env python3
"""
Step 4: Player detection using YOLOv8.

Detects players in corner frames and extracts their bounding boxes.
The foot position (bottom center of bbox) is used for pitch projection.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_yolo_model(model_name: str = 'yolov8x.pt'):
    """
    Load YOLOv8 model for person detection.

    Args:
        model_name: Model name/path (yolov8n/s/m/l/x)

    Returns:
        Loaded YOLO model
    """
    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        return model
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        raise


def detect_players_in_frame(
    frame_path: str,
    model,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5
) -> list:
    """
    Detect players in a single frame.

    Args:
        frame_path: Path to frame image
        model: YOLO model
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS

    Returns:
        List of detections with bboxes and foot positions
    """
    # Run detection (class 0 = person in COCO)
    results = model(
        frame_path,
        classes=[0],  # person class
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )

    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get bbox coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()

            # Calculate foot position (bottom center of bbox)
            foot_x = (x1 + x2) / 2
            foot_y = y2  # Bottom of box

            # Calculate bbox dimensions
            width = x2 - x1
            height = y2 - y1

            # Filter out very small or very large detections (likely not players)
            if width < 10 or height < 20:
                continue
            if width > 200 or height > 400:
                continue

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'bbox_width': width,
                'bbox_height': height,
                'confidence': conf,
                'foot_position': [foot_x, foot_y],
                'center_position': [(x1 + x2) / 2, (y1 + y2) / 2]
            })

    return detections


def detect_all_frames(
    frames_df: pd.DataFrame,
    output_file: str,
    model_name: str = 'yolov8x.pt',
    conf_threshold: float = 0.3
) -> list:
    """
    Detect players in all frames.

    Args:
        frames_df: DataFrame with frame paths
        output_file: Output JSON file
        model_name: YOLO model name
        conf_threshold: Confidence threshold

    Returns:
        List of detection results
    """

    # Load model
    logger.info(f"Loading YOLO model: {model_name}")
    model = load_yolo_model(model_name)

    results = []

    for idx, row in tqdm(frames_df.iterrows(), total=len(frames_df), desc="Detecting players"):
        frame_path = row['frame_path']

        if not Path(frame_path).exists():
            logger.warning(f"Frame not found: {frame_path}")
            continue

        detections = detect_players_in_frame(
            frame_path,
            model,
            conf_threshold
        )

        result = {
            'frame_idx': idx,
            'frame_path': frame_path,
            'corner_id': row['corner_id'],
            'offset_ms': row['offset_ms'],
            'num_detections': len(detections),
            'detections': detections
        }
        results.append(result)

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    total_detections = sum(r['num_detections'] for r in results)
    avg_detections = total_detections / len(results) if results else 0
    print(f"\nTotal detections: {total_detections}")
    print(f"Average per frame: {avg_detections:.1f}")

    # Distribution of detection counts
    detection_counts = [r['num_detections'] for r in results]
    print(f"Min/Max detections: {min(detection_counts)}/{max(detection_counts)}")

    return results


def visualize_detections(
    frame_path: str,
    detections: list,
    output_path: str
):
    """
    Visualize detections on a frame for debugging.

    Args:
        frame_path: Path to original frame
        detections: List of detections
        output_path: Path to save visualization
    """
    image = cv2.imread(frame_path)
    if image is None:
        return

    for det in detections:
        bbox = det['bbox']
        foot = det['foot_position']
        conf = det['confidence']

        # Draw bbox
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw foot position
        cv2.circle(image, (int(foot[0]), int(foot[1])), 5, (0, 0, 255), -1)

        # Draw confidence
        cv2.putText(image, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(output_path, image)


def main():
    parser = argparse.ArgumentParser(description='Detect players using YOLOv8')
    parser.add_argument('--frames-index',
                        default='/home/mseo/CornerTactics/soccersegcal-pipeline/data/frames_index.csv',
                        help='Path to frames index CSV')
    parser.add_argument('--output',
                        default='/home/mseo/CornerTactics/soccersegcal-pipeline/data/player_detections.json',
                        help='Output detections file')
    parser.add_argument('--model',
                        default='yolov8x.pt',
                        help='YOLO model to use')
    parser.add_argument('--conf-threshold',
                        type=float,
                        default=0.3,
                        help='Confidence threshold')
    parser.add_argument('--limit',
                        type=int,
                        default=None,
                        help='Limit number of frames to process')
    parser.add_argument('--visualize',
                        action='store_true',
                        help='Save visualizations')
    parser.add_argument('--vis-dir',
                        default='/home/mseo/CornerTactics/soccersegcal-pipeline/outputs/visualizations',
                        help='Directory for visualizations')
    args = parser.parse_args()

    # Load frames index
    frames_df = pd.read_csv(args.frames_index)
    print(f"Loaded {len(frames_df)} frames")

    if args.limit:
        frames_df = frames_df.head(args.limit)
        print(f"Limited to {len(frames_df)} frames")

    # Detect players
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = detect_all_frames(
        frames_df,
        args.output,
        args.model,
        args.conf_threshold
    )

    # Optionally visualize
    if args.visualize:
        Path(args.vis_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Saving visualizations...")
        for result in tqdm(results[:10], desc="Visualizing"):  # First 10
            frame_path = result['frame_path']
            vis_path = Path(args.vis_dir) / f"vis_{Path(frame_path).stem}.jpg"
            visualize_detections(frame_path, result['detections'], str(vis_path))

    print(f"\nDetections saved to {args.output}")


if __name__ == "__main__":
    main()
