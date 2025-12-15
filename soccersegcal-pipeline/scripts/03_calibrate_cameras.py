#!/usr/bin/env python3
"""
Step 3: Camera calibration using soccersegcal.

This script runs Spiideo's soccersegcal model on extracted frames to get
camera parameters (homography, intrinsics, extrinsics) for projecting
image coordinates to pitch coordinates.

Reference: https://github.com/Spiideo/soccersegcal
"""

import sys
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import logging

# Add soccersegcal to path
sys.path.insert(0, '/home/mseo/CornerTactics/soccersegcal-pipeline/soccersegcal')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_calibration_model(checkpoint_path: str, device: str = 'cuda'):
    """
    Load pretrained soccersegcal model.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Loaded model
    """
    try:
        from soccersegcal.model import SegCalModel
        model = SegCalModel.load_from_checkpoint(checkpoint_path, map_location=device)
        model = model.to(device)
        model.eval()
        return model
    except ImportError:
        logger.warning("soccersegcal not found, trying alternative import")
        # Try alternative import structure
        try:
            from model import SegCalModel
            model = SegCalModel.load_from_checkpoint(checkpoint_path, map_location=device)
            model = model.to(device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


def preprocess_image(image_path: str, target_size: tuple = (720, 1280)):
    """
    Preprocess image for soccersegcal.

    Args:
        image_path: Path to image
        target_size: Target (height, width)

    Returns:
        Preprocessed tensor and original image
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image_rgb.shape[:2]

    # Resize to model input size
    resized = cv2.resize(image_rgb, (target_size[1], target_size[0]))

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # Convert to tensor [C, H, W]
    tensor = torch.from_numpy(normalized).permute(2, 0, 1)

    return tensor.unsqueeze(0), {'original_size': original_size, 'image': image_rgb}


def calibrate_single_frame(
    image_path: str,
    model,
    device: str = 'cuda'
) -> dict:
    """
    Calibrate camera for a single frame.

    Args:
        image_path: Path to image
        model: Loaded soccersegcal model
        device: Device to use

    Returns:
        Dict with camera parameters or None if failed
    """
    # Preprocess image
    tensor, metadata = preprocess_image(image_path)
    if tensor is None:
        return {'success': False, 'error': 'Failed to load image'}

    tensor = tensor.to(device)

    try:
        with torch.no_grad():
            # Run model inference
            output = model(tensor)

        # Extract camera parameters from output
        # soccersegcal outputs homography and camera parameters
        if hasattr(output, 'homography'):
            homography = output.homography.cpu().numpy()
        elif isinstance(output, dict) and 'homography' in output:
            homography = output['homography'].cpu().numpy()
        elif isinstance(output, torch.Tensor):
            # Model outputs homography directly
            homography = output.cpu().numpy()
        else:
            # Try to extract from tuple/list output
            homography = output[0].cpu().numpy() if isinstance(output, (tuple, list)) else None

        if homography is None:
            return {'success': False, 'error': 'Could not extract homography'}

        # Ensure homography is 3x3
        if homography.ndim > 2:
            homography = homography.squeeze()
        if homography.shape != (3, 3):
            homography = homography.reshape(3, 3)

        return {
            'success': True,
            'homography': homography.tolist(),
            'original_size': metadata['original_size']
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}


def calibrate_with_fallback(
    image_path: str,
    model,
    device: str = 'cuda'
) -> dict:
    """
    Calibrate with fallback methods if main method fails.

    Tries:
    1. soccersegcal model
    2. Line detection + homography estimation
    """

    # Try soccersegcal first
    result = calibrate_single_frame(image_path, model, device)
    if result['success']:
        return result

    # Fallback: line detection based homography
    logger.info(f"Trying fallback calibration for {image_path}")

    try:
        image = cv2.imread(image_path)
        if image is None:
            return {'success': False, 'error': 'Failed to load image'}

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=100,
            minLineLength=100, maxLineGap=10
        )

        if lines is None or len(lines) < 4:
            return {'success': False, 'error': 'Not enough lines detected'}

        # For now, return failure and rely on frames where model works
        return {'success': False, 'error': 'Fallback not fully implemented'}

    except Exception as e:
        return {'success': False, 'error': f'Fallback failed: {str(e)}'}


def calibrate_all_frames(
    frames_df: pd.DataFrame,
    checkpoint_path: str,
    output_file: str,
    device: str = 'cuda',
    batch_size: int = 1
) -> list:
    """
    Calibrate all extracted frames.

    Args:
        frames_df: DataFrame with frame paths
        checkpoint_path: Path to model checkpoint
        output_file: Output JSON file
        device: Device to use
        batch_size: Batch size for inference

    Returns:
        List of calibration results
    """

    # Load model
    logger.info(f"Loading calibration model from {checkpoint_path}")
    try:
        model = load_calibration_model(checkpoint_path, device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Will use placeholder calibrations - you may need to install soccersegcal properly")
        model = None

    results = []

    for idx, row in tqdm(frames_df.iterrows(), total=len(frames_df), desc="Calibrating"):
        frame_path = row['frame_path']

        if model is not None:
            calib = calibrate_with_fallback(frame_path, model, device)
        else:
            # Placeholder for when model isn't available
            calib = {
                'success': False,
                'error': 'Model not loaded'
            }

        result = {
            'frame_idx': idx,
            'frame_path': frame_path,
            'corner_id': row['corner_id'],
            'offset_ms': row['offset_ms'],
            **calib
        }
        results.append(result)

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    print(f"\nCalibration success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description='Camera calibration using soccersegcal')
    parser.add_argument('--frames-index',
                        default='/home/mseo/CornerTactics/soccersegcal-pipeline/data/frames_index.csv',
                        help='Path to frames index CSV')
    parser.add_argument('--checkpoint',
                        default='/home/mseo/CornerTactics/soccersegcal-pipeline/models/soccersegcal_snapshot.ckpt',
                        help='Path to model checkpoint')
    parser.add_argument('--output',
                        default='/home/mseo/CornerTactics/soccersegcal-pipeline/data/camera_calibrations.json',
                        help='Output calibrations file')
    parser.add_argument('--device',
                        default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--limit',
                        type=int,
                        default=None,
                        help='Limit number of frames to process')
    args = parser.parse_args()

    # Load frames index
    frames_df = pd.read_csv(args.frames_index)
    print(f"Loaded {len(frames_df)} frames")

    if args.limit:
        frames_df = frames_df.head(args.limit)
        print(f"Limited to {len(frames_df)} frames")

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Calibrate
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = calibrate_all_frames(
        frames_df,
        args.checkpoint,
        args.output,
        args.device
    )

    print(f"\nCalibrations saved to {args.output}")


if __name__ == "__main__":
    main()
