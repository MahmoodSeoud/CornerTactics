# Corner Frame Extraction Feature

## Goal
Extract single frames at the exact timestamp when each corner kick is taken from SoccerNet broadcast videos.

## Understanding the Feature
- Phase 1 of two-phase approach for corner analysis
- Extract ONE frame per corner at exact moment of kick
- Input: Labels-v3.json (corner timestamps) + broadcast videos
- Output: Single JPEG frames for each corner + metadata CSV

## Key Requirements
- Extract frame at exact timestamp from Labels-v3.json
- Handle both halves (1_720p.mkv, 2_720p.mkv)
- Parse gameTime format (e.g., "15:23")
- Save frames with descriptive filenames
- Generate metadata CSV with frame paths and corner info

## Technical Approach
- Use ffmpeg for frame extraction at specific timestamps
- Convert gameTime to seconds for ffmpeg -ss parameter
- Handle both first and second half videos
- Error handling for missing videos or invalid timestamps

## Expected Output
- ~4,836 corner frames (1 per corner)
- Frame filenames: `{game}_{half}_{timestamp}.jpg`
- Metadata CSV: frame_path, game, half, timestamp, team, etc.
- Total storage: ~1GB (vs 300GB for full videos)