import os
import cv2
from pathlib import Path
from typing import List


def list_videos(video_dir: str) -> List[str]:
    """List all video files in directory."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    return [str(f) for f in Path(video_dir).iterdir() if f.suffix.lower() in video_extensions]


def extract_frames(video_path: str, output_dir: str, frame_interval: int = 2, max_frames: int = 150) -> List[str]:
    """
    Extract frames from video.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_interval: Extract every nth frame
        max_frames: Maximum number of frames to extract per video
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get video name for frame prefixing
    video_name = Path(video_path).stem

    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    saved_paths = []

    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract every nth frame
        if frame_count % frame_interval == 0:
            # Generate frame filename
            frame_path = os.path.join(output_dir, f"{video_name}_frame_{saved_count:04d}.jpg")

            # Save frame
            cv2.imwrite(frame_path, frame)
            saved_paths.append(frame_path)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_paths


def prepare_dataset(
    video_dir: str = "data/videos", frames_dir: str = "data/frames", frame_interval: int = 2, max_frames_per_video: int = 150
) -> None:
    """
    Prepare dataset by extracting frames from all videos.
    """
    # Ensure directories exist
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    # List all videos
    videos = list_videos(video_dir)
    if not videos:
        print(f"No videos found in {video_dir}")
        print("Please add some videos first!")
        return

    print(f"Found {len(videos)} videos")
    total_frames = 0

    # Process each video
    for video_path in videos:
        print(f"\nProcessing {Path(video_path).name}...")
        frames = extract_frames(video_path, frames_dir, frame_interval, max_frames_per_video)
        total_frames += len(frames)
        print(f"Extracted {len(frames)} frames")

    print(f"\nTotal frames extracted: {total_frames}")
    print(f"Frames saved in: {frames_dir}")


if __name__ == "__main__":
    # Example usage
    prepare_dataset()
