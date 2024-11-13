import cv2
import numpy as np
import os
from typing import List, Tuple


def extract_frames(video_path: str, output_dir: str) -> List[str]:
    """
    Extract frames from a video file and save them to output_dir.
    Returns list of saved frame paths.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_paths = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_count += 1

    cap.release()
    return frame_paths


def frames_to_video(frame_paths: List[str], output_path: str, fps: float = 30.0, frame_size: Tuple[int, int] = None) -> None:
    """
    Convert a sequence of frames back to video using H.264 codec.
    """
    if not frame_paths:
        raise ValueError("No frames provided")

    # Read first frame to get dimensions if not specified
    if frame_size is None:
        first_frame = cv2.imread(frame_paths[0])
        frame_size = (first_frame.shape[1], first_frame.shape[0])

    # Use H.264 codec
    if output_path.endswith(".mp4"):
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 codec
    else:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Default fallback

    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    if not out.isOpened():
        # Fallback to MP4V if H.264 is not available
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)

    out.release()

    # Optional: Use ffmpeg to ensure compatibility if available
    try:
        import subprocess

        temp_output = output_path + ".temp.mp4"
        os.rename(output_path, temp_output)
        subprocess.run(["ffmpeg", "-i", temp_output, "-c:v", "libx264", "-preset", "medium", "-crf", "23", output_path])
        os.remove(temp_output)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("FFmpeg not available, using original output")


def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Preprocess frame for model input.
    """
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize
    frame = cv2.resize(frame, target_size)
    # Convert to float and normalize to [-1, 1]
    frame = frame.astype(np.float32) / 127.5 - 1
    return frame


def postprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Convert model output back to uint8 format for video writing.
    """
    # Denormalize from [-1, 1] to [0, 255]
    frame = (frame + 1) * 127.5
    # Clip values
    frame = np.clip(frame, 0, 255)
    # Convert to uint8
    frame = frame.astype(np.uint8)
    # Convert RGB to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame
