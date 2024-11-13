import os
from utils.video import extract_frames, frames_to_video, preprocess_frame, postprocess_frame
import cv2
import numpy as np
import tensorflow as tf
from model import StyleTransferModel


def create_demo_effect(frame):
    """
    Temporary placeholder for style transfer - just applies a simple effect.
    We'll replace this with actual style transfer later.
    """
    # Apply some basic effect (grayscale + colormap) as a placeholder
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    return colored


def process_video_effect(input_video: str, output_video: str):
    # Create temporary directory for frames
    temp_dir = "temp_frames"
    styled_dir = "styled_frames"

    for directory in [temp_dir, styled_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    try:
        # Extract frames
        print("Extracting frames...")
        frame_paths = extract_frames(input_video, temp_dir)

        # Process each frame
        print("Processing frames...")
        styled_frame_paths = []
        for i, frame_path in enumerate(frame_paths):
            # Read frame
            frame = cv2.imread(frame_path)

            # Apply effect (placeholder for style transfer)
            styled_frame = create_demo_effect(frame)

            # Save styled frame
            styled_path = os.path.join(styled_dir, f"styled_{i:06d}.jpg")
            cv2.imwrite(styled_path, styled_frame)
            styled_frame_paths.append(styled_path)

            if i % 10 == 0:
                print(f"Processed {i}/{len(frame_paths)} frames")

        # Create video from styled frames
        print("Creating output video...")
        frames_to_video(styled_frame_paths, output_video)

    finally:
        # Cleanup
        print("Cleaning up...")
        for directory in [temp_dir, styled_dir]:
            for file in os.listdir(directory):
                os.remove(os.path.join(directory, file))
            os.rmdir(directory)


def process_video_model(
    input_video: str, output_video: str, weights_path: str = "models/style_transfer/weights/style_transfer_basic.weights.h5"
):
    def load_style_transfer_model(model_path):
        """Custom load function for StyleTransferModel"""
        model = StyleTransferModel()

        # Build the model with a dummy input
        dummy_input = tf.zeros((1, 256, 256, 3))
        _ = model(dummy_input)

        # Load weights
        model.load_weights(model_path)
        return model

    # Load the model
    model = load_style_transfer_model(weights_path)

    # model.compile()

    # Create temporary directories
    temp_dir = "temp_frames"
    styled_dir = "styled_frames"

    for directory in [temp_dir, styled_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    try:
        # Extract frames
        print("Extracting frames...")
        frame_paths = extract_frames(input_video, temp_dir)

        # Process each frame
        print("Processing frames...")
        styled_frame_paths = []
        for i, frame_path in enumerate(frame_paths):
            # Read and preprocess frame
            frame = cv2.imread(frame_path)
            frame = preprocess_frame(frame)
            frame = np.expand_dims(frame, 0)

            # Apply style transfer
            styled_frame = model.predict(frame)
            styled_frame = np.squeeze(styled_frame, 0)
            styled_frame = postprocess_frame(styled_frame)

            # Save styled frame
            styled_path = os.path.join(styled_dir, f"styled_{i:06d}.jpg")
            cv2.imwrite(styled_path, styled_frame)
            styled_frame_paths.append(styled_path)

            if i % 10 == 0:
                print(f"Processed {i}/{len(frame_paths)} frames")

        # Create video from styled frames
        print("Creating output video...")
        frames_to_video(styled_frame_paths, output_video)

    finally:
        # Cleanup
        print("Cleaning up...")
        for directory in [temp_dir, styled_dir]:
            for file in os.listdir(directory):
                os.remove(os.path.join(directory, file))
            os.rmdir(directory)  # Example usage


if __name__ == "__main__":
    input_video = "data/input.mp4"
    output_video = "data/output.mp4"

    if not os.path.exists(input_video):
        print(f"Please provide an input video at {input_video}")
    else:
        process_video_effect(input_video, output_video)
        print("Processing complete!")
