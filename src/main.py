import os
import argparse
from demo import process_video_effect, process_video_model
from train import train_basic, train_temporal


def parse_args():
    parser = argparse.ArgumentParser(description="Style Transfer for Videos")

    # Main operation mode
    parser.add_argument("--demo", choices=["effect"], help="Run demo with simple effect")
    parser.add_argument("--train", action="store_true", help="Train a model")
    parser.add_argument("--test", action="store_true", help="Test/apply a trained model")

    # Model type
    parser.add_argument("--model", choices=["basic", "temporal"], help="Model type to train or test")

    # Input/output paths
    parser.add_argument("--input", help="Path to input video")
    parser.add_argument("--output", help="Path to output video")

    # Optional parameters
    parser.add_argument("--style", default="data/style/yellow-neon.png", help="Path to style image")
    parser.add_argument("--frames-dir", default="data/frames", help="Directory for video frames")
    parser.add_argument("--models-dir", default="models/style_transfer", help="Directory for saving models")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create necessary directories
    os.makedirs("data/style", exist_ok=True)
    os.makedirs(args.frames_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    # Demo mode
    if args.demo == "effect":
        if not args.input or not args.output:
            print("Error: --input and --output required for demo mode")
            return
        print("\nProcessing video with simple effect...")
        process_video_effect(args.input, args.output)
        return

    # Training mode
    if args.train:
        if not args.model:
            print("Error: --model required for training mode")
            return

        if not os.path.exists(args.style):
            print(f"Error: Style image not found at {args.style}")
            return

        if args.model == "basic":
            print("\nTraining basic style transfer model...")
            model_basic = train_basic(
                args.frames_dir,
                args.style,
                os.path.join(args.models_dir, "basic"),
                epochs=args.epochs,
            )
        else:  # temporal
            print("\nTraining style transfer model with temporal consistency...")
            model_temporal = train_temporal(
                args.frames_dir,
                args.style,
                os.path.join(args.models_dir, "temporal", "style_transfer_temporal.keras"),
                epochs=args.epochs,
            )
        return

    # Testing mode
    if args.test:
        if not all([args.model, args.input, args.output]):
            print("Error: --model, --input, and --output required for test mode")
            return

        model_path = os.path.join(args.models_dir, args.model, "weights", f"style_transfer_{args.model}.weights.h5")
        # if not os.path.exists(model_path + ".index"):  # Check for weights file
        #     print(f"Error: Model weights not found at {model_path}")
        #     return

        print(f"\nProcessing video with {args.model} style transfer...")
        process_video_model(args.input, args.output, model_path)
        return


if __name__ == "__main__":
    main()
