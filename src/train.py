import os

import matplotlib.pyplot as plt
import tensorflow as tf
from model import StyleTransferModel
from utils.style import load_and_preprocess_image


def train_basic(content_dir: str, style_image_path: str, save_dir: str, epochs: int = 10):
    """
    Basic training without temporal consistency.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "epochs"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)

    # Load style image
    style_image = load_and_preprocess_image(style_image_path)
    style_image = tf.expand_dims(style_image, 0)

    # Create model
    model = StyleTransferModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), style_weight=1e-5, content_weight=1e8)

    best_style_loss = float("inf")
    best_content_loss = float("inf")
    style_loss_history = []
    content_loss_history = []

    # Create dataset from content images
    content_paths = sorted([os.path.join(content_dir, f) for f in os.listdir(content_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    dataset = tf.data.Dataset.from_tensor_slices(content_paths)
    dataset = dataset.map(lambda x: tf.py_function(func=load_and_preprocess_image, inp=[x], Tout=tf.float32))
    dataset = dataset.batch(8)

    sample_content = next(iter(dataset))
    tf.keras.preprocessing.image.save_img(
        os.path.join(save_dir, "samples", "sample_input.png"),
        (sample_content[0] + 1) * 127.5,  # Denormalize from [-1,1] to [0,255]
    )
    tf.keras.preprocessing.image.save_img(os.path.join(save_dir, "samples", "style_reference.png"), (style_image[0] + 1) * 127.5)

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_style_losses = []
        epoch_content_losses = []

        for i, content_batch in enumerate(dataset):
            loss = model.train_step((content_batch, style_image))
            epoch_style_losses.append(loss["style_loss"])
            epoch_content_losses.append(loss["content_loss"])

            if i % 10 == 0:
                print(
                    f"Batch {i}, "
                    f"Total Loss: {loss['loss']:.4f}, "
                    f"Style Loss: {loss['style_loss']:.4f} (weighted: {loss['style_loss'] * model.style_weight:.4f}), "
                    f"Content Loss: {loss['content_loss']:.4f} (weighted: {loss['content_loss'] * model.content_weight:.4f})"
                )

                # Save sample outputs every 50 batches
                if i % 50 == 0:
                    # Save original content for comparison
                    tf.keras.preprocessing.image.save_img(
                        os.path.join(save_dir, "samples", f"epoch_{epoch+1}_batch_{i}_content.png"), (content_batch[0] + 1) * 127.5
                    )
                    # Save stylized output
                    styled_output = model(content_batch)
                    tf.keras.preprocessing.image.save_img(
                        os.path.join(save_dir, "samples", f"epoch_{epoch+1}_batch_{i}_stylized.png"), (styled_output[0] + 1) * 127.5
                    )

        style_loss = sum(epoch_style_losses) / len(epoch_style_losses)
        style_loss_history.append(style_loss)

        content_loss = sum(epoch_content_losses) / len(epoch_content_losses)
        content_loss_history.append(content_loss)

        print(f"Epoch {epoch+1} Average Style Loss: {style_loss:.4f}, Average Content Loss: {content_loss:.4f}")

        # Save if best model
        if content_loss < best_content_loss and style_loss < best_style_loss:
            best_style_loss = style_loss
            best_content_loss = content_loss
            model.save_weights(os.path.join(save_dir, "weights", "style_transfer_basic_best.weights.h5"))

        # Plot loss history
        plt.figure(figsize=(10, 5))
        plt.plot(style_loss_history)
        plt.plot(content_loss_history)
        plt.title("Training Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Style Loss", "Content Loss"])
        plt.savefig(os.path.join(save_dir, "loss_history.png"))
        plt.close()

    dummy_input = tf.zeros((1, 256, 256, 3))
    _ = model(dummy_input)  # This ensures the model is built

    # Save weights
    os.makedirs(os.path.join(save_dir, "weights"), exist_ok=True)
    model.save_weights(os.path.join(save_dir, "weights", "style_transfer_basic.weights.h5"))

    return model


def train_temporal(content_dir: str, style_image_path: str, save_dir: str, epochs: int = 10):
    """
    Training with temporal consistency loss.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "epochs"), exist_ok=True)

    # Load style image
    style_image = load_and_preprocess_image(style_image_path)
    style_image = tf.expand_dims(style_image, 0)

    # Create model
    model = StyleTransferModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), style_weight=1e-2, content_weight=1e4, temporal_weight=2.0)

    # Create dataset of consecutive frame pairs
    content_paths = sorted([os.path.join(content_dir, f) for f in os.listdir(content_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    # Create pairs of consecutive frames
    frame_pairs = [(content_paths[i], content_paths[i + 1]) for i in range(len(content_paths) - 1)]

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for i, (frame1_path, frame2_path) in enumerate(frame_pairs):
            frame1 = load_and_preprocess_image(frame1_path)
            frame2 = load_and_preprocess_image(frame2_path)
            frame1 = tf.expand_dims(frame1, 0)
            frame2 = tf.expand_dims(frame2, 0)

            loss = model.train_step((frame1, style_image, frame2))
            if i % 10 == 0:
                print(f"Pair {i}, Loss: {loss['loss']:.4f}")

    # model.save(os.path.join(save_dir, "style_transfer_temporal.keras"))
    dummy_input = tf.zeros((1, 256, 256, 3))
    _ = model(dummy_input)  # This ensures the model is built

    # Save weights
    os.makedirs(os.path.join(save_dir, "weights"), exist_ok=True)
    model.save_weights(os.path.join(save_dir, "weights", "style_transfer_temporal.weights.h5"))

    return model


if __name__ == "__main__":
    # Example usage
    content_dir = "data/frames"  # Directory with training frames
    style_image_path = "data/style/yellow-neon.png"  # Path to style image
    save_dir = "models/style_transfer"

    # Train basic model
    print("Training basic model...")
    model_basic = train_basic(content_dir, style_image_path, os.path.join(save_dir, "basic"))

    # Train temporal model
    print("\nTraining model with temporal consistency...")
    model_temporal = train_temporal(content_dir, style_image_path, os.path.join(save_dir, "temporal"))
