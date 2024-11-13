import tensorflow as tf
import tensorflow_hub as hub
from utils.optical_flow import compute_optical_flow, warp_with_flow

# VGG model for feature extraction
vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
style_layers = ["block1_conv1", "block2_conv1", "block3_conv1"]
content_layers = ["block4_conv2", "block5_conv2"]
vgg_style_content = tf.keras.Model(vgg.input, [vgg.get_layer(name).output for name in style_layers + content_layers])


def temporal_consistency_loss(styled_frame1, styled_frame2, content_frame1, content_frame2):
    """Calculate temporal consistency loss between consecutive frames."""
    # Compute optical flow between content frames
    flow = compute_optical_flow(content_frame1, content_frame2)

    # Warp the second stylized frame to match the first frame
    warped_styled_frame2 = warp_with_flow(styled_frame2, flow)

    # Calculate temporal loss as the difference between the first stylized frame
    # and the warped second stylized frame
    temporal_loss = tf.reduce_mean(tf.abs(styled_frame1 - warped_styled_frame2))

    return temporal_loss


def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


class StyleContentModel(tf.keras.models.Model):
    def __init__(self):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_style_content
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed)
        style_outputs, content_outputs = (outputs[: self.num_style_layers], outputs[self.num_style_layers :])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        return {"content": content_outputs, "style": style_outputs}


# Base encoder using MobileNetV2
base_model = tf.keras.applications.MobileNetV2(input_shape=(256, 256, 3), include_top=False, weights="imagenet")


def build_decoder():
    inputs = tf.keras.layers.Input(shape=(8, 8, 1280))
    x = tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu", name="decoder_conv2d_transpose_1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="decoder_batch_norm_1")(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu", name="decoder_conv2d_transpose_2")(x)
    x = tf.keras.layers.BatchNormalization(name="decoder_batch_norm_2")(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu", name="decoder_conv2d_transpose_3")(x)
    x = tf.keras.layers.BatchNormalization(name="decoder_batch_norm_3")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu", name="decoder_conv2d_transpose_4")(x)
    x = tf.keras.layers.BatchNormalization(name="decoder_batch_norm_4")(x)
    x = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu", name="decoder_conv2d_transpose_5")(x)
    outputs = tf.keras.layers.Conv2D(3, 3, activation="tanh", padding="same", name="decoder_conv2d_final")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="decoder")


class StyleTransferModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(StyleTransferModel, self).__init__(**kwargs)
        self.encoder = base_model
        self.decoder = build_decoder()
        self.style_content_model = StyleContentModel()

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        model = cls(**config)
        model.encoder = base_model
        model.decoder = build_decoder()
        model.style_content_model = StyleContentModel()
        return model

    def compile(self, optimizer, style_weight=1e-2, content_weight=1e4, temporal_weight=2.0):
        super().compile()
        self.optimizer = optimizer
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.temporal_weight = temporal_weight

    def call(self, inputs):
        features = self.encoder(inputs)
        return self.decoder(features)

    def train_step(self, data):
        if len(data) == 2:
            # Single frame training
            content_image, style_image = data
            frame2 = None
        else:
            # Sequential frames training
            content_image, style_image, frame2 = data

        with tf.GradientTape() as tape:
            # Generate stylized image
            styled_image = self(content_image)

            # Calculate features - separate for style and content targets
            style_targets = self.style_content_model(style_image)
            content_targets = self.style_content_model(content_image)  # Get content features from content image
            outputs = self.style_content_model(styled_image)

            # Style loss - compare gram matrices with style image
            style_loss = tf.reduce_mean(
                [
                    tf.reduce_mean((style_target - style_output) ** 2)
                    for style_target, style_output in zip(style_targets["style"], outputs["style"])
                ]
            )

            # Content loss - compare features with content image
            content_loss = tf.reduce_mean(
                [
                    tf.reduce_mean((content_target - content_output) ** 2)
                    for content_target, content_output in zip(content_targets["content"], outputs["content"])
                ]
            )

            total_loss = self.style_weight * style_loss + self.content_weight * content_loss

            # Add temporal consistency loss if we have sequential frames
            if frame2 is not None:
                styled_frame2 = self(frame2)
                temporal_loss = temporal_consistency_loss(styled_image, styled_frame2, content_image, frame2)
                total_loss += self.temporal_weight * temporal_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        if frame2 is not None:
            return {"loss": total_loss, "style_loss": style_loss, "content_loss": content_loss, "temporal_loss": temporal_loss}
        else:
            return {"loss": total_loss, "style_loss": style_loss, "content_loss": content_loss}
