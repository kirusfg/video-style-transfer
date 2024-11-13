import tensorflow as tf


def load_and_preprocess_image(path):
    """Load and preprocess an image for style transfer."""
    # print(path)
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    # print(type(img))
    # print(img.shape)
    img = tf.image.resize(img, (256, 256))
    # print(type(img))
    img = img / 127.5 - 1  # Normalize to [-1, 1]
    img.set_shape([256, 256, 3])  # Set the shape explicitly

    return img


def gram_matrix(input_tensor):
    """Calculate Gram matrix for style loss."""
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations
