import cv2
import numpy as np
import tensorflow as tf

def compute_optical_flow(frame1, frame2):
    """Compute optical flow between two frames using Farneback algorithm"""
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, 
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow

def warp_with_flow(frame, flow):
    """Warp frame according to optical flow"""
    h, w = flow.shape[:2]
    flow_map = tf.stack([
        tf.cast(tf.range(h), tf.float32)[None, :, None] + flow[:, :, 1],
        tf.cast(tf.range(w), tf.float32)[:, None, None] + flow[:, :, 0]
    ], axis=-1)
    
    return tf.keras.backend.map_fn(
        lambda x: tf.gather_nd(frame, tf.cast(x, tf.int32)),
        flow_map
    )