import numpy as np
import cv2
import tensorflow as tf

def compute_gradcam_plus_plus(model, img_array, layer_name='conv2d'):
    """
    Computes GradCAM++ for a given model and an input image array (1, H, W, C).
    layer_name is the name of the last convolutional layer in your model.
    Returns a heatmap (2D array) normalized between [0,1].
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                conv_outputs, predictions = grad_model(img_array)
                pred_index = tf.argmax(predictions[0])
                loss = predictions[:, pred_index]
            grads = tape3.gradient(loss, conv_outputs)
        grads2 = tape2.gradient(grads, conv_outputs)
    grads3 = tape1.gradient(grads2, conv_outputs)
    
    # Remove batch dimension
    conv_outputs = conv_outputs[0]
    grads = grads[0]
    grads2 = grads2[0]
    grads3 = grads3[0]
    
    # Compute alpha (GradCAM++ weighting)
    numerator = grads2
    denominator = 2.0 * grads2 + tf.multiply(conv_outputs, grads3)
    denominator = tf.where(denominator != 0.0, denominator, tf.ones_like(denominator))
    alpha = numerator / denominator
    alpha = tf.where(grads > 0, alpha, 0)
    
    weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(0, 1))
    activation_map = tf.reduce_sum(tf.multiply(conv_outputs, weights), axis=-1)
    activation_map = tf.nn.relu(activation_map)
    
    # Normalize heatmap to [0,1]
    heatmap = activation_map / tf.reduce_max(activation_map)
    return heatmap.numpy()

def overlay_gradcam(heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays the heatmap on top of the original grayscale image.
    - heatmap: 2D array with values between 0 and 1.
    - original_img: grayscale or color image (H, W) or (H, W, 3).
    - alpha: transparency factor for the overlay.
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img
