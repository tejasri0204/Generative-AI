pip install tensorflow numpy matplotlib

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import urllib

# Load the pre-trained VGG19 model with ImageNet weights
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# Specify the layers for content and style representation
content_layers = ['block5_conv2']

# Style layer names for VGG19
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def vgg_layers(layer_names):
    """Create a VGG model that returns a list of intermediate output values."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

# Calculate gram matrix to extract style features
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)

# Define a function to load and preprocess images from URLs
def load_and_preprocess_image(image_url):
    response = urllib.request.urlopen(image_url)
    image = Image.open(response)
    image = np.array(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (256, 256))
    image = image[tf.newaxis, :]
    return image

# Sample content and style images from public URLs
content_image_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
style_image_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'

# Create a function to extract content and style features
def get_feature_representations(model, content_path, style_path):
    content_image = load_and_preprocess_image(content_path)
    style_image = load_and_preprocess_image(style_path)
    
    style_outputs = model(style_image)
    content_outputs = model(content_image)
    
    # Extract content features from the specified content layer
    content_features = content_outputs[num_style_layers:]
    
    # Extract style features from the specified style layers
    style_features = [gram_matrix(style_output) for style_output in style_outputs]
    
    return style_features, content_features

# Extract style and content features using VGG19
style_features, content_features = get_feature_representations(
    vgg_layers(style_layers + content_layers),
    content_image_url,
    style_image_url
)

# Create a function to compute the total loss
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    
    model_outputs = model(init_image)
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]
    
    style_score = 0
    content_score = 0
    
    # Accumulate style and content losses from all layers
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
    
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)
    
    style_score *= style_weight
    content_score *= content_weight
    
    total_loss = style_score + content_score
    
    return total_loss, style_score, content_score

def visualize_features(style_features, content_features, style_channel=None):
    for i, (style_feature, content_feature) in enumerate(zip(style_features, content_features)):
        plt.figure(figsize=(10, 2))
        
        # Visualize style feature
        if style_channel is not None:
            if style_channel >= style_feature.shape[-1]:
                print(f"Style channel {style_channel} is out of range for feature {i}.")
                continue
            plt.subplot(1, 2, 1)
            plt.imshow(style_feature[0, :, :, style_channel], cmap='viridis')
            plt.title(f'Style Feature (Channel {style_channel})')
            plt.axis('off')
        else:
            # Visualize as an RGB image if style features have 3 channels
            if style_feature.shape[-1] == 3:
                plt.subplot(1, 2, 1)
                plt.imshow(style_feature[0, :, :, :])
                plt.title('Style Feature (RGB)')
                plt.axis('off')
        
        # Visualize content feature
        plt.subplot(1, 2, 2)
        plt.imshow(content_feature[0, :, :, :], cmap='gray')  # Assuming content feature is grayscale
        plt.title('Content Feature')
        plt.axis('off')
        
        plt.show()