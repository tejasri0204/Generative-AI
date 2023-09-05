import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(train_images, _), (test_images, _) = mnist.load_data()

# Normalize and reshape images
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Add Gaussian noise to the images
noise_factor = 0.5
train_noisy_images = train_images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_images.shape)
test_noisy_images = test_images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test_images.shape)

# Clip the pixel values to be between 0 and 1
train_noisy_images = np.clip(train_noisy_images, 0., 1.)
test_noisy_images = np.clip(test_noisy_images, 0., 1.)

# Define the Denoising Autoencoder architecture
autoencoder = tf.keras.Sequential([
    # Encoder
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),

    # Decoder
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(28 * 28, activation='sigmoid'),
    tf.keras.layers.Reshape((28, 28))
])

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the Denoising Autoencoder
autoencoder.fit(train_noisy_images, train_images, epochs=10, batch_size=128, shuffle=True, validation_data=(test_noisy_images, test_images))

# Use the trained autoencoder to denoise test images
denoised_images = autoencoder.predict(test_noisy_images)

# Visualize the original, noisy, and denoised images
num_images_to_show = 5
plt.figure(figsize=(12, 6))
for i in range(num_images_to_show):
    # Original Image
    plt.subplot(3, num_images_to_show, i + 1)
    plt.imshow(test_images[i], cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # Noisy Image
    plt.subplot(3, num_images_to_show, i + 1 + num_images_to_show)
    plt.imshow(test_noisy_images[i], cmap='gray')
    plt.title('Noisy')
    plt.axis('off')

    # Denoised Image
    plt.subplot(3, num_images_to_show, i + 1 + 2 * num_images_to_show)
    plt.imshow(denoised_images[i], cmap='gray')
    plt.title('Denoised')
    plt.axis('off')

plt.tight_layout()
plt.show()
