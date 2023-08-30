import sys
print(sys.version)

!python3.7 -m venv venv

!pip install numpy tensorflow keras matplotlib

!pip install --upgrade tensorflow

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K

# Load and preprocess MNIST dataset
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

# Build the VAE model
latent_dim = 2  # Dimension of the latent space

# Encoder
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)

# Mean and log variance outputs
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Reparameterization trick
z = layers.Lambda(sampling)([z_mean, z_log_var])

# Define the encoder model
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

# Define the decoder model
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

# Define the VAE model
vae_outputs = decoder(encoder(encoder_inputs)[2])
vae = keras.Model(encoder_inputs, vae_outputs, name="vae")

# Define the loss function for VAE
def vae_loss(x, x_decoded_mean):
    reconstruction_loss = keras.losses.binary_crossentropy(x, x_decoded_mean) * 28 * 28
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return reconstruction_loss + kl_loss

vae.compile(optimizer='adam', loss=vae_loss)

# Train the VAE model
vae.fit(x_train, x_train, epochs=10, batch_size=128, validation_data=(x_test, x_test))

# Generate new digits
num_digits = 10
latent_samples = np.random.normal(size=(num_digits, latent_dim))
generated_images = decoder.predict(latent_samples)

# Visualize the generated digits
plt.figure(figsize=(10, 2))
for i in range(num_digits):
    plt.subplot(1, num_digits, i + 1)
    plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()