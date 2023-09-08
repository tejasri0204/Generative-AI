pip install tensorflow numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images to a 784-dimensional vector
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Define input layer
input_layer = Input(shape=(784,))

# Encoder
encoder = Dense(256, activation='relu')(input_layer)
z_mean = Dense(2)(encoder)
z_log_var = Dense(2)(encoder)

# Sampling layer using the reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], 2), mean=0.0, stddev=1.0)
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_input = Input(shape=(2,))
decoder = Dense(256, activation='relu')(decoder_input)
output_layer = Dense(784, activation='sigmoid')(decoder)

# Define VAE model
encoder_model = Model(input_layer, [z_mean, z_log_var, z], name='encoder')
decoder_model = Model(decoder_input, output_layer, name='decoder')
vae_output = decoder_model(encoder_model(input_layer)[2])
vae = Model(input_layer, vae_output, name='vae')

# Define VAE loss function
reconstruction_loss = tf.keras.losses.binary_crossentropy(input_layer, vae_output)
reconstruction_loss *= 784  # To account for the dimensionality
kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
kl_loss = tf.reduce_sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

# Compile the VAE model
vae.compile(optimizer='adam')

# Train the VAE
vae.fit(x_train, epochs=50, batch_size=128, validation_data=(x_test, None))

# Generate images using the decoder
n = 15  # Number of images to generate
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# Define the grid of coordinates in the latent space
grid_x = np.linspace(-2, 2, n)
grid_y = np.linspace(-2, 2, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder_model.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys')
plt.show()