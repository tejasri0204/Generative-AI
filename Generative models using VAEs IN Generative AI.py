import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Define the VAE architecture
latent_dim = 2  # Size of the latent space

# Encoder architecture
input_layer = Input(shape=(784,))
encoder_hidden = Dense(256, activation='relu')(input_layer)
z_mean = Dense(latent_dim)(encoder_hidden)
z_log_var = Dense(latent_dim)(encoder_hidden)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Decoder architecture
decoder_hidden = Dense(256, activation='relu')(z)
output_layer = Dense(784, activation='sigmoid')(decoder_hidden)

# Build the VAE model
vae = Model(input_layer, output_layer)

# Define the loss function with reconstruction loss and KL divergence term
reconstruction_loss = tf.keras.losses.binary_crossentropy(input_layer, output_layer)
reconstruction_loss = tf.reduce_mean(reconstruction_loss)
kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
vae_loss = reconstruction_loss + kl_loss

vae.add_loss(vae_loss)

# Compile the model
vae.compile(optimizer='adam')

# Train the VAE
vae.fit(x_train, epochs=50, batch_size=128, validation_data=(x_test, None))

# Define a decoder model for generating new images
decoder_input = Input(shape=(latent_dim,))
decoder_hidden = Dense(256, activation='relu')(decoder_input)
decoder_output = Dense(784, activation='sigmoid')(decoder_hidden)
decoder = Model(decoder_input, decoder_output)

# Generate new digits using the trained VAE
num_samples = 10
random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
generated_images = decoder.predict(random_latent_vectors)

# Display the generated images
plt.figure(figsize=(10, 2))
for i in range(num_samples):
    ax = plt.subplot(1, num_samples, i + 1)
    plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.show()