pip install tensorflow

pip install numpy

pip install matplotlib

# Import necessary libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load and preprocess the MNIST dataset
(train_images, _), (test_images, _) = mnist.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Define the VAE architecture
latent_dim = 2  # Dimensionality of the latent space

# Encoder network
encoder_inputs = tf.keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation='relu')(x)  # Latent mean
z_mean = layers.Dense(latent_dim)(x)  # Mean of the latent space
x = layers.Dense(16, activation='relu')(x)  # Latent log variance
z_log_var = layers.Dense(latent_dim)(x)  # Log variance of the latent space

# Reparameterization trick to sample from the latent space
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim), mean=0.0, stddev=1.0)
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

# Decoder network
decoder_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation='relu')(decoder_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)

# Define the VAE model
encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
decoder = models.Model(decoder_inputs, decoder_outputs, name='decoder')

# The VAE combines the encoder and decoder
vae_inputs = encoder_inputs
vae_outputs = decoder(z)
vae = models.Model(vae_inputs, vae_outputs, name='vae')

# Define the loss function
def vae_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, 28, 28, 1))
    y_pred = tf.reshape(y_pred, shape=(-1, 28, 28, 1))
    
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return reconstruction_loss + kl_loss

# Compile the VAE model with the modified loss function
vae.compile(optimizer='adam', loss=vae_loss)

# Train the VAE
epochs = 5
batch_size = 128

# Create TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).
shuffle(len(train_images)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)

# Training loop
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

for epoch in range(epochs):
    total_loss = 0
    for batch in train_dataset:
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(batch)
            reconstructed_images = decoder(z)
            loss = vae_loss(batch, reconstructed_images)

        gradients = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        total_loss += loss

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_dataset)}")

# Generate and visualize some images
n = 15  # Number of images to generate
figure = np.zeros((28 * n, 28 * n))
grid_x = np.linspace(-2, 2, n)
grid_y = np.linspace(-2, 2, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(28, 28)
        figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()

