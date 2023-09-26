import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load MNIST dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Preprocess and normalize images
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

# GAN parameters
latent_dim = 100
generator_input_shape = (latent_dim,)

# Build the generator
generator = models.Sequential([
    layers.Dense(128, input_shape=generator_input_shape),
    layers.LeakyReLU(alpha=0.2),
    layers.Dense(784, activation='tanh'),
    layers.Reshape((28, 28, 1))
])

# Build the discriminator
discriminator = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128),
    layers.LeakyReLU(alpha=0.2),
    layers.Dense(1, activation='sigmoid')
])

# Compile the discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Combine generator and discriminator to form GAN
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = models.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training loop
epochs = 10000
batch_size = 128

for epoch in range(epochs):
    # Train discriminator
    real_images = train_images[np.random.randint(0, train_images.shape[0], batch_size)]
    fake_images = generator.predict(np.random.randn(batch_size, latent_dim))
    discriminator_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
    discriminator_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
    discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

    # Train generator
    noise = np.random.randn(batch_size, latent_dim)
    generator_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {discriminator_loss[0]}, G Loss: {generator_loss}")

        # Generate and save generated images
        generated_images = generator.predict(np.random.randn(25, latent_dim))
        fig, axes = plt.subplots(5, 5, figsize=(5, 5))
        for i, ax in enumerate(axes.flat):
            ax.imshow(generated_images[i].reshape(28, 28), cmap='gray')
            ax.axis('off')
        plt.savefig(f'gan_generated_image_epoch_{epoch}.png')
        plt.close()

