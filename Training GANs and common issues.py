import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Generate synthetic Gaussian data
def generate_real_samples(n_samples):
    x = np.random.normal(0.5, 0.2, n_samples)
    y = x * 2 + np.random.normal(0, 0.1, n_samples)
    return np.vstack((x, y)).T

# Define the generator network
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(16, activation='relu', input_dim=10),
        layers.Dense(2, activation='linear')
    ])
    return model

# Define the discriminator network
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Dense(16, activation='relu', input_dim=2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Build the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    return model

# Define loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy()

# Initialize networks and GAN
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Define optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop
batch_size = 64
n_epochs = 10000
for epoch in range(n_epochs):
    real_samples = generate_real_samples(batch_size)
    noise = np.random.normal(0, 1, (batch_size, 10))
    generated_samples = generator.predict(noise)

    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    # Train the discriminator
    with tf.GradientTape() as tape:
        real_loss = cross_entropy(real_labels, discriminator(real_samples))
        fake_loss = cross_entropy(fake_labels, discriminator(generated_samples))
        discriminator_loss = real_loss + fake_loss
    gradients = tape.gradient(discriminator_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, 10))
    with tf.GradientTape() as tape:
        generated_samples = generator(noise)
        generator_loss = cross_entropy(real_labels, discriminator(generated_samples))
    gradients = tape.gradient(generator_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

    if epoch % 500 == 0:
        print(f"Epoch {epoch}/{n_epochs}, D Loss: {discriminator_loss.numpy()}, G Loss: {generator_loss.numpy()}")

# Generate and visualize generated samples
generated_samples = generator.predict(np.random.normal(0, 1, (100, 10)))
import matplotlib.pyplot as plt

plt.scatter(generated_samples[:, 0], generated_samples[:, 1], color='blue', label='Generated')
real_data = generate_real_samples(100)
plt.scatter(real_data[:, 0], real_data[:, 1], color='red', label='Real')
plt.legend()
plt.title('Generated vs Real Data')
plt.show()
