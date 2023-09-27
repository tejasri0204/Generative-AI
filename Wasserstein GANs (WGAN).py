import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(128, input_dim=latent_dim))
    model.add(layers. LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1))
    return model

def build_discriminator():
    model = models.Sequential()
    model.add(layers.Dense(128, input_dim=1))
    model.add(layers. LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1))
    return model

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def train_wgan(generator, discriminator, iterations, batch_size, latent_dim, critic_iters):
    for _ in range(iterations):
        for _ in range(critic_iters):
            noise = np.random.randn(batch_size, latent_dim)
            generated_data = generator.predict(noise)

            real_data = np.random.normal(0, 1, (batch_size, 1))

            d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(generated_data, -np.ones((batch_size, 1)))

            for layer in discriminator.layers:
                weights = [np.clip(w, -0.01, 0.01) for w in layer.get_weights()]
                layer.set_weights(weights)

        noise = np.random.randn(batch_size, latent_dim)
        g_loss = discriminator.train_on_batch(generator.predict(noise), np.ones((batch_size, 1)))

        print(f"Iteration: {_+1}, D Loss Real: {d_loss_real}, D Loss Fake: {d_loss_fake}, G Loss: {g_loss}")

        if (_ + 1) % 100 == 0:
            generate_and_plot_samples(generator, latent_dim, batch_size, _ + 1)
def generate_and_plot_samples(generator, latent_dim, batch_size, iteration):
    noise = np.random.randn(1000, latent_dim)
    generated_data = generator.predict(noise)
    plt.figure()
    plt.hist(generated_data, bins=50, density=True)
    plt.title(f'Generated Samples - Iteration {iteration}')
    plt.show()

latent_dim = 10
generator = build_generator(latent_dim)
discriminator = build_discriminator()

generator.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.RMSprop(lr=0.00005))
discriminator.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.RMSprop(lr=0.00005))

iterations = 10000
batch_size = 64
critic_iters = 5

train_wgan(generator, discriminator, iterations, batch_size, latent_dim, critic_iters)
