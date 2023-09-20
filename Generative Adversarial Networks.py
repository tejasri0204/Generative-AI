import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# Hyperparameters
random_dim = 100
epochs = 10000
batch_size = 128

# Load MNIST dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 127.5 - 1.0
train_images = np.expand_dims(train_images, axis=-1)

# Build generator
generator = tf.keras.Sequential([
    layers.Input(shape=(random_dim,)),
    layers.Dense(7 * 7 * 128),
    layers.Reshape((7, 7, 128)),
    layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu'),
    layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='tanh')
])

# Build discriminator
discriminator = tf.keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'),
    layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu'),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Combined model (generator -> discriminator)
discriminator.trainable = False
combined_model = tf.keras.Sequential([generator, discriminator])
combined_model.compile(loss='binary_crossentropy', optimizer='adam')

# Training loop
for epoch in range(epochs):
    idx = np.random.randint(0, train_images.shape[0], batch_size)
    real_images = train_images[idx]
    
    noise = np.random.normal(0, 1, (batch_size, random_dim))
    generated_images = generator.predict(noise)
    
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    noise = np.random.normal(0, 1, (batch_size, random_dim))
    g_loss = combined_model.train_on_batch(noise, real_labels)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
        # Generate and save generated images
        num_examples_to_generate = 16
        generated_images = generator.predict(np.random.normal(0, 1, (num_examples_to_generate, random_dim)))
        generated_images = 0.5 * generated_images + 0.5
        fig, axs = plt.subplots(4, 4)
        count = 0
        for i in range(4):
            for j in range(4):
                axs[i, j].imshow(generated_images[count, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                count += 1
        plt.show()

