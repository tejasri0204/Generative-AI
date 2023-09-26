import tensorflow as tf
print(tf.__version__)

!pip install --upgrade tensorflow

!pip install --upgrade keras

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

# Load and preprocess the MNIST dataset
(train_images, train_labels), (_, _) = mnist.load_data()
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)

# Generator
def build_generator():
    noise_input = Input(shape=(100,))
    label_input = Input(shape=(10,))
    
    x = concatenate([noise_input, label_input])
    x = Dense(128 * 7 * 7, activation='relu')(x)
    x = Reshape((7, 7, 128))(x)
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    generated_image = tf.keras.layers.Conv2DTranspose(1, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='tanh')(x)
    
    generator = Model([noise_input, label_input], generated_image)
    return generator

# Discriminator
def build_discriminator():
    image_input = Input(shape=(28, 28, 1))
    label_input = Input(shape=(10,))
    
    x = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(image_input)
    x = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = concatenate([x, label_input])
    x = Dense(1, activation='sigmoid')(x)
    
    discriminator = Model([image_input, label_input], x)
    return discriminator

# Build the generator and discriminator
generator = build_generator()
discriminator = build_discriminator()

# Compile the discriminator model
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# Build and compile the cGAN model
def build_cgan(generator, discriminator):
    noise_input = Input(shape=(100,))
    label_input = Input(shape=(10,))
    generated_image = generator([noise_input, label_input])
    cgan_output = discriminator([generated_image, label_input])
    
    cgan = Model([noise_input, label_input], cgan_output)
    discriminator.trainable = False
    cgan.compile(optimizer='adam', loss='binary_crossentropy')
    return cgan

# Create the cGAN model
cgan = build_cgan(generator, discriminator)

# Training loop
def train_cgan(generator, discriminator, cgan, epochs, batch_size):
    for epoch in range(epochs):
        for _ in range(train_images.shape[0] // batch_size):
            # Train the discriminator
            idx = np.random.randint(0, train_images.shape[0], batch_size)
            real_images = train_images[idx]
            real_labels = train_labels[idx]
            noise = np.random.normal(0, 1, (batch_size, 100))
            fake_labels = np.random.randint(0, 10, batch_size)  # Random labels for fake images
            fake_images = generator.predict([noise, tf.keras.utils.to_categorical(fake_labels, num_classes=10)])
            
            d_loss_real = discriminator.train_on_batch([real_images, real_labels], np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch([fake_images, tf.keras.utils.to_categorical(fake_labels, num_classes=10)], np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, 100))
            fake_labels = np.random.randint(0, 10, batch_size)  # Random labels for fake images
            g_loss = cgan.train_on_batch([noise, tf.keras.utils.to_categorical(fake_labels, num_classes=10)], np.ones((batch_size, 1)))
            
        print(f"Epoch {epoch+1}, D Loss: {d_loss}, G Loss: {g_loss}")
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1)

# Generate and save sample images
def generate_and_save_images(generator, epoch, examples=10):
    noise = np.random.normal(0, 1, (examples, 100))
    labels = np.eye(10)
    generated_images = generator.predict([noise, labels])
    generated_images = 0.5 * generated_images + 0.5  # Denormalize to [0, 1]

    plt.figure(figsize=(10, 1))
    for i in range(examples):
        plt.subplot(1, examples, i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'generated_image_epoch_{epoch}.png')
    plt.show()

# Hyperparameters
epochs = 5
batch_size = 128

# Train the cGAN
train_cgan(generator, discriminator, cgan, epochs, batch_size)

# Visualize the generated images at specific epochs
for epoch in range(1, epochs + 1):
    generate_and_save_images(generator, epoch)