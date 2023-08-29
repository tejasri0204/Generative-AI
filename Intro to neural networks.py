pip install tensorflow

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Generate random data
num_samples = 1000
input_dim = 20

data = np.random.random((num_samples, input_dim))

# Define the dimensions of the encoding space
encoding_dim = 10

# Define the encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)

# Define the decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Create the autoencoder model
autoencoder = Model(input_layer, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

epochs = 50
batch_size = 32

autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size)

encoded_data = autoencoder.predict(data)

import matplotlib.pyplot as plt

# Select a random sample for visualization
sample_index = np.random.randint(0, num_samples)

plt.figure(figsize=(10, 4))

# Original data
plt.subplot(1, 2, 1)
plt.title("Original Data")
plt.imshow(data[sample_index].reshape(5, 4), cmap='gray')

# Reconstructed data
plt.subplot(1, 2, 2)
plt.title("Reconstructed Data")
plt.imshow(encoded_data[sample_index].reshape(5, 4), cmap='gray')

plt.tight_layout()
plt.show()