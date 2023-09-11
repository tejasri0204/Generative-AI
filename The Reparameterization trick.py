import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse

# Generate toy dataset
np.random.seed(0)
n_samples = 1000
n_features = 10
latent_dim = 2

x_train = np.random.randn(n_samples, n_features)

# Define VAE architecture
input_layer = Input(shape=(n_features,))
hidden_encoder = Dense(16, activation='relu')(input_layer)
z_mean = Dense(latent_dim)(hidden_encoder)
z_log_var = Dense(latent_dim)(hidden_encoder)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.0, stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

hidden_decoder = Dense(16, activation='relu')(z)
output_layer = Dense(n_features, activation='linear')(hidden_decoder)

# Define VAE model
vae = Model(input_layer, output_layer)

# VAE loss function
reconstruction_loss = mse(input_layer, output_layer)
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)

# Compile VAE model
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Train VAE model
vae.fit(x_train, epochs=50, batch_size=32, verbose=2)

# Generate samples from the trained VAE
n_samples_to_generate = 10
generated_samples = vae.predict(np.random.randn(n_samples_to_generate, n_features))

print("Generated Samples:")
print(generated_samples)
