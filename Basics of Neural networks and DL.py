import numpy as np
import tensorflow as tf
from tensorflow import keras

# Define the input text
text = "Hello, how are you doing today? I hope you're having a great time!"

# Create a character mapping (unique characters to indices)
chars = sorted(list(set(text)))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# Convert the input text to a sequence of indices
sequence_length = 40
sequences = []
next_chars = []
for i in range(0, len(text) - sequence_length):
    sequences.append(text[i:i + sequence_length])
    next_chars.append(text[i + sequence_length])

X = np.zeros((len(sequences), sequence_length, len(chars)), dtype=np.bool)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

# Build the model
model = keras.Sequential([
    keras.layers.LSTM(128, input_shape=(sequence_length, len(chars))),
    keras.layers.Dense(len(chars), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X, y, epochs=50, batch_size=32)

# Generate text
seed_text = "Hello, how are you doing today?"
generated_text = seed_text

for _ in range(200):
    x_pred = np.zeros((1, sequence_length, len(chars)))
    for t, char in enumerate(seed_text):
        x_pred[0, t, char_to_idx[char]] = 1

    predicted_char_idx = np.argmax(model.predict(x_pred), axis=-1)[0]
    predicted_char = idx_to_char[predicted_char_idx]

    generated_text += predicted_char
    seed_text = seed_text[1:] + predicted_char

print(generated_text)
