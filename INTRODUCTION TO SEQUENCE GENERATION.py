import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Replace this with your own text data
text_data = """
To be or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And, by opposing, end them?
"""
text_data = text_data.lower()  # Convert to lowercase for consistency

# Create a set of unique characters in the text
chars = sorted(set(text_data))

# Create character-to-integer and integer-to-character mappings
char_to_int = {char: i for i, char in enumerate(chars)}
int_to_char = {i: char for i, char in enumerate(chars)}

# Total number of unique characters
num_chars = len(chars)

# Define the sequence length
seq_length = 100

# Create input and target sequences
input_sequences = []
target_sequences = []

for i in range(0, len(text_data) - seq_length, 1):
    input_seq = text_data[i:i + seq_length]
    target_seq = text_data[i + seq_length]
    input_sequences.append([char_to_int[char] for char in input_seq])
    target_sequences.append(char_to_int[target_seq])

# Reshape the input sequences
X = np.reshape(input_sequences, (len(input_sequences), seq_length, 1))
X = X / float(num_chars)

# One-hot encode the target sequences
y = tf.keras.utils.to_categorical(target_sequences, num_classes=num_chars)

# Build the RNN model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X, y, epochs=50, batch_size=64)

# Define the sequence length
seq_length = 100  # Adjust it back to the original value

# Generate text using the trained model
seed_text = "arthur unsheathed his sword and"
generated_text = []

for i in range(500):  # Generate 500 characters
    # Pad the seed text if it's shorter than the sequence length
    if len(seed_text) < seq_length:
        pad_length = seq_length - len(seed_text)
        seed_text = " " * pad_length + seed_text

    x_input = np.reshape([char_to_int[char] for char in seed_text[-seq_length:]], (1, seq_length, 1))
    x_input = x_input / float(num_chars)
    char_index = np.argmax(model.predict(x_input, verbose=0))
    next_char = int_to_char[char_index]
    generated_text.append(next_char)
    seed_text += next_char

# Print each character as it's generated
for char in generated_text:
    print(char, end='')

# Print a newline character at the end to separate from the input
print()