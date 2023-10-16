# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Example dataset
data = [
    "Hello, how are you?",
    "I am doing well.",
    "What's your name?",
    "My name is AIBot.",
    "Glad to meet you!",
]

# Initialize a tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

# Convert text data to sequences
sequences = tokenizer.texts_to_sequences(data)

# Generate training data
input_sequences = []
output_sequences = []

for sequence in sequences:
    for i in range(1, len(sequence)):
        input_sequences.append(sequence[:i])
        output_sequences.append(sequence[i])

# Pad sequences for a consistent input size
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
output_sequences = np.array(output_sequences)

# Create the RNN model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_sequence_length))
model.add(SimpleRNN(128, activation='relu'))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(input_sequences, output_sequences, epochs=100, verbose=2)

# Generate text
seed_text = "Hello,"
generated_text = []

while True:
    input_sequence = tokenizer.texts_to_sequences([seed_text])[0]
    input_sequence = pad_sequences([input_sequence], maxlen=max_sequence_length, padding='pre')
    predicted_word_index = np.argmax(model.predict(input_sequence), axis=-1)
    predicted_word = tokenizer.index_word.get(predicted_word_index[0], "")

    if predicted_word in ['?', '.', '!']:
        generated_text.append(seed_text)
        seed_text = ""  # Reset seed_text for the next sentence
    elif seed_text and seed_text[-1] not in ['?', '.', '!'] and predicted_word:
        seed_text += " " + predicted_word
    else:
        seed_text += predicted_word

    if len(generated_text) >= 4:
        break

print("Generated Text:")
print("\n".join(generated_text))