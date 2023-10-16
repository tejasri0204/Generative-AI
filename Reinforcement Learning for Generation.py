!pip install nltk
!pip install tensorflow
!pip install numpy

import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK data for tokenization
nltk.download("punkt")

# Sample dataset (you can replace this with your data)
data = [
    {"keyword": "hello", "response": "Hello! How can I help you today?"},
    {"keyword": "weather", "response": "The weather today is sunny with a high of 25°C."},
    {"keyword": "greeting", "response": "Greetings! How can I assist you?"},
    {"keyword": "time", "response": "The current time is 2:30 PM."},
]

# Tokenize and preprocess text data
keywords = [word_tokenize(example["keyword"]) for example in data]
responses = [word_tokenize(example["response"]) for example in data]

# Build a vocabulary
vocab = FreqDist(np.concatenate(responses))
vocab_size = len(vocab)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(responses)
sequences = tokenizer.texts_to_sequences(responses)
input_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=input_length)

# Reinforcement Learning Setup
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=input_length))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Define a simple loss function for illustration (replace with appropriate loss function)
def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Convert y_true to float32 for consistency
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Compile the model with the custom loss function
model.compile(loss=custom_loss, optimizer="adam")

# Simulated training loop (replace this with actual training data)
for _ in range(100):
    # Generate random target data for illustration
    target_data = np.random.randint(vocab_size, size=len(padded_sequences))
    
    # Train the model
    model.fit(padded_sequences, target_data)

# Generate a response for user input
user_input = "What's the weather like today?"
tokenized_input = word_tokenize(user_input)

# Use the model to generate a response
input_sequence = tokenizer.texts_to_sequences([tokenized_input])
padded_input_sequence = pad_sequences(input_sequence, maxlen=input_length)
response_probabilities = model.predict(padded_input_sequence)

# Decode the response
response_index = np.argmax(response_probabilities)
response = tokenizer.index_word[response_index]

# Print the generated response
print("Generated Response:", response)
