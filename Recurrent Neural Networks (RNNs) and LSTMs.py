import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample training data (poetry)
text_corpus = """
Roses are red,
Violets are blue,
I am learning AI,
And so can you!
"""

# Step 2: Preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_corpus])
total_words = len(tokenizer.word_index) + 1

# Create input-output sequences
input_sequences = []
for line in text_corpus.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences to a fixed length
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Create predictors and labels
X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Step 3: Build the LSTM Model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_length-1))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Training (on the sample data)
X_train, y_train = X, y  # Replace with your actual training data
epochs = 100  # Adjust as needed
batch_size = 32  # Adjust as needed

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Step 5: Text Generation (after training)
def generate_text(seed_text, next_words, model, max_sequence_length):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_token = np.argmax(predicted_probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_token:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

generated_text = generate_text("Roses are", 5, model, max_sequence_length)
print(generated_text)