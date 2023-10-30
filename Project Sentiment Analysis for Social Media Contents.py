import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Input, Attention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split

# Load the IMDb dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Preprocess the data
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
X_train = tokenizer.sequences_to_matrix(train_data, mode='binary')
X_test = tokenizer.sequences_to_matrix(test_data, mode='binary')

X_train = pad_sequences(train_data, maxlen=100)
X_test = pad_sequences(test_data, maxlen=100)

# Convert labels to numerical values (positive, negative, neutral)
positive_reviews = (train_labels == 1)
negative_reviews = (train_labels == 0)
neutral_reviews = ~positive_reviews & ~negative_reviews

train_labels[positive_reviews] = 0
train_labels[negative_reviews] = 1
train_labels[neutral_reviews] = 2

# Split the data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, train_labels, test_size=0.2, random_state=42)

# Create an attention-based model
input_layer = Input(shape=(100,))
embed = Embedding(10000, 128)(input_layer)
lstm = Bidirectional(LSTM(128, return_sequences=True))(embed)
attention = Attention()([lstm, lstm])
attended_lstm = tf.reduce_sum(attention * lstm, axis=1)
output_layer = Dense(3, activation='softmax')(attended_lstm)
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, test_labels)
print(f'Test accuracy: {test_acc}')

# Plot accuracy and loss graphs
plt.figure(figsize=(12, 6))

# Accuracy graph
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss graph
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

