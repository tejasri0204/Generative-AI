import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention
from tensorflow.keras.models import Model

# Define the vocabulary sizes for English and French
en_vocab_size = 10000
fr_vocab_size = 10000

# Define the sequence lengths
en_sequence_length = 20
fr_sequence_length = 20

# Generate random training data (English and French sentences)
num_samples = 1000

en_input_data = np.random.randint(1, en_vocab_size, size=(num_samples, en_sequence_length))
fr_input_data = np.random.randint(1, fr_vocab_size, size=(num_samples, fr_sequence_length))
fr_output_data = np.random.randint(1, fr_vocab_size, size=(num_samples, fr_sequence_length))

# Define the input layers
en_input = Input(shape=(en_sequence_length,))
fr_input = Input (shape=(fr_sequence_length,))

# Create embedding layers
en_embed= Embedding(en_vocab_size, 256) (en_input)
fr_embed= Embedding(fr_vocab_size, 256) (fr_input)

# Define the encoder LSTM
encoder = LSTM(256, return_sequences=True, return_state=True) 
encoder_output, state_h, state_c = encoder(en_embed)

# Define the decoder LSTM
decoder = LSTM(256, return_sequences=True, return_state=True)
decoder_output, _, _ = decoder(fr_embed, initial_state=[state_h, state_c])

# Add attention mechanism
attention = Attention()([decoder_output, encoder_output])

# Concatenate attention output and decoder output
decoder_combined_context = tf.concat([attention, decoder_output], axis=-1)

# Dense layer to produce the output
output_layer = Dense(fr_vocab_size, activation="softmax") 
output = output_layer(decoder_combined_context)

# Define the model
model = Model([en_input, fr_input], output)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# One-hot encode the target data (fr_output_data)
fr_output_data_onehot = tf.one_hot(fr_output_data, fr_vocab_size)

# Train the model
model.fit([en_input_data, fr_input_data], fr_output_data_onehot, epochs=10, batch_size=64)

# Generate random evaluation data (English and French sentences)
num_eval_samples = 200
en_eval_data = np.random.randint(1, en_vocab_size, size=(num_eval_samples, en_sequence_length))
fr_eval_data = np.random.randint(1, fr_vocab_size, size=(num_eval_samples, fr_sequence_length))
fr_eval_output_data = np.random.randint(1, fr_vocab_size, size=(num_eval_samples, fr_sequence_length))

# One-hot encode the evaluation target data (fr_eval_output_data)
fr_eval_output_data_onehot = tf.one_hot(fr_eval_output_data, fr_vocab_size)

# Evaluate the model on the evaluation data
evaluation_loss, evaluation_accuracy = model.evaluate([en_eval_data, fr_eval_data], fr_eval_output_data_onehot)
print(f"Evaluation Loss: {evaluation_loss}, Evaluation Accuracy: {evaluation_accuracy}")

# Now, let's calculate the BLEU score for the model's translations
from nltk.translate.bleu_score import corpus_bleu

# Define a function to translate sentences using the trained model
def translate_sentences(sentences):
    translated_sentences = []
    for sentence in sentences:
        translated_sentence = translate_sentence(sentence)
        translated_sentences.append(translated_sentence)
    return translated_sentences

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

from nltk.translate.bleu_score import SmoothingFunction

smoother = SmoothingFunction().method1  # Choose a smoothing method
bleu_score = corpus_bleu(reference_translations, candidate_translations, smoothing_function=smoother)
print(f"BLEU Score with Smoothing: {bleu_score}")

# Generate a set of English sentences for evaluation
en_eval_sentences = ["this is a test", "translate this sentence", "how are you", "machine translation is amazing"]

# Define reference translations (French)
fr_eval_sentences = ["c'est un test", "traduisez cette phrase", "comment ça va", "la traduction automatique est incroyable"]

# Translate the English sentences to French
fr_translated_sentences = [translate_sentence(en_sentence) for en_sentence in en_eval_sentences]

# Tokenize reference and candidate translations
reference_translations = [[sentence.split()] for sentence in fr_eval_sentences]
candidate_translations = [sentence.split() for sentence in fr_translated_sentences]

# Calculate BLEU score for each sentence and average them
individual_bleu_scores = [sentence_bleu(reference, candidate) for reference, candidate in zip(reference_translations, candidate_translations)]
average_bleu_score = np.mean(individual_bleu_scores)

print(f"Individual BLEU Scores: {individual_bleu_scores}")
print(f"Average BLEU Score: {average_bleu_score}")