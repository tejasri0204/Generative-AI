pip install torch
pip install numpy
pip install matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size)
        self.decoder = nn.GRU(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        input_length = input_seq.size(0)
        target_length = target_seq.size(0)
        batch_size = target_seq.size(1)
        output_vocab_size = self.output_layer.out_features

        # Initialize hidden states for the encoder and decoder
        encoder_hidden = torch.zeros(1, batch_size, self.hidden_size)
        decoder_hidden = encoder_hidden

        # Initialize the outputs tensor
        outputs = torch.zeros(target_length, batch_size, output_vocab_size)

        # Encoder
        input_seq = self.embedding(input_seq)
        encoder_outputs, encoder_hidden = self.encoder(input_seq, encoder_hidden)

        # Decoder
        decoder_input = torch.tensor([[0] * batch_size], dtype=torch.long)  # Start of sequence token
        for t in range(target_length):
            decoder_input = self.embedding(decoder_input)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            output = self.output_layer(decoder_output)
            outputs[t] = output
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(2)
            decoder_input = (target_seq[t].unsqueeze(0) if teacher_force else top1)

        return outputs

# Define the training process
def train(input_seq, target_seq, model, optimizer, criterion):
    optimizer.zero_grad()
    output = model(input_seq, target_seq)
    output_dim = output.shape[-1]
    output = output[1:].view(-1, output_dim)
    target_seq = target_seq[1:].view(-1)
    loss = criterion(output, target_seq)
    loss.backward()
    optimizer.step()
    return loss.item()

# Define a function to translate a sentence
def translate_sentence(sentence, model, input_vocab, output_vocab, max_length=50):
    model.eval()
    tokens = sentence.split()
    input_seq = [input_vocab[token] for token in tokens]
    input_seq = torch.LongTensor(input_seq).unsqueeze(1)
    input_length = input_seq.size(0)
    batch_size = 1  # We are translating one sentence at a time
    target_length = max_length
    target_seq = torch.zeros(target_length, batch_size)

    with torch.no_grad():
        encoder_hidden = torch.zeros(1, batch_size, model.hidden_size)
        input_seq = model.embedding(input_seq)
        encoder_outputs, encoder_hidden = model.encoder(input_seq, encoder_hidden)

        decoder_input = torch.tensor([[0]], dtype=torch.long)  # Start of sequence token
        decoder_hidden = encoder_hidden

        for t in range(target_length):
            decoder_input = model.embedding(decoder_input)
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            output = model.output_layer(decoder_output)
            top1 = output.argmax(2)
            # Convert top1 to an integer
            top1_int = top1.item()
    
            target_seq[t] = top1

            if top1_int == output_vocab["<EOS>"]:
                break  # Stop generating when <EOS> is encountered

            decoder_input = top1

    translation = [output_vocab[token.item()] for token in target_seq]
    return " ".join(translation)

# Sample data
input_vocab = {"<PAD>": 0, "hello": 1, "world": 2, "how": 3, "are": 4, "you": 5}
output_vocab = {"<PAD>": 0, "hola": 1, "mundo": 2, "cómo": 3, "estás": 4}

# Make sure your model is in evaluation mode
model.eval()

# Define a function to translate a sentence
def translate_sentence(sentence, model, input_vocab, output_vocab):
    model.eval()
    tokens = sentence.split()
    input_seq = [input_vocab.get(token, 0) for token in tokens]  # Use 0 for unknown words
    input_seq = torch.LongTensor(input_seq).unsqueeze(1)
    target_length = 2 * len(tokens)  # Allow for some extra length for translations

    with torch.no_grad():
        output_seq = model(input_seq, torch.zeros(target_length, 1, dtype=torch.long))
        output_seq = output_seq.argmax(2)

    translation = []
    for i in output_seq:
        word = output_vocab.get(i.item(), "hola")  
        if word == "<EOS>":
            break
        translation.append(word)

    return " ".join(translation)

# Translate a sentence
input_sentence = "hello how are you world"

# Perform translation using the model and dictionaries
translation = translate_sentence(input_sentence, model, input_vocab, output_vocab)
print(f"Input: {input_sentence}")
print(f"Translation: {translation}")