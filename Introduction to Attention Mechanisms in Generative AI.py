pip install torch numpy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as numpy

source_sequence = torch.tensor([[1, 2, 3, 4, 5]])
target_sequence = torch.tensor([[6, 7, 8, 9, 10]])

class Seq2SeqAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Seq2SeqAttention, self).__init__()

        # Define the encoder and decoder layers here

        # Encoder
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.encoder_rnn = nn.LSTM(hidden_dim, hidden_dim)

        # Attention layer
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)

        # Decoder
        self.decoder_rnn = nn.LSTM(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

class Seq2SeqAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Seq2SeqAttention, self).__init__()

        # Define the encoder and decoder layers here

        # Encoder
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.encoder_rnn = nn.LSTM(hidden_dim, hidden_dim)

        # Attention layer
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)

        # Decoder
        self.decoder_rnn = nn.LSTM(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

def forward(self, src, trg):
        # Encoder
        src_embedded = self.embedding(src)
        encoder_outputs, (encoder_hidden, _) = self.encoder_rnn(src_embedded)

        # Decoder
        trg_embedded = self.embedding(trg)
        decoder_hidden = encoder_hidden  # Initialize decoder hidden state

        # Initialize attention scores
        attention_scores = torch.zeros(trg.shape[0], trg.shape[1], src.shape[1]).to(trg.device)

        outputs = []

        for t in range(trg.shape[1]):
            # Calculate attention scores
            attn_input = torch.cat((trg_embedded[:, t, :].unsqueeze(1), decoder_hidden[0].unsqueeze(1)), dim=2)
            attn_score = self.attn(attn_input)

            # Update attention scores
            attention_scores[:, t, :] = attn_score

            # Use attention scores to weight encoder outputs
            weighted_encoder_outputs = torch.matmul(attn_score, encoder_outputs)

            # Pass through decoder LSTM
            decoder_input = torch.cat((trg_embedded[:, t, :].unsqueeze(1), weighted_encoder_outputs), dim=2)
            decoder_output, decoder_hidden = self.decoder_rnn(decoder_input, decoder_hidden)

 # Predict the next word
            output = self.output_layer(decoder_output.squeeze(1))
            outputs.append(output)

        # Stack all the outputs and transpose to get the final result
        outputs = torch.stack(outputs, dim=1)

        return outputs, attention_scores

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for batch in iterator:
        src, trg = batch

        optimizer.zero_grad()
        output, _ = model(src, trg)
        output_dim = output.shape[-1]

        output = output[:, 1:, :].contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Evaluation
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in iterator:
            src, trg = batch
            output, _ = model(src, trg)
            output_dim = output.shape[-1]

            output = output[:, 1:, :].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Define your START_TOKEN and END_TOKEN
START_TOKEN = 0  # Replace with the actual start token value in your dataset
END_TOKEN = 1    # Replace with the actual end token value in your dataset# Define your START_TOKEN and END_TOKEN

# Define a function for making predictions
def predict_sequence(model, source_sequence, max_length=20):
    model.eval()

    with torch.no_grad():
        # Initialize the input sequence
        input_seq = source_sequence.unsqueeze(0)  # Add a batch dimension

        # Initialize the output sequence with the start token (you may need to change this)
        start_token = torch.tensor([START_TOKEN]).unsqueeze(0)  # Replace START_TOKEN with the appropriate value
        output_seq = start_token

        for t in range(max_length):
            # Predict the next token in the sequence
            output, _ = model(input_seq, output_seq)

            # Get the last predicted token (you may need to adjust this depending on your model's architecture)
            predicted_token = output[:, -1, :]

            # Append the predicted token to the output sequence
            output_seq = torch.cat((output_seq, predicted_token.unsqueeze(1)), dim=1)

            # Check for an end token or another stopping condition (e.g., reaching max length)
            if predicted_token.item() == END_TOKEN or t >= max_length - 1:
                break

        return output_seq.squeeze(0)  # Remove the batch dimension from the output sequence

# Assuming source_sequence is a 1-D tensor representing your source sequence
        source_sequence = torch.tensor([1, 2, 3, 4, 5])  # Replace with your actual data

        # Reshape to a 2-D tensor with batch size of 1 and sequence length
        input_seq = source_sequence.unsqueeze(0)

        # Predict the sequence
        predicted_sequence = predict_sequence(model, input_seq)

        # Convert the predicted_sequence tensor to a list of predicted values
        predicted_values = predicted_sequence.tolist()[0]

        # Print the predicted values
        print("Predicted Sequence:", predicted_values)