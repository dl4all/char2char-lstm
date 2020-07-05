import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Char2Char(nn.Module):
    """Custom char2char network architecture.

    Args:
        dictionary_size (int): number of unique characters in the dataset.
        embedding_dimensions (int): number of dimensions for each embedded character.
        lstm_hidden_size (int): LSTM hidden state size.
        number_lstm_layers (int): number of LSTM layers stacked.
        dropout_probability (float): dropout probability after each LSTM layer and before the fully connected layer.
        output_size (int): output size of the fully connected layer.
        device (device): PyTorch class representing the CPU or GPU device.
    """

    def __init__(
        self,
        dictionary_size,
        embedding_dimensions,
        lstm_hidden_size,
        number_lstm_layers,
        dropout_probability,
        output_size,
        device,
    ):
        super(Char2Char, self).__init__()

        self.device = device
        self.lstm_hidden_size = lstm_hidden_size
        self.number_lstm_layers = number_lstm_layers

        self.embedding = nn.Embedding(dictionary_size, embedding_dimensions)
        self.lstm = nn.LSTM(
            input_size=embedding_dimensions,
            hidden_size=lstm_hidden_size,
            num_layers=number_lstm_layers,
            dropout=dropout_probability,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_probability)
        self.fc = nn.Linear(in_features=lstm_hidden_size, out_features=output_size)

    def forward(self, x, previous_hidden_states):
        """
        Args:
            x (int, int): input matrix.
            previous_hidden_states (hidden_state, cell_state): hidden states from the previous step in the recurrent network.

        Returns:
            (y, hidden_states): y with shape (sequences_per_batch * sequence_length, output_size) and the global hidden state and cell hidden state from LSTM.
        """

        """
        x:
            input: (sequences_per_batch, sequence_length)
            output: (sequences_per_batch, sequence_length, embedding_dimensions)
        """
        x = self.embedding(x)
        """
        previous_hidden_states:
            input: (number_lstm_layers, sequences_per_batch, lstm_hidden_size)
        hidden_states:
            output: (number_lstm_layers, sequences_per_batch, lstm_hidden_size)
        x:
            input: (sequences_per_batch, sequence_length, embedding_dimensions)
            output: (sequences_per_batch, sequence_length, lstm_hidden_size)
        """
        x, hidden_states = self.lstm(x, previous_hidden_states)
        # Dropout does not change the shape.
        x = self.dropout(x)
        """
        x:
            input: (sequences_per_batch, sequence_length, lstm_hidden_size)
            output: (sequences_per_batch * sequence_length, lstm_hidden_size)
        """
        x = x.view(x.shape[0] * x.shape[1], self.lstm_hidden_size)
        """
        x:
            input: (sequences_per_batch * sequence_length, lstm_hidden_size)
            output: (sequences_per_batch * sequence_length, output_size)
        """
        x = self.fc(x)
        return x, hidden_states

    def predict(
        self, character, dataset_loader, device, previous_hidden_states, top_k=5
    ):
        x = np.array([[dataset_loader.char2int[character]]])
        x = torch.from_numpy(x).to(device)

        # Make a copy of the previous hidden states.
        previous_hidden_states = tuple([each.data for each in previous_hidden_states])
        y, hidden_states = self.forward(x, previous_hidden_states)

        p = F.softmax(y, dim=1).data.cpu()
        p, top_ch = p.topk(k=top_k)
        top_ch = top_ch.numpy().squeeze()
        p = p.numpy().squeeze()
        character = np.random.choice(top_ch, p=p / p.sum())

        return dataset_loader.int2char[character], hidden_states

    def init_hidden(self, sequences_per_batch):
        """Create the initial hidden states filled with zeros.

        Args:
            sequences_per_batch (int): number of sequences per batch.

        Returns:
            (hidden_state, cell_state): tuple with hidden states.
        """
        return (
            torch.zeros(
                self.number_lstm_layers, sequences_per_batch, self.lstm_hidden_size
            ).to(self.device),
            torch.zeros(
                self.number_lstm_layers, sequences_per_batch, self.lstm_hidden_size
            ).to(self.device),
        )
