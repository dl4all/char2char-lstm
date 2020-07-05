import os
import numpy as np
import torch
import torch.nn as nn
from dataset_loader import DatasetLoader
from char2char import Char2Char
from train import train_model
from predict import get_sample


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_path = os.path.dirname(os.path.abspath(__file__))
dataset_loader = DatasetLoader(
    os.path.join(file_path, "./dataset/sample.txt"), validation_split=0.3
)

# Set hyperparameters.
dictionary_size = dataset_loader.dictionary_size
embedding_dimensions = 128
lstm_hidden_size = 512
number_lstm_layers = 2
dropout_probability = 0.5
output_size = dataset_loader.dictionary_size
learning_rate = 1e-3
# Set other parameters.
sequences_per_batch = 16
sequence_length = 16
number_epochs = 32
clip_gradients_max = 5

model = Char2Char(
    dictionary_size,
    embedding_dimensions,
    lstm_hidden_size,
    number_lstm_layers,
    dropout_probability,
    output_size,
    device,
).to(device)

print(model)

model = train_model(
    model,
    dataset_loader,
    device,
    sequences_per_batch=sequences_per_batch,
    sequence_length=sequence_length,
    number_epochs=number_epochs,
    learning_rate=learning_rate,
    clip_gradients_max=clip_gradients_max,
    show_loss_plot=True,
)

generated_text = get_sample(model, dataset_loader, device, 1000, "int adxl_decode(")
print(generated_text)
