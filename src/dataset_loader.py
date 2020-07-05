import os
import string
import random
import torch
import unidecode
from torch.utils.data.dataset import Dataset
import numpy as np


def read_whole_file(input_path):
    with open(input_path, mode="r") as file:
        return unidecode.unidecode(file.read())


class DatasetLoader(Dataset):
    """Custom dataset loader.

    Args:
        input_path (str): path to the dataset.
        validation_split (float, optional): percentage of the dataset used to validate the model. Defaults to 0.0.
    """

    def __init__(self, input_path, validation_split=0.0):
        self.data = read_whole_file(input_path)
        self.dictionary_size = len(string.printable)
        self.int2char = {i: char for i, char in enumerate(string.printable)}
        self.char2int = {char: i for i, char in enumerate(string.printable)}
        self.data_encoded = np.array([self.char2int[x] for x in self.data])

        validation_index = int(len(self.data_encoded) * (1 - validation_split))
        train_data, validation_data = (
            self.data_encoded[:validation_index],
            self.data_encoded[validation_index:],
        )
        self.train_data = train_data
        self.validation_data = validation_data

    def get_batches(self, sequences_per_batch, sequence_length, validation=False):
        """Get batches to train or validate the model.

        Args:
            sequences_per_batch (int): number of sequences per batch.
            sequence_length (int): number of characters in a sequence.
            validation (bool, optional): True when you want to get validation data. Defaults to False.

        Yields:
            (int, int): tuple of input and target characters.
        """
        data = self.validation_data if validation else self.train_data
        # Number of characters required to fill a batch.
        batch_size = sequences_per_batch * sequence_length
        # Number of batches that can be created from the data.
        number_batches = len(data) // batch_size
        # Get only the data that will be used to create complete batches.
        data = data[: number_batches * batch_size]
        # Reshape the data to `sequences_per_batch` rows.
        data = data.reshape((sequences_per_batch, -1))

        for i in range(0, data.shape[1], sequence_length):
            # Input.
            x = data[:, i : i + sequence_length]
            # Target.
            y = np.zeros_like(x)
            # The target is the input shifted to the right by one character.
            y[:, :-1] = x[:, 1:]
            try:
                y[:, -1] = data[:, i + sequence_length]
            except IndexError:
                # The last batch can raise this error. When that happens, get the first column of the `data`.
                y[:, -1] = data[:, 0]
            yield x, y


if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))
    dataset_loader = DatasetLoader(os.path.join(file_path, "./dataset/sample.txt"))
    for x, y in dataset_loader.get_batches(32, 32):
        assert x.shape == (32, 32)
        assert y.shape == (32, 32)
