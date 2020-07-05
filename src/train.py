import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def train_model(
    model,
    dataset_loader,
    device,
    sequences_per_batch,
    sequence_length,
    number_epochs,
    learning_rate,
    clip_gradients_max,
    show_loss_plot=False,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_over_epochs = []
    validation_loss_over_epochs = []

    for epoch in range(1, number_epochs + 1):
        hidden_states = model.init_hidden_states(sequences_per_batch)
        for x, y in dataset_loader.get_batches(sequences_per_batch, sequence_length):
            x = torch.tensor(x).to(device)
            y = torch.from_numpy(y).to(device)
            # Make a copy of the previous hidden states.
            hidden_states = tuple([each.data for each in hidden_states])
            optimizer.zero_grad()
            predicted_y, hidden_states = model.forward(x, hidden_states)
            loss = criterion(
                predicted_y, y.view(sequences_per_batch * sequence_length).long(),
            )
            loss.backward()
            """Gradient clipping avoids the problem of exploding gradients.
            
            If a gradient has a value higher than clip_max_normalization, all gradients will be normalized to the maximum value defined in clip_max_normalization.

            More information can be found on the links below:
            - https://machinelearningmastery.com/how-to-avoid-exploding-gradients-in-neural-networks-with-gradient-clipping/
            - https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48
            """
            nn.utils.clip_grad_norm_(model.parameters(), clip_gradients_max)
            optimizer.step()
            train_loss_over_epochs.append(loss.item())
        print(f"Epoch {epoch}/{number_epochs}...")

    if show_loss_plot:
        plt.plot(train_loss_over_epochs, label="Train loss")
        plt.plot(validation_loss_over_epochs, label="Validation loss")
        plt.legend()
        plt.title("Loss function")
        plt.show()

    return model
