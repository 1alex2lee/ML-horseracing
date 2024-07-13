import torch
import torch.nn as nn
import os

import numpy as np 
import pandas as pd
import datetime
import pickle
import time
import random

from model import ClassifierModel

def train_model (x_tensor, y_tensor, layers):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # print(f"Using {device} device")


    # in_path = os.path.join("drive","MyDrive","Colab Notebooks","HKJC-ML")

    # x_tensor = torch.load(os.path.join(in_path, "x_tensor")).to(torch.float32).to(device)
    # y_tensor = torch.load(os.path.join(in_path, "y_tensor")).to(torch.float32).to(device)

    x_tensor = x_tensor.to(torch.float32).to(device)
    y_tensor = y_tensor.to(torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)

    rng = torch.Generator().manual_seed(42)
    training_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=rng)

    input_size = x_tensor.shape[1]
    model = ClassifierModel(input_size, layers).to(device)
    # print(model)

    learning_rate = 1e-3
    batch_size = 32

    train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 5000
    patience = 250

    train_loss_plot = []
    val_loss_plot = []
    epochs_plot = []

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for e in range(epochs):
        # if e % 10 == 0:
        #     print(f"Epoch {e}\n-------------------------------")
        size = len(train_dataloader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()
        train_loss_sum = 0
        num_train_batches = 0
        for batch, (X, y) in enumerate(train_dataloader):
            # Compute prediction and loss
            # X = torch.swapaxes(X, 0, 2)
            pred = model(X)
            # y = torch.unsqueeze(y, dim=1)
            # y = torch.squeeze(y)
            # print(pred, y)
            loss = loss_fn(pred, y)
            train_loss_sum += loss.item()
            num_train_batches += 1
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        train_loss_plot.append(train_loss_sum/num_train_batches)

        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.eval()
        size = len(val_dataloader.dataset)
        num_val_batches = len(val_dataloader)
        val_loss, val_acc = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during val mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in val_dataloader:
                pred = model(X)
                # y = torch.unsqueeze(y, dim=1)
                val_loss += loss_fn(pred, y).item()
                # print(pred, y)
                val_acc += (pred == y).type(torch.float).sum().item()

        val_loss /= num_val_batches
        val_acc /= num_val_batches
        # val_loss_plot.append(val_loss)
        # epochs_plot.append(e+1)

        # if e % 100 == 0:
        #     print(f"epoch {e:d} val loss: {val_loss:>8f} val acc: {val_acc:>8f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    # plt.plot(epochs_plot, train_loss_plot, label = "training loss")
    # plt.plot(epochs_plot, val_loss_plot, label = "validation loss")
    # plt.legend()
    # print('best val loss', best_val_loss)
            
    # print('best val loss', best_val_loss, 'best val acc', best_val_acc)
            
    # return best_model_state

    file_name = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}_"
    for l in layers:
        file_name += f'{str(l)}_'
    file_name += f"{batch_size}_{e}_{str(f'{best_val_loss:.3g}').split('.')[-1]}"

    out_path = os.path.join('model_configs', file_name)
    torch.save(best_model_state, out_path)
    
    print(file_name, 'saved', 'best val loss', best_val_loss, 'best val acc', best_val_acc)

    return file_name, best_val_loss, best_val_acc
