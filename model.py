import torch
import torch.nn as nn

class ClassifierModel(nn.Module):
    def __init__(self, input_size, layers):
        super(ClassifierModel, self).__init__()
        modules = nn.ModuleList([nn.Flatten(), nn.Linear(input_size, layers[0]), nn.ReLU()])

        for idx, size in enumerate(layers[:-1]):
            modules.append(nn.Linear(size, layers[idx + 1]))
            modules.append(nn.ReLU())

        modules.append(nn.Linear(layers[-1], 2))
        modules.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)