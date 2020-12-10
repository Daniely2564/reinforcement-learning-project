import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, ann_layers, no_action, batch_size):
        super().__init__()
        self.D = input_dim

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ann = []
        input_count = self.D

        for layer in ann_layers:
            ann.append(nn.Linear(input_count, layer))
            ann.append(nn.ReLU())
            ann.append(nn.BatchNorm1d(layer))
            ann.append(nn.Dropout(.5))

            input_count = layer
        ann.append(nn.Linear(input_count, no_action))
        self.layers = nn.Sequential(*ann)

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        out = self.layers(x)
        return out

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_model(torch.load(path))