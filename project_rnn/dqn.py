import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, no_rnn_hidden, no_rnn_layer, ann_layers, no_action, batch_size):
        super().__init__()
        self.D = input_dim
        self.M = no_rnn_hidden
        self.L = no_rnn_layer
        self.B = batch_size
        
        self.lstm = nn.LSTM(self.D, self.M, self.L, batch_first=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ann = []
        input_count = self.M

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
        N, _, _ = x.shape
        h0, c0 = torch.zeros(self.L, N, self.M, device=self.device), torch.zeros(self.L, N, self.M, device=self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:,-1,:]
        out = self.layers(out)
        return out

    def set_hidden(self, hidden, cell):
        self.hidden = hidden.detach()
        self.cell = cell.detach()

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_model(torch.load(path))