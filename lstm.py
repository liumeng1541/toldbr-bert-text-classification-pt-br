from torch import nn
import torch

class DisModel(nn.Module):
    def __init__(self, latent_dim=120):
        super(DisModel, self).__init__()
        self.latent_dim = latent_dim
        self.discriminator = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.discriminator(z)
        return out

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bert_size = 2
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size,
                            num_layers=num_layers,
                            bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(self.hidden_size * self.num_directions, self.output_size)
        
        # Define MLPs for Bert to LSTM and LSTM to Bert transformations
        self.bert_to_lstm_mlp = MLP(self.bert_size, 64, self.output_size)  # Example hidden size
        self.lstm_to_bert_mlp = MLP(self.output_size, 64, self.bert_size)  # Example hidden size

    def forward(self, x):
        x = x.unsqueeze(1)
        batch_size, seq_len = x.shape[0], x.shape[2]
        x_in = x.permute(1, 0, 2)
        device = next(self.lstm.parameters()).device
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, _ = self.lstm(x_in, (h_0, c_0))
        out = self.dropout(output.squeeze())
        out = self.linear(out)
        return out

    def BertToLSTM(self, x):
        # Assuming x is of shape [batch_size, bert_size]
        out = self.bert_to_lstm_mlp(x)
        return out

    def LSTMtoBert(self, x):
        # Assuming x is of shape [batch_size, hidden_size * num_directions]
        out = self.lstm_to_bert_mlp(x)
        return out

