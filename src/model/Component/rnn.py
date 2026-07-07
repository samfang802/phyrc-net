import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.bi_factor = 2 if bidirectional else 1
        
    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            output, (hn, cn) = self.lstm(x)
        else:
            output, (hn, cn) = self.lstm(x, (h0, c0))
        return output, (hn, cn)
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        
    def forward(self, x, h0=None):
        if h0 is None:
            output, hn = self.gru(x)
        else:
            output, hn = self.gru(x, h0)
        return output, hn
    
class Predictor(nn.Module):
    def __init__(self, num_layers, activation, bidirectional=False, hidden_state_dim=1024):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_state_dim = hidden_state_dim
        
        modules = []
        hidden_dim = self.hidden_state_dim*2 if bidirectional else self.hidden_state_dim
        match activation:
            case "relu":
                activation = nn.ReLU()
            case "softsign":
                activation = nn.Softsign()
            case "silu":
                activation = nn.SiLU()
        
        
        
        for i in range(num_layers):
            if i != (num_layers - 1):
                modules.append(nn.Linear(hidden_dim, hidden_dim // 2))
                modules.append(activation)
                hidden_dim = hidden_dim // 2
            else:
                modules.append(nn.Linear(hidden_dim, 1))
        self.fcn = nn.Sequential(*modules)

    def forward(self, x):
        x = self.fcn(x)
        return x
    
    

        