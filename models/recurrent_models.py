import torch
import torch.nn as nn

class SimpleRNNModel(nn.Module):
    """
    Standard Elman Recurrent Neural Network for Autoregressive Next-Word Prediction.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        super(SimpleRNNModel, self).__init__()
        self.model_name = "Simple RNN"
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len)
        embeds = self.dropout(self.embedding(x)) # (batch_size, seq_len, embed_dim)
        out, hidden = self.rnn(embeds, hidden)   # out: (batch_size, seq_len, hidden_dim)
        # Use representation from the last time step to predict the next token
        last_step = out[:, -1, :]                # (batch_size, hidden_dim)
        logits = self.fc(self.dropout(last_step)) # (batch_size, vocab_size)
        return logits, hidden


class BidirectionalRNNModel(nn.Module):
    """
    Bidirectional Recurrent Neural Network for Autoregressive Next-Word Prediction.
    Encodes the input prompt sequence in both forward and backward directions to form
    a comprehensive sequence representation before predicting the next token.
    Hidden dimension per direction is hidden_dim // 2, so the concatenated representation
    matches hidden_dim for fair parameter comparison.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        super(BidirectionalRNNModel, self).__init__()
        self.model_name = "Bidirectional RNN"
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dir_hidden = hidden_dim // 2

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.birnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=self.dir_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.dir_hidden * 2, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.dropout(self.embedding(x))
        out, hidden = self.birnn(embeds, hidden) # out: (batch_size, seq_len, dir_hidden * 2)
        # In PyTorch bidirectional RNN, output at last step concatenates forward state at T
        # and backward state at T.
        # To capture full sequence context from both ends:
        # forward summary is at last step out[:, -1, :self.dir_hidden]
        # backward summary (which processed w_L -> w_1) is at step 0 out[:, 0, self.dir_hidden:]
        fwd_last = out[:, -1, :self.dir_hidden]
        bwd_first = out[:, 0, self.dir_hidden:]
        context = torch.cat([fwd_last, bwd_first], dim=1) # (batch_size, hidden_dim)

        logits = self.fc(self.dropout(context))
        return logits, hidden


class LSTMModel(nn.Module):
    """
    Long Short-Term Memory (LSTM) Network for Autoregressive Next-Word Prediction.
    Utilizes cell states and gating mechanisms (forget, input, output gates) to
    mitigate vanishing gradients and capture long-range dependencies.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.model_name = "LSTM"
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.dropout(self.embedding(x))
        out, hidden = self.lstm(embeds, hidden)
        last_step = out[:, -1, :]
        logits = self.fc(self.dropout(last_step))
        return logits, hidden


class GRUModel(nn.Module):
    """
    Gated Recurrent Unit (GRU) Network for Autoregressive Next-Word Prediction.
    Employs update and reset gates for efficient long-term dependency modeling
    with fewer parameters than LSTM.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.model_name = "GRU"
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.dropout(self.embedding(x))
        out, hidden = self.gru(embeds, hidden)
        last_step = out[:, -1, :]
        logits = self.fc(self.dropout(last_step))
        return logits, hidden


def count_parameters(model):
    """Counts total trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_all_models(vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
    """Instantiates all 4 models with identical hyperparameters."""
    models = {
        "Simple RNN": SimpleRNNModel(vocab_size, embed_dim, hidden_dim, num_layers, dropout),
        "Bidirectional RNN": BidirectionalRNNModel(vocab_size, embed_dim, hidden_dim, num_layers, dropout),
        "LSTM": LSTMModel(vocab_size, embed_dim, hidden_dim, num_layers, dropout),
        "GRU": GRUModel(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    }
    return models
