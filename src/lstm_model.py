import torch.nn as nn

from src.config import EMBED_DIM, HIDDEN_DIM


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        final_hidden = hidden[-1]

        x = self.fc1(final_hidden)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits