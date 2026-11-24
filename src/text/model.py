"""
CatalogUpdate AI - Character-Level LSTM Language Model.

Architecture:
- Character Embedding layer (vocab_size -> embedding_dim).
- 2-layer stacked LSTM (hidden_dim = 256, batch_first=True, optional dropout).
- Fully-connected Linear projection (hidden_dim -> vocab_size).
"""

from typing import Tuple, Optional, Union
import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    """
    2-Layer Character-Level LSTM for autoregressive specification text generation.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through Embedding, 2-layer LSTM, and output Linear layer.

        Args:
            x: Tensor of character indices with shape (batch_size, seq_len).
            hidden: Optional (h_0, c_0) state tuple.

        Returns:
            Tuple of:
            - logits: Unnormalized predictions with shape (batch_size, seq_len, vocab_size).
            - (h_n, c_n): Updated hidden and cell states.
        """
        # x: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embed = self.embedding(x)
        out, hidden = self.lstm(embed, hidden)
        # out: (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, vocab_size)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(
        self,
        batch_size: int,
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes zeroed hidden and cell states (num_layers, batch_size, hidden_dim)."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)
