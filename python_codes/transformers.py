import torch
import torch.nn as nn
import math


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super(TransformerClassifier, self).__init__()
        self.input_fc = nn.Linear(
            input_dim, d_model
        )  # Project input to d_model dimension

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        # Transformer Decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=num_layers
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_fc(x)  # (batch_size, seq_len, d_model)
        x = self.relu(x)
        memory = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        x = self.transformer_decoder(x, memory)  # (batch_size, seq_len, d_model)
        x = x.mean(dim=1)  # Global Average Pooling
        x = self.dropout(x)
        x = self.fc_out(x)
        return x
