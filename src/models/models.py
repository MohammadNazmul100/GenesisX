import torch
import torch.nn as nn

class GenesisXModel(nn.Module):
    def __init__(self, vocabulary_size=5000, embedding_dim=256, hidden_layer_dim=512, num_transformer_layers=4):
        super(GenesisXModel, self).__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_dim)
        self.transformer_block = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dim_feedforward=hidden_layer_dim, batch_first=True)
        self.encoder_stack = nn.TransformerEncoder(self.transformer_block, num_layers=num_transformer_layers)
        self.output_layer = nn.Linear(embedding_dim, vocabulary_size)
    
    def forward(self, input_tokens):
        embedded_tokens = self.embedding_layer(input_tokens)
        encoded_output = self.encoder_stack(embedded_tokens)
        final_output = self.output_layer(encoded_output)
        return final_output