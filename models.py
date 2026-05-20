import torch
import torch.nn as nn

# Teacher Model: Hybrid Transformer [cite: 208, 210]
class TeacherModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # standard multi-head attention (4-6 layers) [cite: 210]
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        x = self.transformer(x)
        return self.fc(x)

# Student Model: Linearized Attention [cite: 84, 177]
class StudentModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Simplified layer for edge efficiency [cite: 219, 228]
        self.linear_layer = nn.Linear(input_dim, 64)
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.linear_layer(x))
        return self.classifier(x)