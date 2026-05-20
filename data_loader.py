import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data 
import DataLoader, TensorDataset

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # Basic cleaning [cite: 202]
    df.drop_duplicates(inplace=True)
    
    # Separate features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Normalize features [cite: 202]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to float32 tensors [cite: 206]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def get_loaders(X, y, batch_size=64):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)