import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class IoTDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32).reshape(-1, 1)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        return self.decoder(self.encoder(x))

def generate_training_data(num_samples=1000):
    data = np.random.normal(loc=25, scale=2, size=num_samples)  # Normal temps
    return data

def train_autoencoder():
    # Data
    data = generate_training_data()
    dataset = IoTDataset(data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(50):
        for batch in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "autoencoder.pth")
    return model

if __name__ == "__main__":
    train_autoencoder()