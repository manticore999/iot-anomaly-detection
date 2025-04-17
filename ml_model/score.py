import json
import torch
import torch.nn as nn
import numpy as np
import os

# Define the Autoencoder class (must match your training code)
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        return self.decoder(self.encoder(x))

# Global model variable
model = None

def init():
    global model
    # Load the model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "outputs"), "autoencoder.pth")
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully")

def run(raw_data):
    try:
        # Parse the input JSON
        input_data = json.loads(raw_data)
        if "data" not in input_data:
            return json.dumps({"error": "Input must contain 'data' key with temperature values"})

        # Convert input to tensor
        data = np.array(input_data["data"], dtype=np.float32).reshape(-1, 1)
        input_tensor = torch.tensor(data, dtype=torch.float32)

        # Run inference
        with torch.no_grad():
            reconstructed = model(input_tensor)
            # Calculate reconstruction error (MSE per sample)
            mse = torch.mean((reconstructed - input_tensor) ** 2, dim=1).numpy().tolist()

        # Prepare response
        result = {
            "reconstructed": reconstructed.numpy().flatten().tolist(),
            "reconstruction_error": mse
        }
        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": str(e)})