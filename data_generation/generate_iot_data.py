import numpy as np
import json
import time
from azure.iot.device import IoTHubDeviceClient, Message
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()
CONNECTION_STRING = os.getenv("IOT_HUB_CONNECTION_STRING")

def generate_iot_data(num_samples=10):
    client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
    
    client.connect()
    for _ in range(num_samples):
        temp = np.random.normal(loc=25, scale=2)
        if np.random.rand() < 0.05:
            temp += np.random.uniform(10, 15)
        data = {
            "device_id": "A1",
            "temperature": round(temp, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        message = Message(json.dumps(data))
        client.send_message(message)
        print(f"Sent: {json.dumps(data)}")
        time.sleep(1)
    client.disconnect()

if __name__ == "__main__":
    generate_iot_data()