import torch

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
try:
    response = client.ping()  # Assuming `ping()` or similar exists for a health check
    print("API key is valid:", response)
except Exception as e:
    print("Error:", e)
