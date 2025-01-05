import requests

query = "List at max 5 people with the tag: Architecture and Construction located in Arunachal Pradesh"

response = requests.get("http://127.0.0.1:5000/search-engine",params={'query':query})

print(response.json())