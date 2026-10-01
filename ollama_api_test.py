import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "qwen2.5:3b",
    "prompt": "Say hello in one sentence.",
    "stream": False,
}

response = requests.post(url, json=payload, timeout=120)
response.raise_for_status()
data = response.json()
print(data.get("response", "No response received"))
