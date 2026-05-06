import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "input": [25,60,170,1,7,2,0,1,0,22]
}

res = requests.post(url, json=data)

print("Status:", res.status_code)
print("Response:", res.text)