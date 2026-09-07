import requests

# FastAPIが動いているURL（デフォルトは8000番）
url = "http://127.0.0.1:8000/predict"

# 送信するデータ
data = {
    "feature1": 10.5,
    "feature2": 5.0
}

# POSTリクエストを送信
response = requests.post(url, json=data)

# 結果の表示
if response.status_code == 200:
    result = response.json()
    print(f"Prediction Result: {result['prediction']}")
else:
    print(f"Error: {response.status_code}")