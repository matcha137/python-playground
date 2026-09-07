from fastapi.testclient import TestClient
from main import app  # main.pyからappインスタンスをインポート

client = TestClient(app)

# 1. ルートエンドポイントのテスト
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "ML Model API is running"}

# 2. 推論エンドポイントの成功テスト
def test_predict_success():
    payload = {"feature1": 10.0, "feature2": 5.0}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    # 計算ロジック (10.0 * 0.5) + (5.0 * 1.2) = 5.0 + 6.0 = 11.0
    assert data["prediction"] == 11.0

# 3. バリデーションエラーのテスト (不正なデータ型)
def test_predict_invalid_data():
    # feature1に文字列を送ってみる
    payload = {"feature1": "invalid", "feature2": 5.0}
    response = client.post("/predict", json=payload)

    # FastAPI(Pydantic)が自動で 422 Unprocessable Entity を返すはず
    assert response.status_code == 422