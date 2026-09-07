from fastapi import FastAPI
from pydantic import BaseModel

# 1. APIの初期化
app = FastAPI()

# 2. リクエストデータの形式を定義（バリデーション）
class InputData(BaseModel):
    feature1: float
    feature2: float

@app.get("/")
def read_root():
    return {"message": "ML Model API is running"}

# 3. 推論エンドポイントの作成
@app.post("/predict")
def predict(data: InputData):
    # 本来はここで model.predict() を行う
    # 今回は例として簡単な数式でシミュレート
    prediction = (data.feature1 * 0.5) + (data.feature2 * 1.2)

    return {
        "status": "success",
        "prediction": prediction
    }