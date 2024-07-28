from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel, Field
import tensorflow as tf
import numpy as np
import os
from PIL import Image

app = FastAPI()

try:
    model = tf.keras.models.load_model('card.h5')
    print(model.summary())
except Exception as e:
    print(e)

class CardImage(BaseModel):
    image_dir: str = Field(..., alias='imageDir')
    file_names: List[str] = Field(..., alias='fileNames')

def preprocess_image(image_path: str) -> np.ndarray:
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        # 이미지 로드
        image = Image.open(image_path)
        # 이미지 전처리
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)  # 배치 차원 추가
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

@app.post("/extract/card")
async def extract(card: CardImage):
    print(card.image_dir)
    card_names = ['black', 'busik', 'byul', 'changing', 'dal', 'diun', 'dotae', 'gwangdae', 'gwangki', 'gyunhyun', 'hwanhui', 'royal', 'samdusa', 'simpan', 'undefined', 'unsu', 'xEmpty', 'yuryeong', 'zEmpty']
    results = []
    for image in card.file_names:
        file_name = image
        print(file_name)
        image = preprocess_image(card.image_dir + '\\' + image)

        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)
        print(card_names[predicted_class])
        results.append({"fileName": file_name, "cardName": card_names[predicted_class]})
    return results