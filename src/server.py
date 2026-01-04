from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv
import base64

app = FastAPI()
model = YOLO("yolov8n.pt")

class ImagePayload(BaseModel):
    image: str  # base64 encoded image
    width: int
    height: int

@app.post("/detect")
def detect(payload: ImagePayload):
    image_bytes = base64.b64decode(payload.image)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    output = []
    for box, cls, conf in zip(
        detections.xyxy,
        detections.class_id,
        detections.confidence
    ):
        output.append({
            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3]),
            "class": int(cls),
            "confidence": float(conf)
        })

    return {"detections": output}