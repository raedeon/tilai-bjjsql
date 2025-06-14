from fastapi import FastAPI, Request
from ocr_manager import OCRManager

app = FastAPI()
manager = OCRManager(device_idx=0)

@app.post("/ocr")
async def ocr(request: Request) -> dict[str, list[str]]:
    inputs_json = await request.json()
    images_b64 = [instance["b64"] for instance in inputs_json.get("instances", [])]
    predictions = manager.batch_ocr(images_b64)
    return {"predictions": predictions}

@app.get("/health")
def health() -> dict[str, str]:
    if hasattr(manager, 'predictor') and manager.predictor is not None:
        return {"message": "health ok", "model_status": "loaded"}
    else:
        return {"message": "unhealthy", "model_status": "error or not loaded"}