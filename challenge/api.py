import fastapi
try:
    from challenge.model import ClassificatorModel
except ImportError:
    from model import ClassificatorModel
import pandas as pd
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

import logging

app = fastapi.FastAPI()

model = ClassificatorModel()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors()}
    )


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(request: dict) -> dict:
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame(request["data"])
        # Preprocess the data
        features = model.preprocess(data)
        # Predict the classes
        predictions = model.predict(features)
        # Decode predictions to original class labels
        if model._class_le:
            decoded_predictions = model._class_le.inverse_transform(predictions)
        else:
            decoded_predictions = predictions
    
    except Exception as e:
        # Log the error to view details in the server logs
        logging.error(f"Invalid format for predict Class data. Error to process: {e}")
        # Return a descriptive error message
        raise HTTPException(status_code=400, detail=str(e))
    
    #return {"predict": predictions}
    return {"prediction": decoded_predictions.tolist()}