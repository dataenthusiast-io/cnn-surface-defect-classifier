"""FastAPI inference server.

Start:
    uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import DEVICE, API_HOST, API_PORT
from api.pipeline import InferencePipeline

app = FastAPI(title="CNN Defect Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: InferencePipeline | None = None


@app.on_event("startup")
def startup() -> None:
    global pipeline
    pipeline = InferencePipeline()
    print(f"InferencePipeline loaded. Test-set size: {len(pipeline.dataset)}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model":  "best_model.pth",
        "device": DEVICE,
        "test_size": len(pipeline.dataset) if pipeline else 0,
    }


@app.get("/predict/next")
def predict_next():
    return pipeline.next()


@app.get("/predict/reset")
def predict_reset():
    pipeline.reset()
    return {"status": "reset"}


@app.get("/stats")
def get_stats():
    return pipeline.get_stats()
