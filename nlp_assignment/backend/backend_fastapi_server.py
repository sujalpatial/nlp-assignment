"""
Project Sentinel FastAPI Backend — Fixed Version
Real-time WebSocket streaming for mechanistic interpretability analysis
"""

import asyncio
import json
import logging
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference_engine import (
    InferenceEngine,
    DriftAnalysis,
    InterventionResult,
    create_mock_faithful_activations,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Project Sentinel", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_engine: Optional[InferenceEngine] = None
detector_data: Optional[dict] = None
DETECTOR_PATH = os.environ.get("DETECTOR_PATH", "data/detector.pkl")


class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.7


class TokenAnalysisMessage(BaseModel):
    token_id: int
    token_text: str
    drift_score: float
    is_hallucination: bool
    entropy: float
    logit_lens_entropy: List[float]
    layer_id: int
    p_halluc: float = 0.0
    color: str = "green"


class InterventionMessage(BaseModel):
    peak_layer_id: int
    steering_magnitude: float
    semantic_shift: float
    baseline_top_tokens: List[tuple]
    steered_top_tokens: List[tuple]


@app.on_event("startup")
async def startup_event():
    global inference_engine, detector_data
    logger.info("Initializing Project Sentinel...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference_engine = InferenceEngine(device=device)
    if os.path.exists(DETECTOR_PATH):
        try:
            with open(DETECTOR_PATH, "rb") as f:
                detector_data = pickle.load(f)
            faithful_activations = detector_data.get("faithful_mean_hs", {})
            if faithful_activations:
                inference_engine.initialize_manifold(faithful_activations)
                logger.info("Real detector loaded. AUROC=%.4f", detector_data.get("auroc", 0))
            else:
                _init_mock_manifold()
        except Exception as e:
            logger.warning("Failed to load detector.pkl: %s. Using demo mode.", e)
            _init_mock_manifold()
    else:
        logger.warning("detector.pkl not found. Running in DEMO mode. Run: python train_detector.py")
        _init_mock_manifold()
    logger.info("Startup complete. Device=%s", device)


def _init_mock_manifold():
    faithful_activations = create_mock_faithful_activations(
        n_samples=500,
        hidden_dim=inference_engine.model.config.hidden_size,
        n_layers=inference_engine.n_layers,
    )
    inference_engine.initialize_manifold(faithful_activations)
    logger.info("Mock manifold initialised (demo mode)")


def _halluc_probability(drift_score: float) -> float:
    return float(1 / (1 + np.exp(-0.9 * (drift_score - 2.5))))


def _color(p: float) -> str:
    if p < 0.35: return "green"
    if p < 0.65: return "yellow"
    return "red"


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "engine_ready": inference_engine is not None,
        "detector_loaded": detector_data is not None,
        "mode": "real" if (detector_data and "gbc" in detector_data) else "demo",
        "auroc": detector_data.get("auroc") if detector_data else None,
    }


@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        request = GenerationRequest(**json.loads(raw))
        logger.info("Generation request: '%s...'", request.prompt[:60])
        if not inference_engine:
            await websocket.send_json({"type": "error", "message": "Engine not initialised"})
            await websocket.close()
            return
        generated_text, analyses = inference_engine.generate_with_analysis(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        for analysis in analyses:
            p = _halluc_probability(analysis.drift_score)
            msg = TokenAnalysisMessage(
                token_id=analysis.token_id,
                token_text=analysis.token_text,
                drift_score=analysis.drift_score,
                is_hallucination=analysis.is_hallucination,
                entropy=analysis.entropy,
                logit_lens_entropy=analysis.logit_lens_entropy,
                layer_id=analysis.layer_id,
                p_halluc=p,
                color=_color(p),
            )
            await websocket.send_json({"type": "token", "data": msg.dict()})
            await asyncio.sleep(0.05)
        if analyses:
            peak_layer = analyses[int(np.argmax([a.drift_score for a in analyses]))].layer_id
            try:
                result = inference_engine.causal_intervention(peak_layer)
                baseline_top = torch.topk(result.baseline_logits, 5)
                steered_top  = torch.topk(result.steered_logits, 5)
                intervention_msg = InterventionMessage(
                    peak_layer_id=peak_layer,
                    steering_magnitude=result.steering_magnitude,
                    semantic_shift=result.semantic_shift,
                    baseline_top_tokens=[
                        (inference_engine.tokenizer.decode([idx.item()]), prob.item())
                        for idx, prob in zip(baseline_top.indices, baseline_top.values)
                    ],
                    steered_top_tokens=[
                        (inference_engine.tokenizer.decode([idx.item()]), prob.item())
                        for idx, prob in zip(steered_top.indices, steered_top.values)
                    ],
                )
                await websocket.send_json({"type": "intervention", "data": intervention_msg.dict()})
            except Exception as e:
                logger.error("Intervention failed: %s", e)
                await websocket.send_json({"type": "intervention_error", "message": str(e)})
        await websocket.send_json({"type": "complete", "generated_text": generated_text})
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        finally:
            await websocket.close()


@app.post("/api/analyze-token")
async def analyze_token_endpoint(token_id: int, token_text: str):
    if not inference_engine:
        return {"error": "Engine not initialised"}
    try:
        a = inference_engine.analyze_token(token_id, token_text)
        p = _halluc_probability(a.drift_score)
        return {"token_id": a.token_id, "token_text": a.token_text,
                "drift_score": a.drift_score, "is_hallucination": a.is_hallucination,
                "entropy": a.entropy, "p_halluc": p, "color": _color(p)}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/intervention")
async def intervention_endpoint(peak_layer_id: int):
    if not inference_engine:
        return {"error": "Engine not initialised"}
    try:
        r = inference_engine.causal_intervention(peak_layer_id)
        return {
            "steering_magnitude": r.steering_magnitude,
            "semantic_shift": r.semantic_shift,
            "baseline_entropy": float(torch.sum(-r.baseline_probs * torch.log(r.baseline_probs + 1e-10))),
            "steered_entropy":  float(torch.sum(-r.steered_probs  * torch.log(r.steered_probs  + 1e-10))),
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
