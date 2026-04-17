"""
Project Sentinel FastAPI Backend
Real-time WebSocket streaming for mechanistic interpretability analysis
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import torch

from inference_engine import (
    InferenceEngine,
    DriftAnalysis,
    InterventionResult,
    create_mock_faithful_activations
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# FastAPI app
app = FastAPI(title="Project Sentinel", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine instance
inference_engine: Optional[InferenceEngine] = None


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.7


class TokenAnalysisMessage(BaseModel):
    """WebSocket message for per-token analysis."""
    token_id: int
    token_text: str
    drift_score: float
    is_hallucination: bool
    entropy: float
    logit_lens_entropy: List[float]
    layer_id: int


class InterventionMessage(BaseModel):
    """WebSocket message for causal intervention results."""
    peak_layer_id: int
    steering_magnitude: float
    semantic_shift: float
    baseline_top_tokens: List[tuple]
    steered_top_tokens: List[tuple]


@app.on_event("startup")
async def startup_event():
    """Initialize the inference engine on startup."""
    global inference_engine
    logger.info("Initializing inference engine...")
    
    try:
        # Initialize engine (uses OPT-1.3b by default)
        inference_engine = InferenceEngine(device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize with mock faithful activations (in production, use real data)
        faithful_activations = create_mock_faithful_activations(
            n_samples=1000,
            hidden_dim=inference_engine.model.config.hidden_size,
            n_layers=inference_engine.n_layers
        )
        inference_engine.initialize_manifold(faithful_activations)
        
        logger.info("Inference engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine_ready": inference_engine is not None
    }


@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """
    WebSocket endpoint for real-time text generation with drift analysis.
    
    Protocol:
    1. Client sends: {"prompt": "...", "max_new_tokens": 50, "temperature": 0.7}
    2. Server streams: {"type": "token", "data": TokenAnalysisMessage}
    3. Server streams: {"type": "intervention", "data": InterventionMessage}
    4. Server sends: {"type": "complete", "generated_text": "..."}
    """
    await websocket.accept()
    
    try:
        # Receive generation request
        data = await websocket.receive_text()
        request = GenerationRequest(**json.loads(data))
        
        logger.info(f"Received generation request: prompt='{request.prompt[:50]}...'")
        
        if not inference_engine:
            await websocket.send_json({"type": "error", "message": "Engine not initialized"})
            await websocket.close()
            return
        
        # Generate with analysis
        generated_text, analyses = inference_engine.generate_with_analysis(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        )
        
        # Stream token analyses
        for analysis in analyses:
            token_msg = TokenAnalysisMessage(
                token_id=analysis.token_id,
                token_text=analysis.token_text,
                drift_score=analysis.drift_score,
                is_hallucination=analysis.is_hallucination,
                entropy=analysis.entropy,
                logit_lens_entropy=analysis.logit_lens_entropy,
                layer_id=analysis.layer_id
            )
            
            await websocket.send_json({
                "type": "token",
                "data": token_msg.dict()
            })
            
            # Small delay to simulate streaming
            await asyncio.sleep(0.05)
        
        # Perform causal intervention on peak drift layer
        if analyses:
            # Find peak drift layer
            max_drift_idx = np.argmax([a.drift_score for a in analyses])
            peak_layer = analyses[max_drift_idx].layer_id
            
            try:
                intervention_result = inference_engine.causal_intervention(peak_layer)
                
                # Get top tokens for both baseline and steered
                baseline_top = torch.topk(intervention_result.baseline_logits, 5)
                steered_top = torch.topk(intervention_result.steered_logits, 5)
                
                baseline_tokens = [
                    (inference_engine.tokenizer.decode([idx.item()]), prob.item())
                    for idx, prob in zip(baseline_top.indices, baseline_top.values)
                ]
                steered_tokens = [
                    (inference_engine.tokenizer.decode([idx.item()]), prob.item())
                    for idx, prob in zip(steered_top.indices, steered_top.values)
                ]
                
                intervention_msg = InterventionMessage(
                    peak_layer_id=peak_layer,
                    steering_magnitude=intervention_result.steering_magnitude,
                    semantic_shift=intervention_result.semantic_shift,
                    baseline_top_tokens=baseline_tokens,
                    steered_top_tokens=steered_tokens
                )
                
                await websocket.send_json({
                    "type": "intervention",
                    "data": intervention_msg.dict()
                })
            except Exception as e:
                logger.error(f"Intervention failed: {e}")
                await websocket.send_json({
                    "type": "intervention_error",
                    "message": str(e)
                })
        
        # Send completion message
        await websocket.send_json({
            "type": "complete",
            "generated_text": generated_text
        })
        
        logger.info("Generation completed successfully")
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
        finally:
            await websocket.close()


@app.post("/api/analyze-token")
async def analyze_token(token_id: int, token_text: str):
    """
    Synchronous endpoint for analyzing a single token.
    (Used for testing; real-time analysis happens via WebSocket)
    """
    if not inference_engine:
        return {"error": "Engine not initialized"}
    
    try:
        analysis = inference_engine.analyze_token(token_id, token_text)
        return {
            "token_id": analysis.token_id,
            "token_text": analysis.token_text,
            "drift_score": analysis.drift_score,
            "is_hallucination": analysis.is_hallucination,
            "entropy": analysis.entropy,
            "logit_lens_entropy": analysis.logit_lens_entropy,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/intervention")
async def perform_intervention(peak_layer_id: int):
    """
    Synchronous endpoint for causal intervention.
    (Used for testing; real-time intervention happens via WebSocket)
    """
    if not inference_engine:
        return {"error": "Engine not initialized"}
    
    try:
        result = inference_engine.causal_intervention(peak_layer_id)
        return {
            "steering_magnitude": result.steering_magnitude,
            "semantic_shift": result.semantic_shift,
            "baseline_entropy": float(torch.sum(-result.baseline_probs * torch.log(result.baseline_probs + 1e-10))),
            "steered_entropy": float(torch.sum(-result.steered_probs * torch.log(result.steered_probs + 1e-10))),
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    
    # Run on port 8000 (backend)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
