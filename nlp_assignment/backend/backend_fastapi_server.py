"""
Project Sentinel - FastAPI Backend Server
Real-time WebSocket streaming for hallucination detection and causal intervention
"""

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import json
import torch
from typing import Dict, List, Optional
from pydantic import BaseModel
import logging

# Import inference engine
from backend_inference_engine import SentinelInferenceEngine, TokenAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Project Sentinel",
    description="Real-time Hallucination Detection & Causal Intervention",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference engine (global)
engine: Optional[SentinelInferenceEngine] = None


# Pydantic models
class GenerateRequest(BaseModel):
    """Request model for text generation"""
    prompt: str
    max_new_tokens: int = 20
    temperature: float = 0.7
    top_p: float = 0.9


class AnalysisResponse(BaseModel):
    """Response model for analysis"""
    token_id: int
    token_text: str
    drift_score: float
    is_hallucination: bool
    sigma: float
    confidence: float
    entropy: float
    peak_layer_id: int


class InterventionResponse(BaseModel):
    """Response model for intervention"""
    peak_layer_id: int
    steering_magnitude: float
    semantic_shift: float
    baseline_top_tokens: List[str]
    steered_top_tokens: List[str]


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup"""
    global engine
    logger.info("Initializing inference engine...")
    try:
        engine = SentinelInferenceEngine(
            model_name="facebook/opt-1.3b",
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        logger.info("Inference engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine_loaded": engine is not None,
        "device": str(engine.device) if engine else "N/A",
    }


# Analysis endpoint
@app.post("/api/analyze")
async def analyze(request: GenerateRequest):
    """Analyze text for hallucinations"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        result = engine.generate_response(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
        )
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Manifold endpoint
@app.get("/api/manifold")
async def get_manifold():
    """Get faithful manifold statistics"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return {
        "mean_shape": list(engine.faithful_mean.shape),
        "cov_shape": list(engine.faithful_cov.shape),
        "num_layers": engine.num_layers,
        "hidden_dim": engine.hidden_dim,
    }


# WebSocket endpoint for real-time streaming
@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """WebSocket endpoint for real-time token generation and analysis"""
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            prompt = request_data.get("prompt", "")
            max_new_tokens = request_data.get("max_new_tokens", 20)
            
            if not prompt:
                await websocket.send_json({
                    "type": "error",
                    "message": "Prompt is required"
                })
                continue
            
            logger.info(f"Generating response for prompt: {prompt}")
            
            # Generate response with streaming
            try:
                result = engine.generate_response(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )
                
                # Send token analyses
                for token_analysis in result['token_analyses']:
                    await websocket.send_json({
                        "type": "token",
                        "data": token_analysis,
                    })
                    # Add small delay to simulate streaming
                    await asyncio.sleep(0.1)
                
                # Send statistics
                await websocket.send_json({
                    "type": "statistics",
                    "data": result['statistics'],
                })
                
                # Send complete message
                await websocket.send_json({
                    "type": "complete",
                    "generated_text": result['generated_text'],
                })
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()
        logger.info("WebSocket client disconnected")


# Intervention endpoint
@app.post("/api/intervention")
async def apply_intervention(request: GenerateRequest):
    """Apply causal intervention to suppress hallucinations"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        # Generate response
        result = engine.generate_response(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
        )
        
        # Extract intervention results
        interventions = [
            a.get('intervention')
            for a in result['token_analyses']
            if a.get('intervention')
        ]
        
        return {
            "original_text": result['generated_text'],
            "interventions": interventions,
            "statistics": result['statistics'],
        }
    except Exception as e:
        logger.error(f"Intervention failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration endpoint
@app.get("/api/config")
async def get_config():
    """Get engine configuration"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return {
        "model_name": engine.model_name,
        "num_layers": engine.num_layers,
        "hidden_dim": engine.hidden_dim,
        "device": str(engine.device),
        "dtype": str(engine.dtype),
        "drift_threshold": 3.0,
        "warning_threshold": 2.0,
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Project Sentinel",
        "description": "Real-time Hallucination Detection & Causal Intervention",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/api/analyze",
            "manifold": "/api/manifold",
            "intervention": "/api/intervention",
            "config": "/api/config",
            "websocket": "/ws/generate",
        }
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
