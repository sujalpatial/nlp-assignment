"""Project Sentinel Configuration"""

import torch

# Model Configuration
MODEL_NAME = "facebook/opt-1.3b"
NUM_LAYERS = 24
HIDDEN_DIM = 2048
VOCAB_SIZE = 50257

# Inference Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_P = 0.9

# Hallucination Detection
DRIFT_THRESHOLD = 3.0
WARNING_THRESHOLD = 2.0

# PCA Steering
NUM_PCA_COMPONENTS = 10
STEERING_STRENGTH = 1.0

# Server Configuration
HOST = "0.0.0.0"
PORT = 8000
WORKERS = 1
LOG_LEVEL = "info"

# Paths
MANIFOLD_PATH = "data/faithful_manifold.pkl"
PCA_PATH = "data/pca_components.npy"
