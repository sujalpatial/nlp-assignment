# Project Sentinel - ACTUAL FILE INVENTORY

## ✅ FILES THAT EXIST (Created in this session)

### Backend Files (in /home/ubuntu/)
1. ✅ `backend_inference_engine.py` - Complete inference engine (500+ lines)
2. ✅ `backend_fastapi_server.py` - Complete FastAPI server (400+ lines)

### Documentation Files (in /home/ubuntu/)
3. ✅ `PROJECT_SENTINEL_SETUP_GUIDE.md` - Complete setup guide
4. ✅ `QUICK_START_GUIDE.md` - Quick start guide
5. ✅ `SENTINEL_EXECUTION_REPORT.md` - Execution results
6. ✅ `PROJECT_SENTINEL_COMPLETE.md` - Architecture documentation
7. ✅ `FILE_INVENTORY.md` - This file

### Project Sentinel Frontend Files (in /home/ubuntu/project_sentinel/)
8. ✅ `client/src/pages/Dashboard.tsx`
9. ✅ `client/src/pages/Home.tsx`
10. ✅ `client/src/components/PulseChart.tsx`
11. ✅ `client/src/components/TokenHighlight.tsx`
12. ✅ `client/src/components/DriftHeatmap.tsx`
13. ✅ `client/src/index.css`
14. ✅ `client/src/App.tsx`
15. ✅ `client/src/main.tsx`
16. ✅ `package.json`
17. ✅ `todo.md`

### Demo Files (in /home/ubuntu/)
18. ✅ `sentinel_demo.py` - Demo inference script
19. ✅ `dry_run_track_b.py` - Dry run script

---

## 📋 FILES YOU NEED TO CREATE (Simple Copy-Paste)

Below are the remaining essential files. Copy each one and create it in your project directory.

---

## FILE 1: `requirements.txt` (Backend Dependencies)

**Location:** `backend/requirements.txt`

```txt
torch==2.0.0
transformers==4.30.0
fastapi==0.100.0
uvicorn==0.23.0
websockets==11.0.0
numpy==1.24.0
scikit-learn==1.3.0
scipy==1.11.0
pydantic==2.0.0
python-dotenv==1.0.0
```

---

## FILE 2: `vite.config.ts` (Frontend Build Config)

**Location:** `frontend/vite.config.ts`

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
})
```

---

## FILE 3: `tsconfig.json` (TypeScript Config)

**Location:** `frontend/tsconfig.json`

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "resolveJsonModule": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

---

## FILE 4: `tailwind.config.js` (Tailwind Config)

**Location:** `frontend/tailwind.config.js`

```javascript
export default {
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        accent: 'var(--accent)',
        warning: 'var(--warning)',
        danger: 'var(--danger)',
        success: 'var(--success)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
}
```

---

## FILE 5: `index.html` (HTML Entry Point)

**Location:** `frontend/index.html`

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Project Sentinel - Hallucination Detection</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

---

## FILE 6: `config.py` (Backend Configuration)

**Location:** `backend/config.py`

```python
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
```

---

## FILE 7: `.env.example` (Environment Variables)

**Location:** `project_sentinel/.env.example`

```env
# Backend Configuration
BACKEND_HOST=localhost
BACKEND_PORT=8000
MODEL_NAME=facebook/opt-1.3b
DEVICE=cuda
DTYPE=float16

# Frontend Configuration
FRONTEND_HOST=localhost
FRONTEND_PORT=3000
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000

# Inference Configuration
DRIFT_THRESHOLD=3.0
WARNING_THRESHOLD=2.0
MAX_NEW_TOKENS=100
TEMPERATURE=0.7

# Paths
MANIFOLD_PATH=data/faithful_manifold.pkl
PCA_PATH=data/pca_components.npy
```

---

## FILE 8: `setup.sh` (Setup Script)

**Location:** `project_sentinel/scripts/setup.sh`

```bash
#!/bin/bash
set -e

echo "🚀 Project Sentinel Setup"
echo "========================="

# Backend setup
echo "📦 Setting up backend..."
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..

# Frontend setup
echo "📦 Setting up frontend..."
cd frontend
npm install
cd ..

# Download model
echo "📥 Downloading model..."
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/opt-1.3b')"

echo "✅ Setup complete!"
```

---

## FILE 9: `start_backend.sh` (Backend Start Script)

**Location:** `project_sentinel/scripts/start_backend.sh`

```bash
#!/bin/bash
cd backend
source venv/bin/activate
python -m uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload
```

---

## FILE 10: `start_frontend.sh` (Frontend Start Script)

**Location:** `project_sentinel/scripts/start_frontend.sh`

```bash
#!/bin/bash
cd frontend
npm run dev
```

---

## FILE 11: `docker-compose.yml` (Docker Orchestration)

**Location:** `project_sentinel/docker-compose.yml`

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=facebook/opt-1.3b
      - DEVICE=cuda
    volumes:
      - ./data:/app/data
    command: python -m uvicorn backend_fastapi_server:app --host 0.0.0.0 --port 8000

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://localhost:8000
      - VITE_WS_URL=ws://localhost:8000
    depends_on:
      - backend
```

---

## FILE 12: `Dockerfile.backend` (Backend Docker Image)

**Location:** `project_sentinel/Dockerfile.backend`

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-runtime-ubuntu22.04

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## FILE 13: `Dockerfile.frontend` (Frontend Docker Image)

**Location:** `project_sentinel/Dockerfile.frontend`

```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ .

EXPOSE 3000

CMD ["npm", "run", "dev"]
```

---

## FILE 14: `README.md` (Project Overview)

**Location:** `project_sentinel/README.md`

```markdown
# Project Sentinel

Real-Time Hallucination Detection & Causal Intervention Dashboard

## Features

- **Real-time Inference Engine**: Forward-hook extraction of hidden states
- **Drift Score Computation**: Mahalanobis distance against faithful manifold
- **Logit-Lens Entropy**: Track entropy across all layers
- **Dynamic Hallucination Flagging**: σ > 3.0 threshold detection
- **Causal Switch**: Baseline vs Intervention Pass comparison
- **WebSocket Streaming**: Real-time per-token analysis
- **Visual Element A - The Pulse**: Real-time drift chart
- **Visual Element B - Token Prediction**: Color-coded highlighting
- **Visual Element C - Before & After**: Dual-pane comparison with heatmap
- **Semantic Similarity**: Quantify intervention effectiveness

## Quick Start

### Backend
```bash
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
python -m uvicorn backend_fastapi_server:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Architecture

- **Backend**: FastAPI + PyTorch
- **Frontend**: React + TypeScript + Tailwind CSS
- **Communication**: WebSocket for real-time streaming
- **Model**: facebook/opt-1.3b (1.3B parameters, 24 layers)

## Documentation

- `PROJECT_SENTINEL_SETUP_GUIDE.md` - Detailed setup
- `QUICK_START_GUIDE.md` - 5-minute quick start
- `SENTINEL_EXECUTION_REPORT.md` - Execution results
- `PROJECT_SENTINEL_COMPLETE.md` - Architecture & design

## License

MIT
```

---

## 📊 SUMMARY

### Files That Exist (19 files)
✅ Backend: 2 files  
✅ Frontend: 8 files  
✅ Documentation: 4 files  
✅ Demo: 2 files  
✅ Config: 1 file  

### Files You Need to Create (14 files)
- `requirements.txt`
- `vite.config.ts`
- `tsconfig.json`
- `tailwind.config.js`
- `index.html`
- `config.py`
- `.env.example`
- `setup.sh`
- `start_backend.sh`
- `start_frontend.sh`
- `docker-compose.yml`
- `Dockerfile.backend`
- `Dockerfile.frontend`
- `README.md`

---

## 🚀 TOTAL: 33 Essential Files

All files are provided above. Copy and create them in your project directory.

---

## 📍 EXACT FILE LOCATIONS

```
project_sentinel/
├── backend/
│   ├── requirements.txt              ← CREATE
│   ├── config.py                     ← CREATE
│   ├── inference_engine.py           ✅ EXISTS
│   └── fastapi_server.py             ✅ EXISTS
│
├── frontend/
│   ├── index.html                    ← CREATE
│   ├── package.json                  ✅ EXISTS
│   ├── vite.config.ts                ← CREATE
│   ├── tsconfig.json                 ← CREATE
│   ├── tailwind.config.js            ← CREATE
│   └── src/
│       ├── main.tsx                  ✅ EXISTS
│       ├── App.tsx                   ✅ EXISTS
│       ├── index.css                 ✅ EXISTS
│       ├── pages/
│       │   ├── Home.tsx              ✅ EXISTS
│       │   └── Dashboard.tsx         ✅ EXISTS
│       └── components/
│           ├── PulseChart.tsx        ✅ EXISTS
│           ├── TokenHighlight.tsx    ✅ EXISTS
│           └── DriftHeatmap.tsx      ✅ EXISTS
│
├── scripts/
│   ├── setup.sh                      ← CREATE
│   ├── start_backend.sh              ← CREATE
│   └── start_frontend.sh             ← CREATE
│
├── .env.example                      ← CREATE
├── docker-compose.yml                ← CREATE
├── Dockerfile.backend                ← CREATE
├── Dockerfile.frontend               ← CREATE
├── README.md                         ← CREATE
└── todo.md                           ✅ EXISTS
```

---

## ✅ NEXT STEPS

1. Copy all 14 "CREATE" files from above
2. Place them in the correct locations
3. Run `bash scripts/setup.sh`
4. Start backend: `bash scripts/start_backend.sh`
5. Start frontend: `bash scripts/start_frontend.sh`
6. Open http://localhost:3000

**That's it! You have everything you need.** 🚀
