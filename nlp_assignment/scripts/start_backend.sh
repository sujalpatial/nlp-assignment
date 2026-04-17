#!/bin/bash
cd backend
source venv/bin/activate
python -m uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload
