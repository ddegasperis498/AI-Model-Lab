from __future__ import annotations
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .schemas import (
    MLPConfig, MLPTrainRequest, MLPPredictRequest,
    LLMConfig, LLMTrainRequest, LLMGenerateRequest,
    CheckpointRequest, CheckpointLoadRequest,
)
from .trainer import MLPTrainer, LLMTrainer
from .router_v4 import router as v4_router
from .safe_json import sanitize_for_json

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / 'backend' / 'checkpoints'
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

class SafeJSONResponse(JSONResponse):
    def render(self, content):
        return super().render(sanitize_for_json(content))


app = FastAPI(
    title='AI Model Lab V4.1 — Ultimate Dual Engine Backend',
    version='4.1',
    default_response_class=SafeJSONResponse,
)


@app.middleware('http')
async def correlation_id_middleware(request: Request, call_next):
    correlation_id = request.headers.get('X-Correlation-ID') or uuid.uuid4().hex
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers['X-Correlation-ID'] = correlation_id
    # AI Model Lab is a local development UI: never serve stale frontend assets.
    if request.url.path == '/' or request.url.path.endswith(('.html', '.js', '.css', '.svg')):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    cid = getattr(request.state, 'correlation_id', uuid.uuid4().hex)
    detail = exc.detail
    message = detail if isinstance(detail, str) else 'Richiesta non completata'
    return SafeJSONResponse(
        status_code=exc.status_code,
        content={
            'ok': False,
            'error': {
                'code': f'HTTP_{exc.status_code}',
                'message': message,
                'detail': detail,
                'operation': f'{request.method} {request.url.path}',
                'correlation_id': cid,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            },
        },
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    cid = getattr(request.state, 'correlation_id', uuid.uuid4().hex)
    # The traceback is still emitted by the server logger; do not expose it to normal users.
    return SafeJSONResponse(
        status_code=500,
        content={
            'ok': False,
            'error': {
                'code': 'INTERNAL_SERVER_ERROR',
                'message': 'Errore interno del backend AI Model Lab',
                'detail': str(exc),
                'operation': f'{request.method} {request.url.path}',
                'correlation_id': cid,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            },
        },
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], allow_credentials=False,
    allow_methods=['*'], allow_headers=['*'],
)

lock = RLock()
mlp: Optional[MLPTrainer] = None
llm: Optional[LLMTrainer] = None
current_kind: Optional[str] = None


def safe_name(name: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('._')
    return (cleaned or 'checkpoint')[:80]


@app.get('/api/advanced/health')
def health():
    cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda else 'CPU'
    return {
        'ok': True,
        'pytorch': torch.__version__,
        'cuda_available': cuda,
        'cuda_version': torch.version.cuda,
        'device_name': device_name,
        'cpu_threads': torch.get_num_threads(),
        'current_kind': current_kind,
    }


@app.post('/api/advanced/mlp/create')
def create_mlp(config: MLPConfig):
    global mlp, current_kind
    with lock:
        try:
            mlp = MLPTrainer(config)
            current_kind = 'mlp'
            return mlp.state()
        except Exception as exc:
            raise HTTPException(400, str(exc))


@app.post('/api/advanced/mlp/train')
def train_mlp(req: MLPTrainRequest):
    if mlp is None:
        raise HTTPException(409, 'Crea prima il modello MLP')
    with lock:
        try:
            data = [s.model_dump() for s in req.data]
            return mlp.train(data, req.batch_size, req.steps)
        except Exception as exc:
            raise HTTPException(400, str(exc))


@app.post('/api/advanced/mlp/predict')
def predict_mlp(req: MLPPredictRequest):
    if mlp is None:
        raise HTTPException(409, 'Crea prima il modello MLP')
    with lock:
        try:
            return {'predictions': mlp.predict(req.x)}
        except Exception as exc:
            raise HTTPException(400, str(exc))


@app.get('/api/advanced/mlp/state')
def state_mlp():
    if mlp is None:
        raise HTTPException(404, 'MLP non creato')
    return mlp.state()


@app.post('/api/advanced/llm/create')
def create_llm(config: LLMConfig):
    global llm, current_kind
    with lock:
        try:
            llm = LLMTrainer(config)
            current_kind = 'llm'
            return llm.state()
        except Exception as exc:
            raise HTTPException(400, str(exc))


@app.post('/api/advanced/llm/train')
def train_llm(req: LLMTrainRequest):
    if llm is None:
        raise HTTPException(409, 'Crea prima il Transformer LM')
    with lock:
        try:
            return llm.train(req.text, req.batch_size, req.steps)
        except Exception as exc:
            raise HTTPException(400, str(exc))


@app.post('/api/advanced/llm/generate')
def generate_llm(req: LLMGenerateRequest):
    if llm is None:
        raise HTTPException(409, 'Crea prima il Transformer LM')
    with lock:
        try:
            return llm.generate(req.prompt, req.max_new_tokens, req.temperature, req.top_k)
        except Exception as exc:
            raise HTTPException(400, str(exc))


@app.get('/api/advanced/llm/state')
def state_llm():
    if llm is None:
        raise HTTPException(404, 'Transformer LM non creato')
    return llm.state()


@app.post('/api/advanced/checkpoint/save')
def save_checkpoint(req: CheckpointRequest):
    trainer = llm if current_kind == 'llm' else mlp
    if trainer is None:
        raise HTTPException(409, 'Nessun modello attivo')
    name = safe_name(req.name)
    filename = f'{name}_{current_kind}.pt'
    path = CHECKPOINTS / filename
    payload = {
        'kind': current_kind,
        'config': trainer.config.model_dump(),
        'step': trainer.step,
        'model_state': trainer.model.state_dict(),
        'optimizer_state': trainer.optimizer.state_dict(),
    }
    torch.save(payload, path)
    return {'ok': True, 'filename': filename, 'size_bytes': path.stat().st_size}


@app.get('/api/advanced/checkpoint/list')
def list_checkpoints():
    return {'items': [
        {'filename': p.name, 'size_bytes': p.stat().st_size, 'mtime': p.stat().st_mtime}
        for p in sorted(CHECKPOINTS.glob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    ]}


@app.post('/api/advanced/checkpoint/load')
def load_checkpoint(req: CheckpointLoadRequest):
    global mlp, llm, current_kind
    name = safe_name(req.filename)
    path = CHECKPOINTS / name
    if not path.exists() or path.parent != CHECKPOINTS:
        raise HTTPException(404, 'Checkpoint non trovato')
    with lock:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        kind = payload.get('kind')
        if kind == 'mlp':
            trainer = MLPTrainer(MLPConfig(**payload['config']))
            mlp = trainer
        elif kind == 'llm':
            trainer = LLMTrainer(LLMConfig(**payload['config']))
            llm = trainer
        else:
            raise HTTPException(400, 'Tipo checkpoint non valido')
        trainer.model.load_state_dict(payload['model_state'])
        trainer.optimizer.load_state_dict(payload['optimizer_state'])
        trainer.step = int(payload.get('step', 0))
        current_kind = kind
        return trainer.state()


app.include_router(v4_router)

# API routes must be declared before this mount.
app.mount('/', StaticFiles(directory=str(ROOT), html=True), name='frontend')
