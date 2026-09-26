from __future__ import annotations
import re
from pathlib import Path
from threading import RLock
from typing import Optional

import torch
from fastapi import APIRouter, HTTPException

from .experiments_v4 import ExperimentStore
from .profiler_v4 import analytical_decoder_estimate, system_snapshot
from .schemas_v4 import (
    V4CheckpointLoadRequest, V4CheckpointRequest, V4CreateRequest,
    V4ExperimentSaveRequest, V4GenerateRequest, V4JobRequest,
    V4ScaleRequest, V4TokenizerPreviewRequest, V4TrainRequest,
)
from .trainer_v4 import BackgroundTrainingJob, UltimateLMTrainer

ROOT = Path(__file__).resolve().parents[1]
V4_CHECKPOINTS = ROOT / 'backend' / 'checkpoints_v4'
V4_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
STORE = ExperimentStore(ROOT / 'backend' / 'experiments_v4.sqlite3')

router = APIRouter(prefix='/api/v4', tags=['V4 Ultimate Framework'])
lock = RLock()
trainer: Optional[UltimateLMTrainer] = None
job = BackgroundTrainingJob()


def safe_name(name: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('._')
    return (cleaned or 'ultimate')[:120]


@router.get('/health')
def health():
    return {
        'ok': True,
        'version': '4.0',
        'torch': torch.__version__,
        'current_model': trainer is not None,
        'job': job.state(),
        'system': system_snapshot(),
        'features': [
            'RMSNorm','RoPE','SwiGLU','Grouped Query Attention','PyTorch SDPA',
            'LoRA','gradient accumulation','cosine LR + warmup','gradient checkpointing',
            'mixed precision','background training','BPE tokenizer','checkpointing',
            'experiment tracking','embedding similarity','scale profiler',
        ],
    }


@router.post('/lm/create')
def create(req: V4CreateRequest):
    global trainer
    if job.running:
        raise HTTPException(409, 'Ferma prima il job di training attivo')
    with lock:
        try:
            trainer = UltimateLMTrainer(req.config, req.corpus, req.validation_fraction)
            return trainer.state()
        except Exception as exc:
            raise HTTPException(400, str(exc))


@router.get('/lm/state')
def state():
    if trainer is None:
        raise HTTPException(404, 'Crea prima il modello V4')
    return trainer.state(extra={'job': job.state()})


@router.post('/lm/train')
def train_sync(req: V4TrainRequest):
    if trainer is None:
        raise HTTPException(409, 'Crea prima il modello V4')
    if job.running:
        raise HTTPException(409, 'È già attivo un job in background')
    try:
        return trainer.train(req.steps, req.batch_size, req.eval_interval)
    except Exception as exc:
        raise HTTPException(400, str(exc))


@router.post('/lm/job/start')
def start_job(req: V4JobRequest):
    if trainer is None:
        raise HTTPException(409, 'Crea prima il modello V4')
    try:
        job.start(trainer, req.steps, req.batch_size, req.eval_interval)
        return job.state()
    except Exception as exc:
        raise HTTPException(409, str(exc))


@router.get('/lm/job/status')
def job_status():
    payload = job.state()
    if trainer is not None:
        payload['trainer_step'] = trainer.step
        payload['last'] = trainer.last
    return payload


@router.post('/lm/job/stop')
def stop_job():
    job.stop()
    return job.state()


@router.post('/lm/generate')
def generate(req: V4GenerateRequest):
    if trainer is None:
        raise HTTPException(409, 'Crea prima il modello V4')
    if job.running:
        raise HTTPException(409, 'Ferma il job prima della generazione per evitare contesa sul modello')
    try:
        return trainer.generate(
            req.prompt, req.max_new_tokens, req.temperature, req.top_k,
            req.top_p, req.repetition_penalty, req.seed,
        )
    except Exception as exc:
        raise HTTPException(400, str(exc))


@router.post('/tokenizer/preview')
def tokenizer_preview(req: V4TokenizerPreviewRequest):
    if trainer is None:
        raise HTTPException(409, 'Crea prima il modello V4')
    return {
        'kind': trainer.config.tokenizer,
        'vocab_size': trainer.tokenizer.vocab_size,
        'tokens': trainer.tokenizer_preview(req.text, req.limit),
    }


@router.post('/embedding/similarity')
def embedding_similarity(req: V4TokenizerPreviewRequest):
    if trainer is None:
        raise HTTPException(409, 'Crea prima il modello V4')
    return trainer.embedding_similarity(req.text, min(req.limit, 32))


@router.post('/checkpoint/save')
def checkpoint_save(req: V4CheckpointRequest):
    if trainer is None:
        raise HTTPException(409, 'Nessun modello V4 attivo')
    if job.running:
        raise HTTPException(409, 'Ferma il job prima di salvare il checkpoint')
    name = safe_name(req.name)
    filename = f'{name}_v4.pt'
    path = V4_CHECKPOINTS / filename
    torch.save(trainer.checkpoint_payload(), path)
    return {'ok': True, 'filename': filename, 'size_bytes': path.stat().st_size}


@router.get('/checkpoint/list')
def checkpoint_list():
    return {'items': [
        {'filename': p.name, 'size_bytes': p.stat().st_size, 'mtime': p.stat().st_mtime}
        for p in sorted(V4_CHECKPOINTS.glob('*.pt'), key=lambda x: x.stat().st_mtime, reverse=True)
    ]}


@router.post('/checkpoint/load')
def checkpoint_load(req: V4CheckpointLoadRequest):
    global trainer
    if job.running:
        raise HTTPException(409, 'Ferma prima il job di training')
    name = safe_name(req.filename)
    path = V4_CHECKPOINTS / name
    if not path.exists() or path.parent != V4_CHECKPOINTS:
        raise HTTPException(404, 'Checkpoint V4 non trovato')
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        trainer = UltimateLMTrainer.from_checkpoint(payload)
        return trainer.state()
    except Exception as exc:
        raise HTTPException(400, str(exc))


@router.post('/experiment/save')
def experiment_save(req: V4ExperimentSaveRequest):
    if trainer is None:
        raise HTTPException(409, 'Nessun modello V4 attivo')
    metrics = {
        'step': trainer.step,
        'tokens_seen': trainer.tokens_seen,
        'last': trainer.last,
        'params': trainer.state()['params'],
    }
    experiment_id = STORE.save(req.name, 'modern_transformer_lm', trainer.config.model_dump(), metrics, req.notes)
    return {'ok': True, 'id': experiment_id}


@router.get('/experiment/list')
def experiment_list(limit: int = 100):
    return {'items': STORE.list(min(max(1, limit), 500))}


@router.delete('/experiment/{experiment_id}')
def experiment_delete(experiment_id: int):
    STORE.delete(experiment_id)
    return {'ok': True}


@router.post('/scale/estimate')
def scale_estimate(req: V4ScaleRequest):
    if req.d_model % req.n_heads != 0:
        raise HTTPException(400, 'd_model deve essere divisibile per n_heads')
    if req.n_heads % req.n_kv_heads != 0:
        raise HTTPException(400, 'n_heads deve essere divisibile per n_kv_heads')
    return analytical_decoder_estimate(**req.model_dump())
