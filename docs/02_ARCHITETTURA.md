# 02 - Architettura del progetto

## Frontend
`index.html` + `styles.css` + moduli JavaScript. `engine.js` contiene l’MLP didattico; `advanced.js` generalizzazione/diagnostica; `industrial.js` V3 PyTorch; `ultimate.js` V4, Autograd Microscope, Scale Lab ed Experiment Tracker.

## Backend
FastAPI espone `/api/advanced/*` (V3 compatibile) e `/api/v4/*` (Ultimate).

### V4 model stack
`token ids → token embedding → [RMSNorm → RoPE GQA Attention → residual → RMSNorm → SwiGLU → residual] × N → RMSNorm → LM head → logits`

### File principali
- `backend/modern_models.py`: RMSNorm, RoPE, LoRALinear, GQA/SDPA, SwiGLU, ModernTransformerLM.
- `backend/tokenizer_bpe.py`: tokenizer Byte-BPE addestrabile.
- `backend/trainer_v4.py`: batching, cross entropy, autograd, accumulation, clipping, warmup/cosine LR, AMP, generation, background job.
- `backend/router_v4.py`: API V4, checkpoint, experiments e profiler.
- `backend/profiler_v4.py`: sistema e stime analitiche di scala.
- `backend/experiments_v4.py`: SQLite.

## Dati e privacy
Corpus, checkpoint e database restano sul computer locale salvo modifiche dell’utente. Le CDN del frontend richiedono Internet per Tailwind/Chart.js/FontAwesome.
