# AI Model Lab V4.1 — Test Report

Data build: 26/09/2026

## Controlli statici
- `python -m py_compile backend/*.py`: OK
- `node --check js/*.js`: OK

## Self test
- `python -m backend.self_test`: OK
- `python -m backend.self_test_v4`: OK
- `python -m backend.regression_test_v41`: OK

## Regressione NaN / MLP
Configurazione verificata:
- input 1
- hidden 32,32
- output 1
- ReLU
- AdamW
- LR 0.001
- weight decay 0.0001
- gradient clip 1.0
- seed 42
- batch 32

Risultato nel runtime di build:
- parametri: 1.153
- step 1: HTTP 200
- step 51: HTTP 200
- JSON non contiene float `NaN`/`Infinity`
- loss step 1: 22.9234561920166
- loss step 51: 8.578986167907715

## Test HTTP reale
È stato avviato Uvicorn e sono state verificate chiamate reali:
- `GET /api/advanced/health` → 200
- `POST /api/advanced/mlp/create` → 200
- `POST /api/advanced/mlp/train` → 200

Nel computer dell'utente la build PyTorch/CUDA è stata già verificata separatamente con RTX 5050 Laptop GPU e `torch.cuda.is_available() == True`.
