# 06 - Troubleshooting

## Backend offline
Avvia `START_ULTIMATE_V4.bat`. Non è necessario avviare anche `START_ADVANCED_PYTORCH.bat`: V4.1 include già le API Advanced.

## Porta 8765 già in uso
Lo starter V4.1 rileva automaticamente una istanza già attiva e non tenta un secondo bind. Se la porta è occupata da un processo diverso, chiudi quel processo o cambia porta in modo coerente nel backend/frontend.

## `500 Internal Server Error` con `NaN`
V4.1 corregge questo caso con:
- statistiche `std(unbiased=False)`;
- controllo `torch.isfinite`;
- fallback FP32 in caso di AMP non finito;
- sanitizzazione JSON ricorsiva (`NaN`/`±Inf` → `null`).
Esegui `RUN_V41_REGRESSION_TESTS.bat` per verificare.

## CUDA non disponibile
Verifica `nvidia-smi` e poi:
`python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"`

## Out of memory
Riduci batch/context/d_model/layer; abilita gradient checkpointing; usa LoRA/freeze e mixed precision compatibile.

## Error dialog
Le operazioni fallite mostrano un modal con status HTTP, endpoint e correlation ID. Usa **Copia dettagli** per condividere rapidamente l'errore. Il traceback completo rimane server-side.

## 404 `/favicon.ico`
Risolto in V4.1 con `favicon.svg`. Le richieste Chrome `.well-known/appspecific/...` possono essere ignorate.
