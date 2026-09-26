# AI Model Lab V4.1 Professional

AI Model Lab è un laboratorio locale a doppio motore: modalità didattica trasparente e Framework Pro con FastAPI + PyTorch reale. La V4.1 aggiunge feedback operativo professionale e rende robuste le API contro metriche numeriche non finite.

## Novità V4.1
- overlay centrale di caricamento per operazioni PyTorch/Transformer;
- stato pulsanti `Creazione…`, `Training…`, `Generazione…`;
- modal di errore strutturato con HTTP status, endpoint, correlation ID, dettagli tecnici, copia e riprova;
- training realtime con indicatore live non bloccante;
- risposta API JSON-safe: `NaN`/`Infinity` diventano `null`;
- statistiche tensor con deviazione standard sicura (`unbiased=False`);
- fallback automatico FP32 se AMP produce loss/gradienti non finiti;
- contratto errori FastAPI strutturato;
- `START_ULTIMATE_V4.bat` evita una seconda istanza sulla porta 8765;
- `START_ADVANCED_PYTORCH.bat` è ora un alias sicuro di Ultimate V4;
- favicon locale;
- regression test specifico per MLP `1 → 32 → 32 → 1` / ReLU / AdamW.

## Modalità
- **Didattica**: MLP JavaScript leggibile, neuroni/pesi, forward/backprop, generalizzazione, diagnostica, autograd didattico.
- **Framework Pro V4.1**: PyTorch reale, CUDA/CPU, MLP, Transformer LM, Torch X-Ray, checkpoint, Modern Transformer V4, LoRA, GQA, RoPE, RMSNorm, SwiGLU, SDPA, AMP, gradient accumulation/checkpointing, BPE, experiment tracking e scale profiler.

## Avvio consigliato
1. Esegui `INSTALL_ADVANCED_REQUIREMENTS.bat` una sola volta se mancano le dipendenze.
2. Esegui **solo** `START_ULTIMATE_V4.bat`.
3. Apri `http://127.0.0.1:8765`.
4. API docs: `http://127.0.0.1:8765/docs`.
5. Test: `RUN_BACKEND_SELF_TEST.bat`, `RUN_V4_SELF_TEST.bat`, `RUN_V41_REGRESSION_TESTS.bat`.

Se la porta 8765 è già occupata da AI Model Lab, lo starter rileva l'istanza esistente e apre semplicemente il browser.

## Test V4.1 verificato
La configurazione che aveva prodotto `500 Internal Server Error / NaN JSON` è coperta da un test regressivo:

- input: 1
- hidden: 32,32
- output: 1
- ReLU
- AdamW
- batch 32
- LR 0.001
- weight decay 0.0001
- gradient clip 1.0
- seed 42

Il test crea 1.153 parametri, esegue 1 step e poi 50 step e verifica che tutte le metriche JSON siano finite oppure `null`.

## Documentazione
Consulta `docs/00_START_HERE.md`, `docs/01_MANUALE_COMPLETO.md`, `docs/06_TROUBLESHOOTING.md` e `docs/07_V4_1_RELEASE_NOTES.md`.
