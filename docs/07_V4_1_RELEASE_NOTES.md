# AI Model Lab V4.1 — Release Notes

## UX operativa
- Loading overlay globale con spinner, descrizione, device/meta ed elapsed time.
- Button busy state e protezione dal doppio click.
- Error dialog centralizzato con retry/copy/details.
- Realtime badge non bloccante.

## Backend
- `SafeJSONResponse` su tutto FastAPI.
- `sanitize_for_json` ricorsivo.
- Correlation ID middleware.
- Contratto errori uniforme per HTTPException e 500.
- MLP AMP retry FP32 e controlli `isfinite`.
- Gradient/weight telemetry robusta.

## Regressione corretta
Il primo training step del MLP `1→32→32→1` poteva terminare con una response contenente `NaN`, causando:
`ValueError: Out of range float values are not JSON compliant: nan`.
V4.1 impedisce sia la sorgente statistica più comune sia la propagazione di numeri non JSON-compliant.

## Avvio
Ultimate V4.1 è l'unico backend necessario per Framework Pro e V4 Ultimate. Lo starter Advanced reindirizza allo stesso avvio.


## V4.1.2 — AMP multi-step hotfix

- Corretto lo stato `torch.amp.GradScaler` quando un batch AMP produce gradienti non finiti e viene ritentato in FP32.
- Risolto l'errore: `unscale_() has already been called on this optimizer since the last update()`.
- Aggiunto regression test CUDA/AMP con la sequenza reale `1 step` seguita da `50 step`.
- Aggiornata la versione backend/frontend a V4.1.2.
