# 03 - API Reference V4

Base: `/api/v4`

| Metodo | Endpoint | Funzione |
|---|---|---|
| GET | `/health` | Versione, feature, CPU/RAM/GPU e stato job |
| POST | `/lm/create` | Crea tokenizer, train/val token split e ModernTransformerLM |
| GET | `/lm/state` | Stato, metriche, history, tensor snapshot e attention |
| POST | `/lm/train` | Training sincrono per N step |
| POST | `/lm/job/start` | Training background |
| GET | `/lm/job/status` | Progresso job |
| POST | `/lm/job/stop` | Arresto cooperativo |
| POST | `/lm/generate` | Sampling autoregressivo |
| POST | `/tokenizer/preview` | Token IDs/byte preview |
| POST | `/embedding/similarity` | Similarità coseno embedding |
| POST | `/checkpoint/save` | Salva `.pt` V4 |
| GET | `/checkpoint/list` | Elenco checkpoint V4 |
| POST | `/checkpoint/load` | Ripristina modello/optimizer/tokenizer |
| POST | `/experiment/save` | Salva run in SQLite |
| GET | `/experiment/list` | Elenca esperimenti |
| DELETE | `/experiment/{id}` | Elimina record |
| POST | `/scale/estimate` | Stima parametri/memoria/FLOPs |

Per lo schema JSON completo e la validazione Pydantic usa Swagger su `/docs`.
