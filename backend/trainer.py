from __future__ import annotations
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .models import TorchMLP, TinyTransformerLM, ByteTokenizer
from .schemas import MLPConfig, LLMConfig
from .safe_json import sanitize_for_json


def select_device(choice: str) -> torch.device:
    if choice == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA richiesto ma non disponibile')
        return torch.device('cuda')
    if choice == 'cpu':
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(name: str, parameters, lr: float, weight_decay: float):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def parameter_count(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def gradient_stats(model: torch.nn.Module) -> Dict[str, float]:
    sq = 0.0
    max_abs = 0.0
    count = 0
    non_finite = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        finite = torch.isfinite(g)
        non_finite += int((~finite).sum().cpu())
        safe = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        sq += float((safe * safe).sum().cpu())
        if safe.numel():
            max_abs = max(max_abs, float(safe.abs().max().cpu()))
        count += safe.numel()
    return {'l2': sq ** 0.5, 'max_abs': max_abs, 'elements': count, 'non_finite': non_finite}


def gradients_are_finite(model: torch.nn.Module) -> bool:
    return all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())


def parameters_are_finite(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())


def weight_stats(model: torch.nn.Module) -> Dict[str, float]:
    sq = 0.0
    max_abs = 0.0
    count = 0
    non_finite = 0
    with torch.no_grad():
        for p in model.parameters():
            t = p.detach().float()
            finite = torch.isfinite(t)
            non_finite += int((~finite).sum().cpu())
            safe = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            sq += float((safe * safe).sum().cpu())
            if safe.numel():
                max_abs = max(max_abs, float(safe.abs().max().cpu()))
            count += safe.numel()
    return {'l2': sq ** 0.5, 'max_abs': max_abs, 'elements': count, 'non_finite': non_finite}


def module_tree(model: torch.nn.Module, limit: int = 160):
    rows = []
    for name, module in model.named_modules():
        if not name:
            continue
        direct_params = sum(p.numel() for p in module.parameters(recurse=False))
        rows.append({'name': name, 'type': module.__class__.__name__, 'direct_params': direct_params})
        if len(rows) >= limit:
            break
    return rows


def param_snapshot(model: torch.nn.Module, limit: int = 120):
    out = []
    for name, p in model.named_parameters():
        if len(out) >= limit:
            break
        t = p.detach().float()
        g = p.grad.detach().float() if p.grad is not None else None
        out.append({
            'name': name,
            'shape': list(p.shape),
            'numel': p.numel(),
            'mean': float(t.mean().cpu()),
            'std': float(t.std(unbiased=False).cpu()) if t.numel() > 1 else 0.0,
            'min': float(t.min().cpu()),
            'max': float(t.max().cpu()),
            'grad_mean': float(g.mean().cpu()) if g is not None else 0.0,
            'grad_max_abs': float(g.abs().max().cpu()) if g is not None else 0.0,
        })
    return out


class MLPTrainer:
    def __init__(self, config: MLPConfig):
        seed_everything(config.seed)
        self.config = config
        self.device = select_device(config.device)
        self.model = TorchMLP(config.input_size, config.hidden_sizes, config.output_size,
                              config.activation, config.dropout, config.task).to(self.device)
        self.optimizer = make_optimizer(config.optimizer, self.model.parameters(), config.lr, config.weight_decay)
        self.step = 0
        self.scaler = torch.amp.GradScaler('cuda', enabled=(config.amp and self.device.type == 'cuda'))
        self.last = {}

    def _loss(self, pred: torch.Tensor, target: torch.Tensor):
        if self.config.task == 'classification':
            return F.binary_cross_entropy_with_logits(pred, target)
        return F.mse_loss(pred, target)

    def train(self, data: List[dict], batch_size: int, steps: int):
        if not data:
            raise ValueError('Dataset vuoto')
        x_all = torch.tensor([s['x'] for s in data], dtype=torch.float32)
        y_all = torch.tensor([s['y'] for s in data], dtype=torch.float32)
        if x_all.shape[1] != self.config.input_size:
            raise ValueError(f'Input size dataset={x_all.shape[1]}, modello={self.config.input_size}')
        if y_all.shape[1] != self.config.output_size:
            raise ValueError(f'Output size dataset={y_all.shape[1]}, modello={self.config.output_size}')

        losses = []
        t0 = time.perf_counter()
        last_pred = None
        for _ in range(steps):
            ids = torch.randint(0, len(data), (min(batch_size, len(data)),))
            x = x_all[ids].to(self.device)
            y = y_all[ids].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            amp_enabled = bool(self.config.amp and self.device.type == 'cuda')
            used_amp = amp_enabled
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=amp_enabled):
                pred = self.model(x)
                loss = self._loss(pred, y)

            # A non-finite AMP step is retried in FP32 instead of poisoning the optimizer.
            if not bool(torch.isfinite(loss)):
                used_amp = False
                self.optimizer.zero_grad(set_to_none=True)
                pred = self.model(x)
                loss = self._loss(pred, y)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Loss non finita anche in FP32; controlla dati e iperparametri')
                loss.backward()
            elif amp_enabled:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if not gradients_are_finite(self.model):
                    # unscale_() moves GradScaler's per-optimizer state to UNSCALED.
                    # Finalize/reset that state before the FP32 retry, otherwise the
                    # next training iteration raises:
                    # "unscale_() has already been called on this optimizer since the last update()".
                    self.scaler.update()
                    used_amp = False
                    self.optimizer.zero_grad(set_to_none=True)
                    pred = self.model(x)
                    loss = self._loss(pred, y)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError('Loss non finita nel retry FP32')
                    loss.backward()
            else:
                loss.backward()

            if not gradients_are_finite(self.model):
                raise RuntimeError('Gradienti non finiti: training step annullato')

            grad_before_clip = gradient_stats(self.model)
            clip_tensor = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            clip_value = float(torch.nan_to_num(clip_tensor.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).cpu())

            if used_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if not parameters_are_finite(self.model):
                raise RuntimeError('Parametri non finiti dopo optimizer.step(); riduci learning rate o disattiva AMP')

            self.step += 1
            losses.append(float(loss.detach().float().cpu()))
            last_pred = pred.detach().float().cpu()

        elapsed = max(time.perf_counter() - t0, 1e-9)
        with torch.no_grad():
            eval_pred = self.model(x_all.to(self.device))
            eval_loss = float(self._loss(eval_pred, y_all.to(self.device)).detach().cpu())
            if self.config.task == 'classification':
                probs = torch.sigmoid(eval_pred)
                acc = float(((probs >= 0.5) == (y_all.to(self.device) >= 0.5)).float().mean().cpu())
            else:
                acc = None

        self.last = {
            'loss': losses[-1], 'eval_loss': eval_loss, 'accuracy': acc,
            'grad_before_clip': grad_before_clip, 'clip_norm_return': clip_value,
            'elapsed_s': elapsed,
        }
        return self.state(extra={'loss_curve': losses})

    def predict(self, rows: List[List[float]]):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(rows, dtype=torch.float32, device=self.device)
            y = self.model(x)
            if self.config.task == 'classification':
                y = torch.sigmoid(y)
        self.model.train()
        return y.detach().float().cpu().tolist()

    def state(self, extra=None):
        payload = {
            'engine': 'pytorch', 'kind': 'mlp', 'step': self.step,
            'device': str(self.device), 'config': self.config.model_dump(),
            'params': parameter_count(self.model),
            'gradients': gradient_stats(self.model),
            'weights': weight_stats(self.model),
            'modules': module_tree(self.model),
            'parameter_snapshot': param_snapshot(self.model),
            'activations': self.model.activation_stats,
            'last': self.last,
        }
        if extra:
            payload.update(extra)
        return sanitize_for_json(payload)


class LLMTrainer:
    def __init__(self, config: LLMConfig):
        seed_everything(config.seed)
        if config.d_model % config.n_heads != 0:
            raise ValueError('d_model deve essere divisibile per n_heads')
        self.config = config
        self.device = select_device(config.device)
        self.tokenizer = ByteTokenizer()
        self.model = TinyTransformerLM(
            d_model=config.d_model, n_heads=config.n_heads, n_layers=config.n_layers,
            context_length=config.context_length, mlp_ratio=config.mlp_ratio,
            dropout=config.dropout, tie_embeddings=config.tie_embeddings,
        ).to(self.device)
        self.optimizer = make_optimizer(config.optimizer, self.model.parameters(), config.lr, config.weight_decay)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(config.amp and self.device.type == 'cuda'))
        self.step = 0
        self.last = {}
        self.train_history = []
        self.val_history = []
        self.validation_fraction = 0.20

    def _batch(self, tokens: torch.Tensor, batch_size: int):
        T = self.config.context_length
        if len(tokens) < T + 2:
            raise ValueError(f'Testo troppo corto: servono almeno {T+2} byte/token')
        max_start = len(tokens) - T - 1
        starts = torch.randint(0, max_start + 1, (batch_size,))
        x = torch.stack([tokens[s:s+T] for s in starts])
        y = torch.stack([tokens[s+1:s+T+1] for s in starts])
        return x.to(self.device), y.to(self.device)

    def _split_tokens(self, ids, validation_fraction: float):
        T = self.config.context_length
        n = len(ids)
        min_part = T + 2
        if n < min_part * 2:
            raise ValueError(f'Corpus troppo corto per train/validation: servono almeno {min_part*2} byte/token')
        val_n = max(min_part, int(round(n * validation_fraction)))
        val_n = min(val_n, n - min_part)
        split = n - val_n
        return torch.tensor(ids[:split], dtype=torch.long), torch.tensor(ids[split:], dtype=torch.long)

    @torch.no_grad()
    def _evaluate_tokens(self, tokens: torch.Tensor, batch_size: int):
        T = self.config.context_length
        max_start = len(tokens) - T - 1
        if max_start < 0:
            raise ValueError('Validation split troppo corto per il context length')
        count = min(max(1, batch_size), max_start + 1, 32)
        if count == 1:
            starts = torch.tensor([0], dtype=torch.long)
        else:
            starts = torch.linspace(0, max_start, steps=count).round().long()
        x = torch.stack([tokens[int(s):int(s)+T] for s in starts]).to(self.device)
        y = torch.stack([tokens[int(s)+1:int(s)+T+1] for s in starts]).to(self.device)
        was_training = self.model.training
        self.model.eval()
        with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                            enabled=(self.config.amp and self.device.type == 'cuda')):
            _, loss = self.model(x, y)
        if was_training:
            self.model.train()
        return float(loss.detach().float().cpu())

    def _diagnose_generalization(self):
        if len(self.train_history) < 6 or len(self.val_history) < 6:
            return 'warming_up'
        train_now, train_old = self.train_history[-1], self.train_history[-6]
        val_now, val_old = self.val_history[-1], self.val_history[-6]
        train_improved = train_now < train_old * 0.99
        val_improved = val_now < val_old * 0.99
        if train_improved and val_now > val_old * 1.02:
            return 'overfitting'
        if val_improved:
            return 'learning'
        if self.step >= 20 and not train_improved and not val_improved:
            return 'underfitting'
        return 'stable'

    def train(self, text: str, batch_size: int, steps: int, validation_fraction: float = 0.20):
        ids = self.tokenizer.encode(text)
        train_tokens, val_tokens = self._split_tokens(ids, validation_fraction)
        self.validation_fraction = float(validation_fraction)
        train_curve, val_curve = [], []
        tokens_processed = 0
        t0 = time.perf_counter()
        for _ in range(steps):
            x, y = self._batch(train_tokens, batch_size)
            self.optimizer.zero_grad(set_to_none=True)
            amp_enabled = bool(self.config.amp and self.device.type == 'cuda')
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=amp_enabled):
                _, loss = self.model(x, y)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_before_clip = gradient_stats(self.model)
            clip_value = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip).detach().float().cpu())
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.step += 1
            train_loss = float(loss.detach().float().cpu())
            val_loss = self._evaluate_tokens(val_tokens, batch_size)
            train_curve.append(train_loss)
            val_curve.append(val_loss)
            self.train_history.append(train_loss)
            self.val_history.append(val_loss)
            self.train_history = self.train_history[-500:]
            self.val_history = self.val_history[-500:]
            tokens_processed += x.numel()

        elapsed = max(time.perf_counter() - t0, 1e-9)
        train_loss = train_curve[-1]
        val_loss = val_curve[-1]
        gap = val_loss - train_loss
        self.last = {
            'loss': train_loss,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'perplexity': math_exp_safe(train_loss),
            'train_perplexity': math_exp_safe(train_loss),
            'val_perplexity': math_exp_safe(val_loss),
            'generalization_gap': gap,
            'generalization_status': self._diagnose_generalization(),
            'validation_fraction': self.validation_fraction,
            'train_fraction': 1.0 - self.validation_fraction,
            'tokens_per_second': tokens_processed / elapsed,
            'elapsed_s': elapsed,
            'grad_before_clip': grad_before_clip,
            'clip_norm_return': clip_value,
        }
        return self.state(extra={
            'loss_curve': train_curve,
            'train_loss_curve': train_curve,
            'val_loss_curve': val_curve,
        })

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_k: int, seed: int = 42):
        self.model.eval()
        ids = self.tokenizer.encode(prompt)
        if not ids:
            ids = [32]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        generated = list(ids)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        for _ in range(max_new_tokens):
            x = idx[:, -self.config.context_length:]
            logits, _ = self.model(x)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                cutoff = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < cutoff, torch.full_like(logits, float('-inf')), logits)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
            generated.append(int(nxt.item()))
        self.model.train()
        return {'text': self.tokenizer.decode(generated), 'token_ids': generated, 'seed': int(seed)}

    def state(self, extra=None):
        attention = self.model.attention_snapshot(max_tokens=48)
        payload = {
            'engine': 'pytorch', 'kind': 'transformer_lm', 'step': self.step,
            'device': str(self.device), 'config': self.config.model_dump(),
            'params': parameter_count(self.model),
            'gradients': gradient_stats(self.model),
            'weights': weight_stats(self.model),
            'modules': module_tree(self.model),
            'parameter_snapshot': param_snapshot(self.model),
            'last_attention': attention,
            'last': self.last,
            'validation': {
                'fraction': self.validation_fraction,
                'train_fraction': 1.0 - self.validation_fraction,
                'history_points': min(len(self.train_history), len(self.val_history)),
            },
            'tokenizer': {'type': 'byte-level UTF-8', 'vocab_size': 256},
        }
        if extra:
            payload.update(extra)
        return sanitize_for_json(payload)

def math_exp_safe(x: float) -> float:
    import math
    return math.exp(min(float(x), 20.0))
