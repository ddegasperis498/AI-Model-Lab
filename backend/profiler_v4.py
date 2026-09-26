from __future__ import annotations
import os
from typing import Dict

import torch


def _human(n: float) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    v = float(n)
    for u in units:
        if abs(v) < 1024 or u == units[-1]:
            return f'{v:.2f} {u}'
        v /= 1024
    return f'{v:.2f} PB'


def system_snapshot() -> Dict:
    ram_total = ram_available = None
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_total = int(vm.total)
        ram_available = int(vm.available)
    except Exception:
        pass

    cuda = torch.cuda.is_available()
    gpu = None
    if cuda:
        prop = torch.cuda.get_device_properties(0)
        gpu = {
            'name': prop.name,
            'total_vram': int(prop.total_memory),
            'total_vram_human': _human(prop.total_memory),
            'allocated': int(torch.cuda.memory_allocated(0)),
            'reserved': int(torch.cuda.memory_reserved(0)),
            'capability': list(torch.cuda.get_device_capability(0)),
        }
    return {
        'cpu_count': os.cpu_count(),
        'torch_threads': torch.get_num_threads(),
        'ram_total': ram_total,
        'ram_total_human': _human(ram_total) if ram_total else None,
        'ram_available': ram_available,
        'ram_available_human': _human(ram_available) if ram_available else None,
        'cuda_available': cuda,
        'gpu': gpu,
    }


def analytical_decoder_estimate(vocab_size: int, d_model: int, n_heads: int,
                                n_kv_heads: int, n_layers: int, context_length: int,
                                mlp_ratio: float, batch_size: int,
                                precision_bytes: int, optimizer: str) -> Dict:
    head_dim = d_model // max(1, n_heads)
    hidden = int(d_model * mlp_ratio)

    embedding = vocab_size * d_model
    # q + k + v + o, GQA-aware
    attn_per_layer = d_model * d_model + 2 * d_model * (n_kv_heads * head_dim) + d_model * d_model
    # SwiGLU gate + up + down
    mlp_per_layer = d_model * hidden * 3
    norms_per_layer = 2 * d_model
    final_norm = d_model
    params = embedding + n_layers * (attn_per_layer + mlp_per_layer + norms_per_layer) + final_norm

    param_mem = params * precision_bytes
    grad_mem = params * precision_bytes
    if optimizer == 'adamw':
        optimizer_mem = params * 8  # m + v in fp32, rough
    elif optimizer == 'sgd':
        optimizer_mem = params * 4
    else:
        optimizer_mem = 0

    # deliberately labeled rough: hidden activations plus attention scores.
    activations = batch_size * context_length * d_model * n_layers * precision_bytes * 10
    attention_scores = batch_size * n_heads * context_length * context_length * precision_bytes
    train_total = param_mem + grad_mem + optimizer_mem + activations + attention_scores
    inference_total = param_mem + batch_size * context_length * d_model * n_layers * precision_bytes * 3

    # Transformer training rough rule of thumb: ~6 FLOPs per parameter per token.
    tokens = batch_size * context_length
    train_flops_step = 6.0 * params * tokens
    inference_flops_token = 2.0 * params

    return {
        'parameter_count_estimate': int(params),
        'breakdown': {
            'embedding': int(embedding),
            'attention_per_layer': int(attn_per_layer),
            'mlp_per_layer': int(mlp_per_layer),
            'norms_per_layer': int(norms_per_layer),
        },
        'memory': {
            'parameters_bytes': int(param_mem), 'parameters_human': _human(param_mem),
            'gradients_bytes': int(grad_mem), 'gradients_human': _human(grad_mem),
            'optimizer_bytes': int(optimizer_mem), 'optimizer_human': _human(optimizer_mem),
            'activations_rough_bytes': int(activations), 'activations_rough_human': _human(activations),
            'attention_scores_rough_bytes': int(attention_scores), 'attention_scores_rough_human': _human(attention_scores),
            'training_total_rough_bytes': int(train_total), 'training_total_rough_human': _human(train_total),
            'inference_total_rough_bytes': int(inference_total), 'inference_total_rough_human': _human(inference_total),
        },
        'compute': {
            'train_flops_per_step_rough': train_flops_step,
            'inference_flops_per_token_rough': inference_flops_token,
        },
        'notes': [
            'Le stime di memoria attivazioni/FLOPs sono approssimazioni didattiche, non misure del profiler runtime.',
            'Flash/SDPA può evitare di materializzare interamente la matrice di attention e ridurre la memoria reale.',
            'Allocator, kernel workspace, master weights, KV cache e frammentazione possono cambiare il consumo effettivo.',
        ],
    }
