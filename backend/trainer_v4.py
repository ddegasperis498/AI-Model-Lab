from __future__ import annotations
import math
import random
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from .modern_models import ModernTransformerLM
from .schemas_v4 import V4LMConfig
from .tokenizer_bpe import TrainableByteBPETokenizer
from .profiler_v4 import system_snapshot
from .trainer import (
    gradient_stats, make_optimizer, module_tree, param_snapshot,
    parameter_count, seed_everything, select_device, weight_stats,
)
from .models import ByteTokenizer


class ByteTokenizerAdapter:
    vocab_size = 256
    def encode(self, text: str) -> List[int]:
        return ByteTokenizer.encode(text)
    def decode(self, ids: List[int]) -> str:
        return ByteTokenizer.decode(ids)
    def state_dict(self):
        return {'kind': 'byte'}
    def preview(self, text: str, limit: int = 128):
        rows=[]
        for i in self.encode(text)[:limit]:
            b=bytes([i])
            rows.append({'id': i, 'hex': b.hex(' '), 'text': b.decode('utf-8', errors='replace'), 'bytes': 1})
        return rows


class UltimateLMTrainer:
    def __init__(self, config: V4LMConfig, corpus: str, validation_fraction: float = 0.1):
        seed_everything(config.seed)
        self.config = config
        self.device = select_device(config.device)
        self.lock = threading.RLock()
        self.step = 0
        self.tokens_seen = 0
        self.history: List[dict] = []
        self.last: Dict = {}

        if config.tokenizer == 'bpe':
            tok = TrainableByteBPETokenizer()
            tok.train(corpus, target_vocab_size=config.bpe_vocab_size)
            self.tokenizer = tok
        else:
            self.tokenizer = ByteTokenizerAdapter()

        ids = self.tokenizer.encode(corpus)
        if len(ids) < config.context_length + 8:
            repeats = math.ceil((config.context_length + 8) / max(1, len(ids)))
            ids = ids * repeats
        split = int(len(ids) * (1.0 - validation_fraction))
        split = max(config.context_length + 2, min(split, len(ids) - max(2, config.context_length // 2)))
        self.train_ids = torch.tensor(ids[:split], dtype=torch.long)
        self.val_ids = torch.tensor(ids[split:], dtype=torch.long)
        if len(self.val_ids) < config.context_length + 2:
            self.val_ids = self.train_ids[-min(len(self.train_ids), config.context_length * 4):].clone()

        self.raw_model = ModernTransformerLM(config, self.tokenizer.vocab_size).to(self.device)
        self.model = self.raw_model
        if config.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.raw_model)
            except Exception:
                self.model = self.raw_model

        trainable = [p for p in self.raw_model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError('Nessun parametro trainabile: controlla configurazione LoRA/freeze')
        self.optimizer = make_optimizer(config.optimizer, trainable, config.lr, config.weight_decay)

        self.amp_dtype = self._resolve_amp_dtype(config.amp_dtype)
        self.use_autocast = self.amp_dtype is not None
        self.scaler = torch.amp.GradScaler(
            'cuda', enabled=(self.device.type == 'cuda' and self.amp_dtype == torch.float16)
        )

    def _resolve_amp_dtype(self, choice: str):
        if choice == 'none':
            return None
        if self.device.type == 'cuda':
            if choice == 'bf16':
                return torch.bfloat16
            if choice in ('fp16', 'auto'):
                return torch.float16
        if self.device.type == 'cpu' and choice == 'bf16':
            return torch.bfloat16
        return None

    def _lr_for_step(self, step: int) -> float:
        c = self.config
        if c.warmup_steps > 0 and step <= c.warmup_steps:
            return c.lr * step / c.warmup_steps
        progress = min(1.0, max(0.0, (step - c.warmup_steps) / max(1, c.lr_decay_steps - c.warmup_steps)))
        return c.min_lr + 0.5 * (c.lr - c.min_lr) * (1.0 + math.cos(math.pi * progress))

    def _set_lr(self, lr: float):
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def _sample_batch(self, ids: torch.Tensor, batch_size: int):
        T = self.config.context_length
        max_start = len(ids) - T - 1
        if max_start < 1:
            raise ValueError('Corpus troppo corto per context_length')
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([ids[i:i+T] for i in starts]).to(self.device)
        y = torch.stack([ids[i+1:i+T+1] for i in starts]).to(self.device)
        return x, y

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype if self.amp_dtype is not None else torch.float32,
            enabled=self.use_autocast,
        )

    def evaluate(self, batches: int = 8, batch_size: int = 4) -> Dict[str, float]:
        with self.lock:
            self.raw_model.eval()
            vals=[]
            with torch.no_grad():
                for _ in range(batches):
                    x,y=self._sample_batch(self.val_ids, min(batch_size, 16))
                    with self._autocast():
                        _,loss=self.model(x,y)
                    vals.append(float(loss.detach().cpu()))
            self.raw_model.train()
            mean=sum(vals)/len(vals)
            return {'val_loss': mean, 'val_perplexity': float(math.exp(min(mean, 20.0)))}

    def train(self, steps: int, batch_size: int, eval_interval: int = 20,
              callback: Optional[Callable[[dict], None]] = None,
              stop_event: Optional[threading.Event] = None):
        curve=[]
        for _ in range(steps):
            if stop_event is not None and stop_event.is_set():
                break
            t0=time.perf_counter()
            with self.lock:
                self.raw_model.train()
                self.optimizer.zero_grad(set_to_none=True)
                total_loss=0.0
                micro_tokens=0
                for _micro in range(self.config.grad_accum_steps):
                    x,y=self._sample_batch(self.train_ids, batch_size)
                    with self._autocast():
                        _,loss=self.model(x,y)
                        scaled_loss=loss/self.config.grad_accum_steps
                    self.scaler.scale(scaled_loss).backward()
                    total_loss += float(loss.detach().cpu())
                    micro_tokens += int(x.numel())

                self.scaler.unscale_(self.optimizer)
                grad_before=gradient_stats(self.raw_model)
                clip_return=float(torch.nn.utils.clip_grad_norm_(
                    [p for p in self.raw_model.parameters() if p.requires_grad], self.config.grad_clip
                ).detach().cpu())
                next_step=self.step+1
                lr=self._lr_for_step(next_step)
                self._set_lr(lr)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.step=next_step
                self.tokens_seen += micro_tokens
                elapsed=max(time.perf_counter()-t0,1e-9)
                train_loss=total_loss/self.config.grad_accum_steps
                item={
                    'step': self.step,
                    'train_loss': train_loss,
                    'train_perplexity': float(math.exp(min(train_loss,20.0))),
                    'lr': lr,
                    'tokens_per_second': micro_tokens/elapsed,
                    'tokens_seen': self.tokens_seen,
                    'grad_l2': grad_before['l2'],
                    'grad_max_abs': grad_before['max_abs'],
                    'clip_norm_return': clip_return,
                    'elapsed_s': elapsed,
                }
                if self.step % eval_interval == 0:
                    item.update(self.evaluate(batches=4,batch_size=min(batch_size,8)))
                self.last=item
                self.history.append(item)
                if len(self.history)>1000:
                    self.history=self.history[-1000:]
                curve.append(item)
            if callback:
                callback(item)
        return self.state(extra={'new_history': curve})

    def generate(self, prompt: str, max_new_tokens: int, temperature: float,
                 top_k: int, top_p: float, repetition_penalty: float,
                 seed: Optional[int] = None):
        with self.lock:
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            ids=self.tokenizer.encode(prompt)
            if not ids:
                ids=[10]
            idx=torch.tensor([ids],dtype=torch.long,device=self.device)
            self.raw_model.eval()
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    x=idx[:,-self.config.context_length:]
                    with self._autocast():
                        logits,_=self.model(x)
                    logits=logits[:,-1,:].float()
                    if repetition_penalty != 1.0:
                        for token_id in set(idx[0].tolist()[-self.config.context_length:]):
                            val=logits[0,token_id]
                            logits[0,token_id]=val/repetition_penalty if val>0 else val*repetition_penalty
                    logits=logits/max(temperature,1e-6)
                    if top_k>0 and top_k<logits.size(-1):
                        v,_=torch.topk(logits,min(top_k,logits.size(-1)))
                        logits[logits<v[:,-1,None]]=float('-inf')
                    probs=torch.softmax(logits,dim=-1)
                    if top_p<1.0:
                        sorted_probs,sorted_idx=torch.sort(probs,descending=True)
                        cumulative=torch.cumsum(sorted_probs,dim=-1)
                        mask=cumulative>top_p
                        mask[...,1:]=mask[...,:-1].clone()
                        mask[...,0]=False
                        sorted_probs[mask]=0
                        sorted_probs/=sorted_probs.sum(dim=-1,keepdim=True)
                        choice=torch.multinomial(sorted_probs,1)
                        next_id=sorted_idx.gather(-1,choice)
                    else:
                        next_id=torch.multinomial(probs,1)
                    idx=torch.cat([idx,next_id],dim=1)
            self.raw_model.train()
            out=idx[0].tolist()
            return {'text': self.tokenizer.decode(out), 'token_ids': out, 'new_token_count': max_new_tokens}

    def tokenizer_preview(self, text: str, limit: int = 80):
        return self.tokenizer.preview(text,limit)

    def embedding_similarity(self, text: str, limit: int = 16):
        ids=self.tokenizer.encode(text)[:limit]
        if not ids:
            return {'tokens':[],'matrix':[]}
        with self.lock, torch.no_grad():
            W=self.raw_model.token_embedding.weight.detach().float().cpu()
            vec=F.normalize(W[torch.tensor(ids)],dim=-1)
            sim=vec@vec.T
        tokens=[self.tokenizer.decode([i]) for i in ids]
        return {'ids':ids,'tokens':tokens,'matrix':sim.tolist()}

    def state(self, extra: Optional[dict] = None):
        with self.lock:
            counts=parameter_count(self.raw_model)
            trainable=sum(p.numel() for p in self.raw_model.parameters() if p.requires_grad)
            payload={
                'engine':'pytorch-v4','kind':'modern_transformer_lm','step':self.step,
                'device':str(self.device),'config':self.config.model_dump(),
                'tokenizer':{'kind':self.config.tokenizer,'vocab_size':self.tokenizer.vocab_size},
                'dataset':{'train_tokens':len(self.train_ids),'val_tokens':len(self.val_ids)},
                'params':{**counts,'trainable':trainable,'frozen':counts['total']-trainable},
                'gradients':gradient_stats(self.raw_model),
                'weights':weight_stats(self.raw_model),
                'modules':module_tree(self.raw_model,limit=220),
                'parameter_snapshot':param_snapshot(self.raw_model,limit=160),
                'last':self.last,
                'history':self.history[-240:],
                'last_attention':self.raw_model.attention_snapshot(),
                'system':system_snapshot(),
                'capabilities':{
                    'rmsnorm':True,'rope':True,'swiglu':True,'gqa':True,'sdpa':bool(self.config.use_sdpa),
                    'lora':self.config.lora_rank>0,'gradient_checkpointing':self.config.gradient_checkpointing,
                    'torch_compile_requested':self.config.compile_model,'amp_dtype':self.config.amp_dtype,
                },
            }
            if extra: payload.update(extra)
            return payload

    def checkpoint_payload(self):
        return {
            'version':4,'kind':'modern_transformer_lm','config':self.config.model_dump(),
            'step':self.step,'tokens_seen':self.tokens_seen,
            'model_state':self.raw_model.state_dict(),'optimizer_state':self.optimizer.state_dict(),
            'tokenizer_state':self.tokenizer.state_dict(),'history':self.history[-1000:],
            'train_ids':self.train_ids,'val_ids':self.val_ids,
        }

    @classmethod
    def from_checkpoint(cls, payload: dict):
        # Construct with decoded training corpus surrogate, then restore exact tensors/tokenizer/model.
        cfg=V4LMConfig(**payload['config'])
        train_ids=payload['train_ids'].tolist()
        if payload['tokenizer_state']['kind']=='byte':
            tokenizer=ByteTokenizerAdapter()
        else:
            tokenizer=TrainableByteBPETokenizer.from_state_dict(payload['tokenizer_state'])
        corpus=tokenizer.decode(train_ids)
        obj=cls(cfg,corpus,validation_fraction=0.1)
        obj.tokenizer=tokenizer
        obj.train_ids=payload['train_ids'].clone()
        obj.val_ids=payload['val_ids'].clone()
        # Recreate model if tokenizer vocab size differs from constructor result.
        obj.raw_model=ModernTransformerLM(cfg,tokenizer.vocab_size).to(obj.device)
        obj.raw_model.load_state_dict(payload['model_state'])
        obj.model=obj.raw_model
        if cfg.compile_model and hasattr(torch,'compile'):
            try: obj.model=torch.compile(obj.raw_model)
            except Exception: pass
        trainable=[p for p in obj.raw_model.parameters() if p.requires_grad]
        obj.optimizer=make_optimizer(cfg.optimizer,trainable,cfg.lr,cfg.weight_decay)
        obj.optimizer.load_state_dict(payload['optimizer_state'])
        obj.step=int(payload.get('step',0));obj.tokens_seen=int(payload.get('tokens_seen',0));obj.history=payload.get('history',[])
        return obj


class BackgroundTrainingJob:
    def __init__(self):
        self.thread: Optional[threading.Thread]=None
        self.stop_event=threading.Event()
        self.running=False
        self.requested_steps=0
        self.completed_steps=0
        self.started_at=None
        self.error=None
        self.last_metric=None

    def start(self, trainer: UltimateLMTrainer, steps: int, batch_size: int, eval_interval: int):
        if self.running:
            raise RuntimeError('Un job è già in esecuzione')
        self.stop_event.clear();self.running=True;self.requested_steps=steps;self.completed_steps=0;self.started_at=time.time();self.error=None
        def callback(metric):
            self.completed_steps+=1;self.last_metric=metric
        def worker():
            try:
                trainer.train(steps,batch_size,eval_interval,callback=callback,stop_event=self.stop_event)
            except Exception as exc:
                self.error=str(exc)
            finally:
                self.running=False
        self.thread=threading.Thread(target=worker,name='ai-model-lab-v4-training',daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def state(self):
        return {
            'running':self.running,'requested_steps':self.requested_steps,'completed_steps':self.completed_steps,
            'progress':self.completed_steps/max(1,self.requested_steps),'started_at':self.started_at,
            'elapsed_s':time.time()-self.started_at if self.started_at else 0.0,
            'error':self.error,'last_metric':self.last_metric,
        }
