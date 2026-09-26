# 05 - Limiti, scala e interpretazione corretta

V4 usa PyTorch reale, un decoder moderno e tecniche di training reali. Non finge che un modello da miliardi di parametri possa essere addestrato sul PC se l’hardware non lo consente.

## Reale
Tensor, autograd, optimizer, Cross Entropy, CUDA se disponibile, AMP, LoRA, GQA, RoPE, RMSNorm, SwiGLU, SDPA, checkpoint, tokenizer BPE, training e generation.

## Fuori dal core
DistributedDataParallel/FSDP multi-GPU, ZeRO su cluster, dataset web-scale, storage distribuito, serving multi-replica, quantizzazione kernel-specific, RLHF/DPO su larga scala, safety/compliance e fault tolerance datacenter.

## Scale Lab
Le stime sono analitiche e non sostituiscono `torch.profiler`, Nsight o misure VRAM runtime.
