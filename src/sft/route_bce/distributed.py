"""DDP plumbing; the frozen global sampler and per-example loss stay unchanged."""
from contextlib import nullcontext
from datetime import timedelta
import os
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def primary():
    return rank() == 0


def initialize(expected):
    if int(os.environ.get("WORLD_SIZE", "1")) != expected:
        raise RuntimeError(f"Use the launcher: this recipe requires {expected} ranks")
    local = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != expected:
        raise RuntimeError(f"Assign exactly {expected} CUDA devices to this experiment")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    return local


def barrier():
    if dist.is_initialized():
        dist.barrier()


def gather(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * world_size()
    dist.all_gather_object(values, value)
    return values


def ordered_records(local_records):
    """Pairs (original_position, record); restore serial order after sharding."""
    records = sorted((p for part in gather(local_records) for p in part), key=lambda x: x[0])
    if len({i for i, _ in records}) != len(records):
        raise RuntimeError("Duplicate distributed record positions")
    return [value for _, value in records]


def local_indices(global_indices, replica=None, replicas=None):
    replica = rank() if replica is None else replica
    replicas = world_size() if replicas is None else replicas
    if not 0 <= replica < replicas or len(global_indices) % replicas:
        raise ValueError("Global batch must divide evenly over all ranks")
    return list(global_indices[replica::replicas])


def backward_scale(global_batch_size):
    # DDP averages rank gradients: (1/W) * sum_r sum_i (W/B)*grad(loss_i).
    if global_batch_size % world_size():
        raise ValueError("Global batch must be divisible by world size")
    return world_size() / global_batch_size


def accumulation_context(model, is_last):
    # Both forward and backward must be inside no_sync on accumulation steps.
    return model.no_sync() if isinstance(model, DistributedDataParallel) and not is_last else nullcontext()


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def rng_state():
    return {"torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"])
