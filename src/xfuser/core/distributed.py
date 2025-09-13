# src/xfuser/core/distributed.py
import os
import torch.distributed as dist

def init_distributed_environment(backend="nccl", init_method="env://"):
    if dist.is_available() and not dist.is_initialized():
        try:
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=int(os.environ.get("WORLD_SIZE", 1)),
                rank=int(os.environ.get("RANK", 0))
            )
        except Exception as e:
            print(f"[xfuser] init failed, running single-process: {e}")

def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def barrier():
    if dist.is_initialized():
        dist.barrier()
