# src/wan/utils/multitalk_utils.py

def init_distributed_environment():
    print("[multitalk_utils] init_distributed_environment() skipped (stub).")

def get_rank():
    return 0

def get_world_size():
    return 1

def barrier():
    pass


class RotaryPositionalEmbedding1D:
    def __init__(self, dim, base=10000):
        self.dim = dim
        self.base = base

    def __call__(self, x):
        return x


def normalize_and_scale(x, scale=1.0):
    return x * scale


def split_token_counts_and_frame_ids(tokens):
    return [], []
