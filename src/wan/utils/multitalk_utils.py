# src/wan/utils/multitalk_utils.py

# ======================================
# Stub replacements for xfuser functions
# ======================================

def init_distributed_environment():
    # Do nothing (single GPU/CPU mode)
    print("[multitalk_utils] init_distributed_environment() skipped (stub).")

def get_rank():
    # Always return rank 0
    return 0

def get_world_size():
    # Always return world size = 1
    return 1

def barrier():
    # No sync needed
    pass


# Example utility functions (safe defaults)
class RotaryPositionalEmbedding1D:
    def __init__(self, dim, base=10000):
        self.dim = dim
        self.base = base

    def __call__(self, x):
        # Return x directly (no-op)
        return x


def normalize_and_scale(x, scale=1.0):
    # Dummy normalize (return as-is)
    return x * scale


def split_token_counts_and_frame_ids(tokens):
    # Dummy split (return empty lists)
    return [], []
