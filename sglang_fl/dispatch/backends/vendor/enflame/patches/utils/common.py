from functools import lru_cache

@lru_cache(maxsize=1)
def is_cuda_alike():
    return True