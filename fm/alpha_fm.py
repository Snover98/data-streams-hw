import numpy as np

from typing import Callable

class AlphaFm:
    def __init__(self, hash_func: Callable[[int], float]):
        self.hash_func: Callable[[int], float] = np.vectorize(hash_func, otypes=[float])
        self.min_hash: float = 1.0
    
    def update(self, values: np.ndarray) -> 'AlphaFm':
        self.min_hash = min(self.min_hash, np.min(self.hash_func(values)))
        return self
    
    def estimate(self) -> float:
        return 1.0 / self.min_hash