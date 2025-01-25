import numpy as np

from alpha_fm import AlphaFm

from typing import Callable, Sequence

class BetaFm:
    def __init__(self, hash_funcs: Sequence[Callable[[int], float]]):
        self.estimators: tuple[AlphaFm, ...] = tuple(AlphaFm(func) for func in hash_funcs)
    
    def update(self, values: np.ndarray) -> 'BetaFm':
        for est in self.estimators:
            est.update(values)
        return self
    
    def estimate(self) -> float:
        return float(1 / np.mean([est.min_hash for est in self.estimators]) - 1)