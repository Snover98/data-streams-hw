from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from pathlib import Path

from tqdm import tqdm
import numpy as np
from scipy.stats import lognorm

from experiments import seeded_hash

def calc_min_hash(vals: np.ndarray[int], seed: int) -> tuple[int, float]:
    vectorized = np.vectorize(seeded_hash(seed), otypes=[float])
    return seed, float(np.min(vectorized(vals)))


def create_min_hashes_dict(vals: np.ndarray[int], seeds: list[int], num_procs: int) -> dict[int, float]:
    with ProcessPoolExecutor(max_workers=num_procs) as executor:
        # TODO: remove unique
        values = np.unique(vals)

        futures = [executor.submit(calc_min_hash, values, seed) for seed in seeds]
        results = dict(
            f.result() for f in tqdm(as_completed(futures), total=len(futures), desc=f"min hash per seed")
        )

    return results


if __name__ == '__main__':
    Path('min_hashes_dict.pkl').write_bytes(pickle.dumps(
        create_min_hashes_dict(lognorm.rvs(5, size=int(2e5), random_state=206783441).astype(int),
                               np.arange(100 * 1000).tolist(),
                               8)
    ))