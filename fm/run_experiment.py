import sys
from pathlib import Path
import pickle

import numpy as np
from scipy.stats import lognorm

from experiments import run_experiments

def main() -> None:
    num_hashes = int(sys.argv[1])
    num_betas = int(sys.argv[2])
    num_exps = int(sys.argv[3])
    num_procs = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    X = lognorm.rvs(5, size=int(2e5), random_state=206783441).astype(int)

    results = run_experiments(X,
                              base_seeds=np.arange(1, num_exps + 1).tolist(),
                              num_hashes=num_hashes,
                              num_betas=num_betas,
                              num_procs=num_procs)

    Path(f'results_b_{num_betas}_h_{num_hashes}.pkl').write_bytes(pickle.dumps(results))



if __name__ == '__main__':
    main()