import random
import numpy as np
import torch

DEFAULT_SEED = 4242

def reset_seed(seed: int = DEFAULT_SEED, cuda: bool|None = None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cuda is None:
        # agnostic
        cuda = torch.cuda.is_available()

    if cuda:
        torch.cuda.manual_seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)