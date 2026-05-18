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

