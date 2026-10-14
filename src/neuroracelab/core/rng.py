import hashlib
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RNGManager:
    master_seed: int
    def seed_for(self, namespace):
        d=hashlib.sha256(f"{self.master_seed}:{namespace}".encode()).digest()
        return int.from_bytes(d[:8],"big")%(2**32)
    def python(self, namespace): return random.Random(self.seed_for(namespace))
    def numpy(self, namespace): return np.random.default_rng(self.seed_for(namespace))
