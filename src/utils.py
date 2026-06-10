from **future** import annotations

import random
from pathlib import Path

import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

```
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

def ensure_dir(path: str | Path) -> Path:
path = Path(path)
path.mkdir(parents=True, exist_ok=True)
return path

def get_device() -> str:
return "cuda" if torch.cuda.is_available() else "cpu"
