from **future** import annotations

import csv
from pathlib import Path

def save_latency_row(
csv_path: str | Path,
row: dict,
) -> None:
csv_path = Path(csv_path)
csv_path.parent.mkdir(parents=True, exist_ok=True)

```
file_exists = csv_path.exists()

with csv_path.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))

    if not file_exists:
        writer.writeheader()

    writer.writerow(row)
```

