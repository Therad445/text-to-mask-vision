import json
from pathlib import Path

path = Path("notebooks/01_colab_grounding_dino_sam_sanity_check.ipynb")

nb = json.loads(path.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    source = cell.get("source", [])
    fixed_source = []

    for line in source:
        # Replace literal backslash-n with real newlines.
        if "\\n" in line:
            parts = line.replace("\\n", "\n").splitlines(keepends=True)
            fixed_source.extend(parts)
        else:
            fixed_source.append(line)

    cell["source"] = fixed_source

path.write_text(
    json.dumps(nb, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Fixed notebook: {path}")
