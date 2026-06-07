"""Launcher — extracts code cells from app.ipynb and executes them."""
import json, sys

with open("app.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

code_lines = []
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    # skip commented-out pip installs and IPython magics
    stripped = source.lstrip()
    if stripped.startswith("# %") or stripped.startswith("%%") or stripped.startswith("%"):
        continue
    code_lines.append(source)

exec(compile("\n".join(code_lines), "app.ipynb", "exec"))



