"""Convert novelquant.py (with # %% cell markers) to a .ipynb notebook.

Cell format:
  # %% [markdown]
  ... markdown text ...

  # %%
  ... python code ...

The first contiguous comment block under a marker becomes one cell.
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

src = Path("research/novelquant/novelquant.py")
dst = Path("research/novelquant/novelquant.ipynb")

text = src.read_text(encoding="utf-8")
lines = text.splitlines()

# A cell marker is a line whose stripped form starts with "# %%"
# (it can be exactly "# %%" or "# %% [markdown]").
def is_marker(ln):
    s = ln.strip()
    return s == "# %%" or s.startswith("# %% ")

# Find the first cell marker; everything before is the file's top-
# level docstring and is dropped from the notebook.
first_marker = None
for i, ln in enumerate(lines):
    if is_marker(ln):
        first_marker = i
        break
if first_marker is None:
    raise RuntimeError("no # %% cell marker found in source")
lines = lines[first_marker:]

# Walk the lines, building cells. Each marker starts a new cell; the
# body runs until the next marker (either kind) or EOF.
cells = []
i = 0
while i < len(lines):
    marker = lines[i].strip()
    is_markdown = marker.startswith("# %% [")
    i += 1
    body = []
    while i < len(lines) and not is_marker(lines[i]):
        body.append(lines[i])
        i += 1
    source = "\n".join(body).strip("\n")
    if not source:
        continue
    if is_markdown:
        # The py file's markdown cells are written with "# " prefix on
        # each line (Python comment style for cell content). Strip that
        # so the .ipynb shows clean markdown, not "# Markdown".
        if all(ln.startswith("# ") or ln == "#" or not ln for ln in source.splitlines()):
            source = "\n".join(
                ln[2:] if ln.startswith("# ") else ("" if ln == "#" else ln)
                for ln in source.splitlines()
            )
        cells.append(new_markdown_cell(source))
    else:
        cells.append(new_code_cell(source))

nb = new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.10",
}
nbformat.write(nb, str(dst), version=4)
print(f"wrote {dst} ({len(cells)} cells, types: {[c.cell_type for c in cells]})")

