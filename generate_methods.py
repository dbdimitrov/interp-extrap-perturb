"""Generate RST files for methods overview and per‑task pages, plus a child
``toctree`` inside the all‑methods page.

Changes vs previous version
---------------------------
* Per‑task files are now named ``<slug>.rst`` (no ``methods_`` prefix).
* ``tasks`` list passed to the template is a list of ``(task, slug)`` pairs.
  The template lists each ``slug`` directly.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Sequence

import yaml
from jinja2 import Environment, FileSystemLoader

################################################################################
# Helper utilities
################################################################################

def flatten(xs: Sequence) -> List:
    """Recursively flatten nested list/tuple structure into a flat list."""
    out: List = []
    for x in xs:
        if isinstance(x, (list, tuple)):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out

def slugify(text: str) -> str:
    """Return a file‑safe, lower‑case slug using underscores."""
    return re.sub(r"[^a-z0-9_]+", "_", text.lower().replace("-", "_").replace(" ", "_")).strip("_")

################################################################################
# Paths
################################################################################
BASE_DIR      = Path(__file__).resolve().parent
STATIC_YAML   = BASE_DIR / "methods.yaml"
TASKLIST_TXT  = BASE_DIR / "tasklist.txt"
TEMPLATE_DIR  = BASE_DIR
TEMPLATE_FN   = "methods.rst.j2"
OUTPUT_DIR    = BASE_DIR / "source"
OUTPUT_DIR.mkdir(exist_ok=True)

################################################################################
# Load allowed tasks
################################################################################
if not TASKLIST_TXT.exists():
    sys.exit(f"ERROR: {TASKLIST_TXT} not found")

raw_tasks: List[str] = []
for line in TASKLIST_TXT.read_text().splitlines():
    for tok in re.split(r"[;,]", line):
        tok = tok.strip()
        if tok:
            raw_tasks.append(tok)
ALLOWED_TASKS = {t for t in raw_tasks if t}

################################################################################
# Load and normalise methods
################################################################################
methods: List[dict] = yaml.safe_load(STATIC_YAML.read_text())
for m in methods:
    for key in ("Task", "Model", "Inspired by"):
        v = m.get(key)
        m[key] = [] if v is None else flatten(v if isinstance(v, list) else [v])

unknown = sorted({t for m in methods for t in m["Task"] if t not in ALLOWED_TASKS})
if unknown:
    print("WARNING: unknown tasks in methods.yaml: " + ", ".join(unknown), file=sys.stderr)

################################################################################
# Jinja environment
################################################################################
env = Environment(
    loader=FileSystemLoader([TEMPLATE_DIR]),
    trim_blocks=True,
    lstrip_blocks=True,
)
template = env.get_template(TEMPLATE_FN)

################################################################################
# Render all‑methods page
################################################################################
sorted_tasks = sorted(ALLOWED_TASKS)
tasks_info = [(task, slugify(task)) for task in sorted_tasks]
all_rst = template.render(methods=methods, title="All Methods", tasks=tasks_info)
(OUTPUT_DIR / "methods.rst").write_text(all_rst)

################################################################################
# Render per‑task pages
################################################################################
for task, slug in tasks_info:
    subset = [m for m in methods if task in m["Task"]]
    if not subset:
        continue  # skip empty tasks
    content = template.render(methods=subset, title=task)
    (OUTPUT_DIR / f"{slug}.rst").write_text(content)

print(f"Generated methods.rst and {len(tasks_info)} task pages in '{OUTPUT_DIR}/'", file=sys.stderr)
