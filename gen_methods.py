import os
import yaml
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

def flatten(xs):
    out = []
    for x in xs:
        if isinstance(x, list):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out

# 1) Paths
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
STATIC_YAML  = os.path.join(BASE_DIR, "source", "_static", "methods.yaml")
TEMPLATE_DIR = BASE_DIR
TEMPLATE_FN  = "methods.rst.j2"
OUTPUT_RST   = os.path.join(BASE_DIR, "source", "methods.rst")

# 2) Load YAML
with open(STATIC_YAML, 'r') as f:
    methods = yaml.safe_load(f)

# 3) Normalize all list‐style fields
for m in methods:
    for key in ("Task", "Model", "Inspired by"):
        v = m.get(key)
        if v is None:
            m[key] = []
        else:
            # ensure it's a flattened list
            as_list = v if isinstance(v, list) else [v]
            m[key] = flatten(as_list)

# 4) Collect tasks
all_tasks = sorted(
    { str(t) for m in methods for t in m["Task"] },
    key=lambda s: s.lower()
)

# 5) Jinja setup
env = Environment(
    loader=FileSystemLoader([TEMPLATE_DIR]),
    trim_blocks=True,
    lstrip_blocks=True,
)
template = env.get_template(TEMPLATE_FN)

# 6) Render with both methods & all_tasks
rst_out = template.render(
    methods=methods,
    all_tasks=all_tasks,
    title='All Methods'
)

# 7) Write out
with open(OUTPUT_RST, 'w') as f:
    f.write(rst_out)