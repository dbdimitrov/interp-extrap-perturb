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

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
STATIC_YAML  = os.path.join(BASE_DIR, "methods.yaml")
TEMPLATE_DIR = BASE_DIR
TEMPLATE_FN  = "methods.rst.j2"
OUTPUT_RST   = os.path.join(BASE_DIR, "source", "methods.rst")

with open(STATIC_YAML, 'r') as f:
    methods = yaml.safe_load(f)

for m in methods:
    for key in ("Task", "Model", "Inspired by"):
        v = m.get(key)
        if v is None:
            m[key] = []
        else:
            # ensure it's a flattened list
            as_list = v if isinstance(v, list) else [v]
            m[key] = flatten(as_list)

env = Environment(
    loader=FileSystemLoader([TEMPLATE_DIR]),
    trim_blocks=True,
    lstrip_blocks=True,
)
template = env.get_template(TEMPLATE_FN)

rst_out = template.render(
    methods=methods,
    title='All Methods'
)
with open(OUTPUT_RST, 'w') as f:
    f.write(rst_out)