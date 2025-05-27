# Interpretation, Extrapolation, and Perturbation of Single Cells

[![Documentation Status](https://readthedocs.org/projects/interp-extrap-perturb/badge/?version=latest)](https://interp-extrap-perturb.readthedocs.io/en/latest/)

A **living catalogue** of computational methods that interpret or predict single‑cell perturbations.
The project curates peer‑reviewed and pre‑print tools, classifies them by task, and provides a browsable web interface with rich tables and inline descriptions.

---

## Quick links

| Resource              | URL                                                                                            |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| **Docs / Browser**    | [https://interp-extrap-perturb.readthedocs.io/](https://interp-extrap-perturb.readthedocs.io/) |
| **YAML catalogue**    | [`methods.yaml`](methods.yaml)                                                                 |
| **Generation script** | [`generate_methods_rst.py`](generate_methods_rst.py)                                           |

---

## Local build in 3 steps

```bash
# 1. Clone the repo
$ git clone https://github.com/dbdimitrov/interp-extrap-perturb.git
$ cd interp-extrap-perturb

# 2. Create the docs environment (conda)
$ conda env create -f environment.yml
$ conda activate perspective

# 3. Generate method pages and build HTML
$ python generate_methods_rst.py
$ sphinx-build -b html docs docs/_build/html
```

Point a browser to `docs/_build/html/index.html`.

---

## Data flow

```
methods.yaml ─▶ generate_methods_rst.py ─▶ docs/methods*.rst ─▶ Sphinx ▶ HTML ▶ Read the Docs
```

1. **`methods.yaml`** — canonical metadata (method, year, tasks, code link, …).
2. **`generate_methods_rst.py`** converts YAML → ReStructuredText via Jinja2:

   * One overview page (`methods.rst`).
   * One page per task (slugified).
     Tasks are listed in `tasklist.txt`; unknown tasks raise a warning.
3. Sphinx + *sphinx‑book‑theme* renders the site; RTD rebuilds on each push.

---

## Contributing

For contribution guidelines, please refer to the **Contribute** section of the online documentation.

---

## License

MIT © 2025 Daniel Dimitrov, Stefan Schrod, Martin Rohbeck, Oliver Stegle
