# Interpretation, Extrapolation, and Perturbation of Single Cells

[![Documentation Status](https://readthedocs.org/projects/interp-extrap-perturb/badge/?version=latest)](https://interp-extrap-perturb.readthedocs.io/en/latest/)
[![GitHub stars](https://img.shields.io/github/stars/dbdimitrov/interp-extrap-perturb?style=social)](https://github.com/dbdimitrov/interp-extrap-perturb/stargazers)
[![License](https://img.shields.io/github/license/dbdimitrov/interp-extrap-perturb)](LICENSE)


A **living catalogue** of computational methods that interpret or predict single‑cell perturbations.
The project curates over 100 peer‑reviewed and pre‑print tools, classifies them by task, and provides a browsable web interface with informative tables and technical descriptions.

----

## Quick links

| Resource              | URL                                                                                            |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| **Docs / Browser**    | [https://interp-extrap-perturb.readthedocs.io/](https://interp-extrap-perturb.readthedocs.io/) |
| **Individual Methods**| [`methods/`](methods/)                                                                          |
| **Generation script** | [`generate_methods.py`](generate_methods.py)                                                   |

---

## Data flow

```
methods.yaml ─▶ generate_methods.py ─▶ docs/methods*.rst ─▶ Sphinx ▶ Read the Docs
```

1. **`methods.yaml`** — canonical metadata (method, year, tasks, code link, …).
2. **`generate_methods.py`** converts YAML → ReStructuredText via Jinja2:

   * One overview page (`methods.rst`).
   * One page per task (slugified).
     Tasks are listed in `tasklist.txt`; unknown tasks raise a warning.
3. Sphinx + *sphinx‑book‑theme* renders the site; ReadTheDocs rebuilds on each push.

---

## Contributing

For contribution guidelines, please refer to the **Contribute** section of the online documentation.

---

## License

MIT © 2025 Daniel Dimitrov, Stefan Schrod, Martin Rohbeck, Oliver Stegle
