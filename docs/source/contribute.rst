How to Contribute
================================================================================

Thank you for considering contributions to this project! We welcome new methods, enhancements, bug fixes, and improvements to the existing content. To ensure consistency and smooth integration, please follow the guidelines below.

1. Add or update methods in the YAML file
-----------------------------------------

- **Location:** The `methods.yaml` file lives at the project root.
- **Format:** Each method entry should include the fields:
  - `Method`: Name of the method
  - `Year`: Publication year
  - `Task`: A list of tasks or use cases
  - `Model`: A list of models or underlying frameworks
  - `Inspired by`: A list of references or inspirations
  - `Description`: A short description used in the table’s expandable details
  - `Publication`: URL or DOI link to the paper
  - `Code Availability`: URL to the implementation repository
  - `Published`: A boolean (`true`/`false`) indicating publication status

Please follow existing entries for examples and ensure valid YAML syntax.

2. Continuous Integration
-------------------------

A GitHub Actions workflow automatically runs `generate_methods.py` against `methods.rst.j2` whenever your pull request is merged into `main`. This regenerates `source/methods.rst` to reflect your changes, so you do **not** need to run any commands manually for the table update.

3. Local preview (optional)
----------------------------

If you wish to preview the changes locally before pushing your PR:

```bash
conda env create -f environment.yml
# once created:
conda activate iep-singlecell
python generate_methods.py && make -C docs clean html
```

Open `docs/_build/html/index.html` in a browser.