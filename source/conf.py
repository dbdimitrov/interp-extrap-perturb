# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Interpretation, Extrapolation, and Perturbation of Single cells'
copyright = '2025, Daniel Dimitrov*†, Stefan Schrod*†, Martin Rohbeck & Oliver Stegle†'
author = 'Daniel Dimitrov*†, Stefan Schrod*†, Martin Rohbeck & Oliver Stegle†'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []
extensions += [
  "sphinx_fontawesome",
  "sphinx_togglebutton"
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_js_files = [
    'filter.js',
]
