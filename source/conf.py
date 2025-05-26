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

html_theme = 'alabaster'
html_static_path = ['_static']
html_css_files = [
    # DataTables core stylesheet
    'https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css',
]

html_js_files = [
    # jQuery (DataTables dependency)
    'https://code.jquery.com/jquery-3.6.0.min.js',
    # DataTables library
    'https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js',
]