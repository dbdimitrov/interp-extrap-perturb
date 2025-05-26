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

# —– theme —
html_theme = "pydata_sphinx_theme"

# —– basic colours —–
html_theme_options = {
    # your existing options…
    "primary_color":    "#8B0000",
    "secondary_color":  "#777777",
    "navbar_start":     ["navbar-logo"],
    "navbar_end":       ["theme-switcher", "navbar-icon-links"],
    "show_toc_level":   2,
    "secondary_sidebar_items": [],  
}

html_static_path = ['_static']
html_css_files = [
    # DataTables core stylesheet
    'https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css',
    "css/custom.css"
]

html_js_files = [
    # jQuery (DataTables dependency)
    'https://code.jquery.com/jquery-3.6.0.min.js',
    # DataTables library
    'https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js',
]
html_title = project