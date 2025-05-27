project   = 'Interpretation, Extrapolation, and Perturbation of Single cells'
author    = 'Daniel Dimitrov*†, Stefan Schrod*†, Martin Rohbeck & Oliver Stegle†'
release   = '0.0.1'
html_title = project
root_doc = 'index'

# ── Theme setup ──
html_theme = "sphinx_book_theme"
html_theme_options = {
    # keep your sidebar TOC
    "show_navbar_depth": 2,
    "show_toc_level":    2,
    "home_page_in_toc":  False,
    # brand colours (dark grey & red accents)
    "launch_buttons": {},    # disable binder, etc.
}

# static assets & your DataTables CSS/JS remain unchanged
html_static_path = ['_static']
html_css_files = [
    'https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css',
    "css/custom.css",
]
html_js_files = [
    'https://code.jquery.com/jquery-3.6.0.min.js',
    'https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js',
]