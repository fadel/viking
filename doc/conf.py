# Configuration file for Sphinx
project = "VIKING"
copyright = "2025, VIKING authors"
author = "VIKING authors"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]
templates_path = ["_templates"]
exclude_patterns = []
pygments_style = "bw"

intersphinx_mapping = {
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "matfree": ("https://pnkraemer.github.io/matfree/", None),
    "python": ("https://docs.python.org/3/", None),
}

# Options for HTML output
html_theme = "basic"
html_math_renderer = None
html_static_path = ["_static"]
html_css_files = ["custom.css", "katex.min.css"]
html_js_files = [
    ("katex.min.js", {"defer": "defer"}),
    (
        "auto-render.min.js",
        {"defer": "defer", "onload": "renderMathInElement(document.body);"},
    ),
]
html_sidebars = {
    "**": [
        "searchbox.html",
        "globaltoc.html",
        "sourcelink.html",
    ],
}
html_show_sourcelink = False
