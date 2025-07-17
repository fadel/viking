# Configuration file for Sphinx
project = "KIVI"
copyright = "2025, KIVI authors"
author = "KIVI authors"
extensions = ["sphinx.ext.autodoc"]
templates_path = ["_templates"]
exclude_patterns = []

# Options for HTML output
html_theme = "basic"
html_static_path = ["_static"]
html_css_files = ["katex.min.css"]
html_js_files = [
    ("katex.min.js", {"defer": "defer"}),
    ("auto-render.min.js",
     {
        "defer": "defer",
        "onload": "renderMathInElement(document.body);"
     }
    ),
]
