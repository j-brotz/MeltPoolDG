# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


from docutils import nodes
from sphinx.util import docutils
import os
import sys

# Add the root project directory to sys.path to resolve external files
sys.path.insert(0, os.path.abspath('..'))  # Adjust this if needed

# Specify the directories myst-parser should look in for include files
myst_include_dirs = [os.path.abspath('../')]  # Adjust this as needed

# conf.py

# Allow raw HTML in the output (this is required for embedded videos)

# Allow HTML raw directives in reStructuredText files
raw_enabled = True
# Allow raw HTML in the documentation
raw_allowed = ['html']

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = 'MeltPoolDG'
copyright = '2025, MeltPoolDG'
author = 'MeltPoolDG'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['nbsphinx', 'myst_parser']
templates_path = ['_templates']
exclude_patterns = []
source_suffix = {
    '.rst': 'rst',
    '.md': 'markdown',
    '.ipynb': 'ipynb',
}

# Enable MathJax for LaTeX rendering
mathjax_path = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_static_path = ['_static']
# Add custom JavaScript file
html_js_files = [
    'slideshow.js',
]
html_css_files = [
    'custom.css',  # Link to the custom CSS file in the _static folder
]
html_theme = 'sphinx_rtd_theme'
html_logo = '../../logo/logo_icon.png'
