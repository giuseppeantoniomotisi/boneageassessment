# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
baa_dir = os.path.abspath('..')
sys.path.insert(0, baa_dir)
sys.path.insert(0, os.path.join(baa_dir, 'RSNA'))
sys.path.insert(0, os.path.join(baa_dir, 'age'))
sys.path.insert(0, os.path.join(baa_dir, 'preprocessing'))

project = 'Bone Age Assessment'
copyright = '2024, GiuseppeAntonioMotisi&GiuseppeFanciulli'
author = 'GiuseppeAntonioMotisi&GiuseppeFanciulli'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
