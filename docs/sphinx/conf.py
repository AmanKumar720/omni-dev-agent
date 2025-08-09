# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------

project = 'Omni-Dev Agent'
copyright = '2024, Omni-Dev Agent Team'
author = 'Omni-Dev Agent Team'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
    'myst_parser'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'analytics_id': 'G-XXXXXXXXXX',  # Your Google Analytics ID
    'analytics_anonymize_ip': False,
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_static_path = ['_static']
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'

# Custom CSS
html_css_files = [
    'custom.css',
]

# -- Extension configuration -------------------------------------------------

# -- Options for autodoc ----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'flask': ('https://flask.palletsprojects.com/en/2.0.x/', None),
    'opencv': ('https://docs.opencv.org/4.x/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True

# -- Options for Napoleon extension ------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- MyST Parser configuration -----------------------------------------------
myst_enable_extensions = [
    "deflist",
    "tasklist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "linkify",
    "substitution",
    "tasklist",
]

myst_html_meta = {
    "description": "Omni-Dev Agent - AI-powered development and vision analytics platform",
    "keywords": "AI, computer vision, development, automation, object detection, OCR",
}

# -- Custom configuration ----------------------------------------------------

# Add custom roles
rst_prolog = """
.. role:: python(code)
   :language: python
   :class: highlight

.. role:: bash(code)
   :language: bash
   :class: highlight

.. role:: json(code)
   :language: json
   :class: highlight
"""

# Version info
version_info = {
    'version': release,
    'release': release,
}

# GitHub info
github_user = 'yourusername'
github_repo = 'omni-dev-agent'
github_url = f'https://github.com/{github_user}/{github_repo}'

html_context = {
    'github_user': github_user,
    'github_repo': github_repo,
    'github_version': 'main',
    'doc_path': 'docs/sphinx',
}

# Additional HTML options
html_title = f'{project} Documentation'
html_short_title = 'Omni-Dev Agent'
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# LaTeX output options
latex_documents = [
    ('index', 'omni-dev-agent.tex', 'Omni-Dev Agent Documentation',
     'Omni-Dev Agent Team', 'manual'),
]

# EPUB output options  
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# Manual page output options
man_pages = [
    ('index', 'omni-dev-agent', 'Omni-Dev Agent Documentation',
     [author], 1)
]

# Texinfo output options
texinfo_documents = [
    ('index', 'omni-dev-agent', 'Omni-Dev Agent Documentation',
     author, 'omni-dev-agent', 'AI-powered development and vision analytics platform.',
     'Miscellaneous'),
]
