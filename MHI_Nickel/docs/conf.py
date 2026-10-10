"""Sphinx configuration for the hydrogen-permeation method reference.

Build with:
    cd docs && make html

Requires an environment with sphinx, myst-parser and furo, plus the packages
``models`` imports at module scope (numpy, ase, networkx, pandas, matplotlib,
scipy). On this machine that is ``mace_env``:

    ~/anaconda3/envs/mace_env/bin/python -m sphinx -M html . _build

The package is not installed; ``sys.path`` below makes ``models`` importable
straight from the working tree.
"""
import os
import re
import sys

# --- import-time side effects, neutralised BEFORE autodoc imports anything ---
# conf.py runs first, which is what makes this work. setdefault, not assignment,
# so an explicit env var on the command line still wins.
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_HERE, ".mplcache"))

# models/ lives one level up and is not pip-installed.
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

project = "Hydrogen permeation through metal membranes"
author = "Azeez Akinyemi"
copyright = "2026, Azeez Akinyemi"

# Single source of truth is ../pyproject.toml. Parsed by regex because this
# environment is Python 3.9 and has no tomllib.
_pyproject = os.path.join(_HERE, "..", "pyproject.toml")
with open(_pyproject, encoding="utf-8") as _fh:
    release = re.search(r'^version\s*=\s*"([^"]+)"', _fh.read(), re.M).group(1)
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",      # docstrings -> reference
    "sphinx.ext.napoleon",     # NumPy / Google docstring styles
    "sphinx.ext.viewcode",     # [source] link per function
    "sphinx.ext.intersphinx",  # numpy.ndarray etc. become links
    "sphinx.ext.mathjax",      # $$ math in the .md
    "myst_parser",             # REQUIRED: lets Sphinx read .md at all
    "sphinxcontrib.mermaid",   # renders the ```mermaid fences as diagrams
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "_tools",
    "figures",          # toy example + figure/check scripts, not documents
    "README.md",        # included verbatim into index.md; not a page of its own
    "CHECKPOINT.md",    # working state for writing the docs, not part of the reference
]

# --- MyST --------------------------------------------------------------------
myst_enable_extensions = [
    "dollarmath",    # $...$ and $$...$$
    "colon_fence",   # :::{note} — avoids nested-backtick pain inside stubs
]
# Route ```mermaid fences to the mermaid directive instead of Pygments,
# which has no such lexer.
myst_fence_as_directive = ["mermaid"]

myst_heading_anchors = 3        # stable #slug per heading -> deep links survive
myst_dmath_double_inline = True
myst_footnote_transition = False

# --- napoleon ----------------------------------------------------------------
napoleon_google_docstring = False   # exactly ONE style; see trap 3
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# Any docstring section header that is NOT a napoleon built-in must be listed
# here, or docutils reads it as a section title inside a directive and errors.
# This list is load-bearing, not decorative. Extend it when the build complains.
napoleon_custom_sections = [
    # Discovered by scanning every docstring in models/ for underlined headers.
    # Regenerate with docs/_tools/scan_docstring_sections.py when docstrings change.
    ("Usage", "notes"),
    ("Typical usage", "notes"),
    ("Pipeline", "notes"),
    ("Sections", "notes"),
    ("Section A: Slab Construction & Surface Relaxation", "notes"),
    ("Why this module exists", "notes"),
    ("Mapping mode (Finding A)", "notes"),
    ("Public API", "notes"),
    ("Sieverts' law", "notes"),
    ("Richardson-Sieverts permeability", "notes"),
    ("Functions", "notes"),
    ("Label convention", "notes"),
    ("Forward and reverse are built from their own end states", "notes"),
    ("Low-frequency handling (quasi-harmonic)", "notes"),
    ("Layout", "notes"),
    ("Workflow", "notes"),
]

# --- autodoc -----------------------------------------------------------------
autodoc_member_order = "bysource"   # alphabetical destroys a deliberate order
autodoc_typehints = "none"
autodoc_preserve_defaults = True    # keeps seed=42 readable
autodoc_inherit_docstrings = False
autodoc_default_options = {"members": True, "show-inheritance": False}
# Do NOT enable undoc-members globally — see trap 4.

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
intersphinx_timeout = 5             # fail fast when building offline

nitpicky = False

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"{project} {release}"
