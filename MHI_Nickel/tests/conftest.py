"""
tests/conftest.py
==================
Pre-import the real plotting/dataframe stack when the environment has a
working one.

Several test modules stub matplotlib/pandas into sys.modules with
``sys.modules.setdefault(name, MagicMock())`` so they can run in offline
environments where those packages are broken (e.g. NumPy-2.x ABI clashes).
``setdefault`` is only safe if the REAL modules are already imported in a
healthy environment — otherwise the first stubbing test module poisons
every later test that needs real matplotlib (plot tests then "pass" against
MagicMocks or fail asserting real files).

Importing them here, before any test module, makes the stubs no-ops in a
healthy environment while preserving the offline fallback in a broken one.
"""

try:
    import matplotlib
    matplotlib.use('Agg')          # headless backend before any pyplot import
    import matplotlib.pyplot       # noqa: F401
except ImportError:                # broken/absent → per-file stubs take over
    pass

try:
    import pandas                  # noqa: F401
except ImportError:
    pass
