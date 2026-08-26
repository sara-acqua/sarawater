# Configuration file for the Sphinx documentation builder.

# --------------------------- General configuration -------------------------- #
project = "SARAwater"
copyright = "2026, SARAwater team"
author = "SARAwater developers"

templates_path = ["_templates"]
exclude_patterns = []

# ---------- Sync tutorials from root to docs/tutorials during build --------- #
import shutil
import nbformat
from pathlib import Path

# Automatically sync tutorials from repo root to docs/tutorials during build
DOCS_DIR = Path(__file__).parent
REPO_ROOT = DOCS_DIR.parent.parent
SRC_TUTORIALS = REPO_ROOT / "tutorials"
DST_TUTORIALS = DOCS_DIR / "tutorials"


def _add_download_link(nb_path: Path) -> None:
    """Add a download link for the raw .ipynb at the top of the tutorial notebook."""
    nb = nbformat.read(nb_path, as_version=4)
    link_cell = nbformat.v4.new_markdown_cell(
        f"{{download}}`Download this notebook <{nb_path.name}>`"
    )
    nb.cells.insert(0, link_cell)
    nbformat.write(nb, nb_path)


if SRC_TUTORIALS.exists():
    if DST_TUTORIALS.exists():
        shutil.rmtree(DST_TUTORIALS)
    shutil.copytree(SRC_TUTORIALS, DST_TUTORIALS, dirs_exist_ok=True)
    for nb_path in DST_TUTORIALS.rglob("*.ipynb"):
        _add_download_link(nb_path)

# ------------------------- Extension configurations ------------------------- #
extensions = [
    "myst_nb",  # Handles both Markdown (.md) and Notebooks (.ipynb)
    "autoapi.extension",  # Automatic API reference generation
    "sphinxcontrib.bibtex",  # Citation & Bibliography support
    "sphinx.ext.intersphinx",  # Link to external docs (NumPy, Pandas, Python)
    "sphinx.ext.doctest",  # Verify code examples in docstrings
    "sphinx_copybutton",  # Copy button for code blocks
    "sphinx_design",  # Tabs, dropdowns, and cards
]

# MyST Parser / Notebook Settings
myst_enable_extensions = [
    "dollarmath",  # Enable $ inline $ and $$ display $$ math
    "amsmath",  # Enable LaTeX math environments
    "colon_fence",  # Enable ::: directive syntax
    "alert",  # Enable Github-style admonitions (note, warning, etc.)
]

# MyST-NB Notebook Execution Settings
nb_execution_mode = "auto"  # Executes notebooks if missing outputs or updated
nb_execution_allow_errors = False
nb_execution_raise_on_error = True

# Ensures notebook execution working directory is set to each notebook's folder
# (Allows relative paths to data/ and output/ folders to work seamlessly)
nb_execution_in_temp = False

# AutoAPI Configuration
autoapi_type = "python"
autoapi_dirs = ["../../sarawater"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

# Bibliography Configuration
bibtex_bibfiles = ["references.bib"]

# Figure and equation numbering
numfig = True
math_number_all = True

# Intersphinx Configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Ignore code prompts when copying code blocks
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -------------------------- Options for HTML output ------------------------- #
html_theme = "pydata_sphinx_theme"

# PyData Theme Customization
html_theme_options = {
    "github_url": "https://github.com/sara-acqua/sarawater",
    "use_edit_page_button": True,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

html_context = {
    "github_user": "sara-acqua",
    "github_repo": "sarawater",
    "github_version": "main",
    "doc_path": "docs",
}
