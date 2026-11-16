# Contributing

Thank you for your interest in contributing to eval-gate.

## Setup

Clone the repository and sync the dependencies (including the docs group):

```sh
git clone https://github.com/GuillaumeZahnd/eval-gate
cd eval-gate
uv sync --group docs
```

## Running the tests

```sh
uv run pytest
```

## Writing docstrings

When adding or updating Python functions, please use NumPy-style docstrings so they render properly in the API reference:

```python
def example_function(parameter: type) -> return_type:
    """Short summary of the function.

    Parameters
    ----------
    parameter : type
        Description of the parameter.

    Returns
    -------
    return_type
        Description of the return value.
    """
```

## Building the documentation

This project uses [MkDocs](https://www.mkdocs.org/) with `mkdocstrings` for API reference generation. Documentation is automatically built and published to GitHub Pages via GitHub Actions whenever changes are merged into the `main` branch.

To build and preview the documentation locally:

1. Run the live-reload development server:

```sh
uv run mkdocs serve
```

2. Open `http://127.0.0.1:8000` in your browser.

3. Verify the build (runs in strict mode to catch broken links or invalid docstrings):

```sh
uv run mkdocs build --strict
```
