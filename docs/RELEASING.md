# Release procedure

## Local checks

1. Run tests on Python 3.10–3.14 and on each supported operating system.
2. Build fresh platform wheels and an sdist. Run `twine check --strict` on every archive.
3. Install the wheel in a fresh environment outside the checkout, run the
   installed CLI tests, and rebuild/install the sdist separately.
4. Inspect archive contents; no data, logs, caches, credentials, or personal
   configuration should be included.
5. Confirm README, algorithm specification, changelog, and validation report
   describe the same version and method selection.

The package includes two required CPython C extensions, so wheels are specific
to Python ABI, operating system, and architecture; do not publish a `py3-none-any`
wheel. Build and test each wheel on its target platform. A locally built Linux
wheel is not automatically a portable manylinux wheel. Include both C sources
and `setup.py` in the sdist, and verify installation with a C compiler present.
Binary-wheel users do not need a compiler.

The CI configuration performs Linux/macOS/Windows tests and wheel installation.
Local execution of a CI-equivalent command does not mean a hosted CI run passed.

## Publication configuration

- The SPDX license expression and LICENSE file are included in package metadata.
- Maintainer contact is present. Retain the source copyright notice.
- Verify that the Repository, Issues, and Documentation URLs in `pyproject.toml`
  resolve to the standalone `cafferychen777/magma-py` repository.
- Verify package-name availability with PyPI at upload time; an API 404 is not
  a reservation and does not establish that an account can register that name.
- Configure a PyPI/TestPyPI API token or Trusted Publisher using the account owner.
  No credentials belong in this repository.

After approval for external publication:

```bash
python -m twine upload --repository testpypi dist/*
```

Install the exact TestPyPI artifact into a new environment. Install dependencies
from PyPI first, then install `magma-py` with `--no-deps` and TestPyPI as its sole
index. Run the installed-package smoke tests again. Production upload is separate:

```bash
python -m twine upload dist/*
```

Only a successful upload and subsequent installation from the registry verify
publication. Twine's local metadata checks do not validate scientific correctness,
account permissions, package-name ownership, or platform compatibility.
