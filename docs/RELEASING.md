# Release procedure

## Local checks

1. Run tests on Python 3.10–3.14 and on each supported operating system.
2. Build a fresh wheel and sdist. Run `twine check --strict` on both archives.
3. Install the wheel in a fresh environment outside the checkout, run the
   installed CLI tests, and rebuild/install the sdist separately.
4. Inspect archive contents; no data, logs, caches, credentials, or personal
   configuration should be included.
5. Confirm README, algorithm specification, changelog, and validation report
   describe the same version and method selection.

The CI configuration performs Linux/macOS/Windows tests and wheel installation.
Local execution of a CI-equivalent command does not mean a hosted CI run passed.

## Publication configuration

- The SPDX license expression and LICENSE file are included in package metadata.
- Maintainer contact is present. Retain the source copyright notice.
- Create/verify the standalone repository and then add real Repository, Issues,
  and Documentation URLs to `pyproject.toml`. Do not publish invented URLs.
- Verify package-name availability with PyPI at upload time; an API 404 is not
  a reservation and does not establish that an account can register that name.
- Configure a PyPI/TestPyPI API token or Trusted Publisher using the account owner.
  No credentials belong in this repository.

After approval for external publication:

```bash
python -m twine upload --repository testpypi dist/*
```

Install the exact TestPyPI artifact into a new environment. Install dependencies
from PyPI first, then install fastmagma with `--no-deps` and TestPyPI as its sole
index. Run the installed-package smoke tests again. Production upload is separate:

```bash
python -m twine upload dist/*
```

Only a successful upload and subsequent installation from the registry verify
publication. Twine's local metadata checks do not validate scientific correctness,
account permissions, package-name ownership, or platform compatibility.
