# Installation validation

Validated on 18 October 2026 in a new Linux x86-64 CPU virtual environment with Python 3.12.15, PyTorch 2.3.1+cpu, NumPy 1.26.4, Selene 0.6.0, and Captum 0.7.0. No research environment or inherited site packages were used. Python development headers and GCC were available for the Selene/tabix extension builds. Exact resolved packages are in `requirements-tested.txt`.

Passed checks:

- Six installation regression tests, including input checksums and feature order, working-directory-independent configs, normalized genome-reader equivalence, useful placeholder-path errors, CLI help, and propagation of motif-command failures.
- The quickstart downloaded the public checkpoint, verified SHA-256, and produced finite `(16, 919)` predictions. Maximum absolute difference from the original reference predictions was 9.24 × 10⁻⁷.
- DeepLIFT produced finite `(2, 9, 1000)` attributions. The pre-encoded-input and genome-reader routes agreed within 3.73 × 10⁻⁹.
- A one-target CPU optimizer step passed, and the separate Selene training example completed two steps plus evaluation with both positive and negative test labels.
- The installed, bundled TF-MoDISco CLI exposed the shape-specific `--nofliprc` option.

The genome-scale benchmark was not retrained for these software checks. The tiny fixture is not a benchmark or a meaningful motif-discovery cohort. The CI workflow repeats the documented installation and example commands in a clean GitHub runner.
