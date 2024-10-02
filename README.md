# The algorithmic differentation tool for DOLFIN/FEniCS

The full documentation is available [here](https://dolfin-adjoint.github.io/dolfin-adjoint/)

# Installation

First install [FEniCS](http://fenicsproject.org).
Then install `dolfin-adjoint` with:

```bash
python3 -m pip install git+https://github.com/dolfin-adjoint/dolfin-adjoint.git@main
```

or using the [Pypi-package](https://pypi.org/project/dolfin-adjoint/) for the latest stable release

```bash
python3 -m pip install dolfin-adjoint
```

# Reporting bugs

If you found a bug, create an [issue](https://github.com/dolfin-adjoint/dolfin-adjoint/issues/new)

# Contributing

We love pull requests from everyone.

Fork, then clone the repository:

```bash
git clone https://github.com/{your_username}/dolfin-adjoint.git
```

Make sure the tests pass:

```bash
python3 -m pytest tests
```

Make your change. Add tests for your change. Make the tests pass:

```bash
python3 -m pytest tests
```

Push to your fork and [submit a pull request](https://github.com/dolfin-adjoint/dolfin-adjoint/pulls).
At this point you're waiting on us. We may suggest
some changes or improvements or alternatives.

Some things that will increase the chance that your pull request is accepted:

- Write tests.
- Add Python doc-strings that follow the [Google Style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
- Write good commit and pull request message.

# License

This software is licensed under the [GNU LGPL v3](./LICENSE).
