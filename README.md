# The algorithmic differentation tool pyadjoint and add-ons

The full documentation is available [here](http://pyadjoint.readthedocs.io)

# Installation
First install [FEniCS](http://fenicsproject.org)
Then install the pyadjoint with:
    python3 -m pip install git+https://github.com/dolfin-adjoint/dolfin-adjoint.git@main

# Reporting bugs

If you found a bug, create an [issue].

[issue]: https://github.com/dolfin-adjoint/dolfin-adjoint/issues/new

# Contributing

We love pull requests from everyone. 

Fork, then clone the repository:

    git clone https://github.com/dolfin-adjoint/dolfin-adjoint.git

Make sure the tests pass:

    python3 -m pytest tests

Make your change. Add tests for your change. Make the tests pass:

    python3 -m pytest tests

Push to your fork and [submit a pull request][pr].

[pr]: https://github.com/dolfin-adjoint/dolfin-adjoint/pulls

At this point you're waiting on us. We may suggest
some changes or improvements or alternatives.

Some things that will increase the chance that your pull request is accepted:

* Write tests.
* Add Python docstrings that follow the [Google Style][style].
* Write good commit and pull request message.

[style]: http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

# License
This software is licensed under the [GNU LGPL v3][license].

[license]: https://github.com/dolfin-adjoint/dolfin-adjoint/raw/main/LICENSE
