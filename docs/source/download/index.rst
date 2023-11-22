:orphan:

.. _download:

*************************
Installing dolfin-adjoint
*************************

**Note**: If you are looking to install the (deprecated) dolfin-adjoint/libadjoint library, visit the `dolfin-adjoint/libadjoint`_ webpage.

:ref:`dolfin-adjoint-difference`

.. _dolfin-adjoint/libadjoint: http://dolfin-adjoint-doc.readthedocs.io/en/latest/download/index.html


Docker images (all platforms)
=============================

`Docker <https://www.docker.com>`_ allows us to build and ship
consistent high-performance dolfin-adjoint installations with FEniCS for almost any
platform. To get started, follow these 2 steps:

#. Install Docker. Mac and Windows users should install the `Docker
   Toolbox <https://www.docker.com/products/docker-toolbox>`_ (this is
   a simple one-click install) and Linux users should `follow these
   instructions <https://docs.docker.com/linux/step_one/>`_.

If running on Mac or Windows, make sure you run the following 
commands inside the Docker Quickstart Terminal.

dolfin-adjoint with FEniCS:
---------------------------

You can run `dolfin-adjoint` with the pre-built images at::

    ghcr.io/dolfin-adjoint/dolfin-adjoint

First the FEniCS Docker script::

    docker run -ti -v $(pwd):/root/shared --name=name_of_container ghcr.io/dolfin-adjoint/dolfin-adjoint

which can then be restarted at an later instance with::

    docker container start -i name_of_container

To update the image call::

    docker pull ghcr.io/dolfin-adjoint/dolfin-adjoint

and create a new docker container.



PIP (all platforms)
================================

Install dolfin-adjoint and its Python dependencies with pip:

.. code-block:: bash

    python3 -m pip install dolfin-adjoint

Test your installation by running:

.. code-block:: bash

    python3 -c "import fenics_adjoint"


Optional dependencies:
----------------------

- `IPOPT`_ and Python bindings (`cyipopt`_): This is the best available open-source optimisation algorithm. Strongly recommended if you wish to solve :doc:`PDE-constrained optimisation problems <../documentation/optimisation>`. Make sure to compile IPOPT against the `Harwell Subroutine Library`_.

- `Moola`_: A set of optimisation algorithms specifically designed for :doc:`PDE-constrained optimisation problems <../documentation/optimisation>`.

- `Optizelle`_: An Open Source Software Library Designed To Solve General Purpose Nonlinear Optimization Problems.

.. _FEniCS: http://fenicsproject.org
.. _Optizelle: http://www.optimojoe.com/products/optizelle
.. _SLEPc: http://www.grycap.upv.es/slepc/
.. _IPOPT: https://projects.coin-or.org/Ipopt
.. _cyipopt: https://github.com/matthias-k/cyipopt
.. _moola: https://github.com/funsim/moola
.. _Harwell Subroutine Library: http://www.hsl.rl.ac.uk/ipopt/
.. _their installation instructions: http://fenicsproject.org/download


Source code
===========

The source code of `dolfin-adjoint` is available on https://github.com/dolfin-adjoint/dolfin-adjoint.
