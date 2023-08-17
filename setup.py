from itertools import chain
from setuptools import setup

extras = {
    'moola': ['moola>=0.1.6'],
    'test': ['pytest>=3.10', 'flake8', 'coverage'],
    'visualisation': ['tensorflow', 'protobuf',
                      'networkx', 'pygraphviz'],
    'meshing': ['pygmsh', 'meshio'],
}
# 'all' includes all of the above
extras['all'] = list(chain(*extras.values()))

setup(name='dolfin_adjoint',
      version='2023.2.0',
      description='High-level automatic differentiation library for FEniCS.',
      author='Jørgen S. Dokken',
      author_email='dokken@simula.no',
      packages=['fenics_adjoint',
                'fenics_adjoint.types',
                'fenics_adjoint.blocks',
                'dolfin_adjoint'],
      package_dir={'fenics_adjoint': 'fenics_adjoint',
                   'dolfin_adjoint': 'dolfin_adjoint'},
      install_requires=['scipy>=1.0', 'pyadjoint@git+https://github.com/dolfin-adjoint/pyadjoint'],
      extras_require=extras
      )
