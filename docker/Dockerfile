# Builds a Docker image with dolfin-adjoint dev version built from
# git sources.
#
# Authors:
# Simon W. Funke <simon@simula.no>
# Jack S. Hale <jack.hale@uni.lu>
# Jørgen S. Dokken <dokken@simula.no>

FROM ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30
ARG TARGETPLATFORM

RUN apt-get -qq update && \
    apt-get -y install libjsoncpp-dev curl && \
    apt-get -y install python3-dev graphviz libgraphviz-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN /bin/bash -l -c "python3 -m pip install scipy 'numpy<2.0.0'"

WORKDIR /src/

ARG IPOPT_VERSION=3.14.4
ARG MOOLA_BRANCH="master"

# Pull IPOPT source code
RUN curl -O https://www.coin-or.org/download/source/Ipopt/Ipopt-${IPOPT_VERSION}.tar.gz && \
    tar -xvf Ipopt-${IPOPT_VERSION}.tar.gz

WORKDIR /src/Ipopt-releases-${IPOPT_VERSION}

# Build Ipopt for correct architecture
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; \
    then ./configure --build=aarch64-unknown-linux-gnu -with-lapack="-llapack" --prefix="/usr/local"  --enable-debug --enable-shared --with-mumps="yes" --with-mumps-cflags="-I${PETSC_DIR}/${PETSC_ARCH}/include/ -I/usr/include/aarch64-linux-gnu/mpich/" ; \
    make install; \
    elif [ "$TARGETPLATFORM" = "linux/amd64" ]; \
    then ./configure --build=amd64-unknown-linux-gnu --with-lapack="-llapack" --prefix="/usr/local"  --enable-debug --enable-shared --with-mumps="yes" --with-mumps-cflags="-I${PETSC_DIR}/${PETSC_ARCH}/include/ -I/usr/include/x86_64-linux-gnu/mpich/" ; \
    make install; \
    else echo "Unknown TARGETPLATFORM: ${TARGETPLATFORM}, it should be either 'linux/amd64' or 'linux/arm64'"; \
    fi


# Build pyipopt
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
WORKDIR /src
RUN python3 -m pip install cython && \
    python3 -m pip install ipopt

# Install moola
RUN python3 -m pip install git+https://github.com/funsim/moola@${MOOLA_BRANCH}

# Install ROL
ENV LD_LIBRARY_PATH=/usr/lib/python3/dist-packages/roltrilinos/lib/:${LD_LIBRARY_PATH}
RUN python3 -m pip install roltrilinos && \
    python3 -m pip install ROL

# Install dolfin-adjoint
COPY . /src/dolfin-adjoint
WORKDIR /src/dolfin-adjoint

RUN python3 -m pip install -e .[test]

# Run import test
WORKDIR /src/dolfin-adjoint/tests
RUN python3 -c "import dolfin; import dolfin_adjoint; import pyadjoint.ipopt; import cyipopt; import moola"

WORKDIR /src/dolfin-adjoint/examples/poisson-topology
RUN python3 poisson-topology.py

# NOTE: For debugging add in tests
# RUN cd tests/pyadjoint && python3 -m pytest -xvs .
# RUN cd tests/fenics_adjoint && python3 -m pytest -xvs .
# RUN cd tests/migration && python3 -m pytest -xvs .

WORKDIR /src