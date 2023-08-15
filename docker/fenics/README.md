# Docker for dolfin-adjoint

This repository contains the scripts for building various Docker
images for dolfin-adjoint (<http://dolfin-adjoint.org>).

The dolfin-adjoint containers build off of Docker containers
maintained by Simula Research Laboratory <https://github.com/scientificcomputing/packages/>.

## Introduction

To install Docker for your platform (Windows, Mac OS X, Linux, cloud platforms,
etc.), follow the instructions at
<https://docs.docker.com/engine/installation/>.

Once you have Docker installed, you can run any of the images below using the
following command:

```bash
docker run -ti ghcr.io/dolfin-adjoint
```

To start with you probably want to try the `dolfin-adjoint` image which
includes a full stable version of FEniCS and dolfin-adjoint with PETSc, SLEPc,
petsc4py and slepc4py already compiled. The optimisation routines from scipy,
moola and TAO are included. For licensing reasons, we cannot include
IPOPT which depends on the non-Open Source HSL routines.

If you want to share your current working directory into the container use
the following command:

```bash
docker run -ti -v $(pwd):/root/shared ghcr.io/dolfin-adjoint
```


## Building images

Images are hosted on Github, and are automatically built in the cloud on from
the Dockerfiles in this repository. 

To build the images locally, go to the root of the repo, and call
1. Build docker container with `docker buildx build --platform=NAME_OF_PLATFORM -t dev-dolfin-adjoint .`
where `NAME_OF_PLATFORM` should either be `linux/amd64` or `linux/arm64`
2. Start docker container with `docker run -it -v $(pwd):/root/shared dev-dolfin-adjoint`

## Authors

* Jørgen S. Dokken (<dokken@simula.no>)
