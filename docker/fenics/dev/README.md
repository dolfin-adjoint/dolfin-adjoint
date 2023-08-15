# Installation with docker


1. Install docker: https://docs.docker.com/engine/installation/
2. Build docker container with
       `docker buildx build --platform=NAME_OF_PLATFORM -t dev-dolfin-adjoint .`
where `NAME_OF_PLATFORM` should either be `linux/amd64` or `linux/arm64`
3. Start docker container with
       `docker run -it -v $(pwd):/root/shared dev-dolfin-adjoint`
