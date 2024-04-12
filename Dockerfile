FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Add python3 in cuda image
ENV DEBIAN_FRONTEND=non-interactive
RUN apt-get update -q -y && apt-get install -q -y --no-install-recommends python3-pip
RUN pip3 install --upgrade pip

# Setup doc-ufcn library
WORKDIR /src
COPY LICENSE README.md pyproject.toml /src/
COPY doc_ufcn /src/doc_ufcn
RUN pip3 install .[training] --no-cache-dir
