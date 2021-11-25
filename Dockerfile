FROM nvidia/cuda:11.2.2-base-ubuntu20.04

# Add python3 in cuda image
ENV DEBIAN_FRONTEND=non-interactive
RUN apt-get update -q -y && apt-get install -q -y --no-install-recommends python3-pip

# Setup doc-ufcn library
WORKDIR /src
COPY requirements.txt LICENSE setup.py MANIFEST.in README.md VERSION /src/
COPY doc_ufcn /src/doc_ufcn
RUN pip3 install .
