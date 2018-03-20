FROM tensorflow/tensorflow:nightly-devel-py3

RUN pip3 install git+https://github.com/martius-lab/l4-optimizer
RUN pip3 install cached_property

# Install tf-data
RUN pip3 install requests tqdm tf-data

ENV NB_USER fru
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p /src && \
    mkdir -p /data && \
    chown fru /src && \
    chown fru /data

USER fru

WORKDIR /src
