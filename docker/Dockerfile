FROM tensorflow/tensorflow:nightly-devel-gpu-py3

RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp35-cp35m-linux_x86_64.whl 
RUN pip3 install torchvision

RUN pip3 install git+https://github.com/martius-lab/l4-optimizer
RUN pip3 install gpustat
RUN pip3 install cached_property

# Download celeba helper
RUN pip3 install bs4 lxml

# Install tf-data
RUN pip3 install requests tqdm tf-data

# Use tensorboard nightly
RUN pip3 install --upgrade tb-nightly

ENV NB_USER fru
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p /src && \
    mkdir -p /data && \
    chown fru /src && \
    chown fru /data

USER fru

WORKDIR /src
