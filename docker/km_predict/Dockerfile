# ==== Build ====
FROM tensorflow/tensorflow:2.15.0 AS builder

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

WORKDIR /home

# Install pre-dependencies
RUN apt-get update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
	software-properties-common git
# Install build tools and Mapserver dependencies.
RUN apt-get update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake gcc g++ python3-pip libgdal-dev gdal-bin libexpat1-dev libgraphicsmagick++1-dev libcppunit-dev

# Install cget
RUN pip3 install cget

# Get cm-vsm
RUN git clone --recurse-submodules https://github.com/kappazeta/cm-vsm.git && \
    cd cm-vsm && git checkout v0.3.5
# Install dependencies via cget
RUN cd /home/cm-vsm/vsm/build && \
    cget install -DJSON_BuildTests=Off
# Prepare for build
RUN cd /home/cm-vsm/vsm/build && \
    cmake ..
# Build
RUN cd /home/cm-vsm/vsm/build && \
    make -j4 && \
    make install

# ==== Runtime ====
FROM tensorflow/tensorflow:2.15.0 as runtime

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/:/usr/local/lib/x86_64-linux-gnu/

WORKDIR /home

# Install system dependencies.
RUN apt-get update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    lsb-release wget unzip vim-tiny libgdal30 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy binaries.
COPY --from=builder /usr/bin/ /usr/bin/
COPY --from=builder /usr/lib/ /usr/lib/
RUN ldconfig

# Install AWS CLI.
RUN wget "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -O /home/awscliv2.zip && \
    unzip /home/awscliv2.zip && \
    rm /home/awscliv2.zip && \
    ./aws/install && \
    rm -rf ./aws

# Install dependencies.
RUN python3 -m pip install --upgrade pip
# NOTE:: Starting from pillow==10.2.0, TiffImagePlugin throws
#   NotImplementedError: multistrip support not yet implemented
RUN pip3 install --no-cache-dir scikit-image==0.22.0 netcdf4==1.6.5 rasterio==1.3.9 \
    pytest==7.4.3 scikit-learn==1.3.2 pillow==10.1.0 boto3 sentinelhub[AWS]

# Get km-predict
RUN git clone --recurse-submodules https://github.com/kappazeta/km_predict.git && cd km_predict && git checkout v2.3
RUN mkdir -p /home/km_predict/weights && \
    wget -q http://kappamask.s3-website.eu-central-1.amazonaws.com/model_weights/2022-09-13/l1c_deeplabv3plus.hdf5 -O /home/km_predict/weights/l1c_deeplabv3plus.hdf5 && \
    wget -q http://kappamask.s3-website.eu-central-1.amazonaws.com/model_weights/2022-09-13/l2a_deeplabv3plus.hdf5 -O /home/km_predict/weights/l2a_deeplabv3plus.hdf5

COPY get_s3.py km_s3.sh km_local.sh km_test.sh copernicus_s3_utils.py /home/
RUN chmod +x /home/km_s3.sh /home/km_local.sh /home/km_test.sh && \
    ln -s /home/km_s3.sh /usr/local/bin/km_s3 && \
    ln -s /home/km_local.sh /usr/local/bin/km_local && \
    ln -s /home/km_test.sh /usr/local/bin/km_test

ENTRYPOINT ["km_s3"]
# ENTRYPOINT ["km_test"]
# ENTRYPOINT ["km_local"]