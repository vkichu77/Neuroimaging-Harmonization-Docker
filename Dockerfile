# =============================================================================
# BASE UBUNTU
# =============================================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# =============================================================================
# SYSTEM DEPENDENCIES
# =============================================================================
RUN apt-get update && apt-get install -y \
    git wget curl build-essential cmake pkg-config \
    python3 python3-pip python3-dev \
    libeigen3-dev libfftw3-dev libtiff5-dev \
    libpng-dev libjpeg-dev libgl1-mesa-glx libglu1-mesa libxt-dev \
    unzip nano vim \
    r-base r-base-dev \
    && apt-get clean

# =============================================================================
# INSTALL MINICONDA
# =============================================================================
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"

# =============================================================================
# INSTALL ENVIRONMENT (Python + R)
# =============================================================================
COPY environment.yml /tmp/environment.yml

RUN conda install -y mamba -n base -c conda-forge && \
    mamba env update -n base -f /tmp/environment.yml && \
    conda clean -a -y

# Ensure pip inside same env
RUN pip install --upgrade pip

# install DIPY 
RUN pip install dipy

# =============================================================================
# INSTALL longCombat in R
# =============================================================================
COPY install_R_packages.R /tmp/install_R_packages.R
RUN Rscript /tmp/install_R_packages.R

# =============================================================================
# INSTALL MRtrix3
# =============================================================================
RUN git clone https://github.com/MRtrix3/mrtrix3.git /opt/mrtrix3 && \
    cd /opt/mrtrix3 && ./configure && ./build -j $(nproc)

ENV PATH="/opt/mrtrix3/bin:${PATH}"

# =============================================================================
# GPU SUPPORT 
# =============================================================================
RUN apt-get install -y nvidia-cuda-toolkit
ENV CUDA_HOME="/usr/lib/cuda"

# =============================================================================
# CLONE dMRI HARMONIZATION REPO
# =============================================================================
RUN mkdir -p /opt/harmonization_repos && \
    git clone https://github.com/MGH-Harmonization/dMRIharmonization.git \
         /opt/harmonization_repos/dMRIharmonization

ENV PYTHONPATH="/opt/harmonization_repos/dMRIharmonization:${PYTHONPATH}"

# =============================================================================
# PROJECT SCRIPTS INTO IMAGE
# =============================================================================
WORKDIR /workspace

# Copy all scripts, config, and notebooks
COPY scripts/ /workspace/scripts/
COPY config/ /workspace/config/
COPY notebooks/ /workspace/notebooks/

# Make scripts executable
RUN chmod +x /workspace/scripts/*.py || true
RUN chmod +x /workspace/scripts/*.R || true

# =============================================================================
# ENTRYPOINT
# =============================================================================
CMD ["/bin/bash"]
