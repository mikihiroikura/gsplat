FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV TZ=Europe/Rome
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y tzdata

RUN apt update && \
    apt install -y git wget curl \
    libgl1-mesa-dev \
    libglib2.0-0

# Add library for CUDA
RUN apt update && \
    apt install -y libglm-dev

# Add User ID and Group ID
ARG UNAME=gsplat
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME

# Add User into sudoers, can run sudo command without password
RUN apt update && apt install -y sudo
RUN usermod -aG sudo ${UNAME}
RUN echo "${UNAME} ALL=(ALL) NOPASSWD:ALL" | tee /etc/sudoers.d/${UNAME}

ARG CODE_DIR=/home/${UNAME}

### Install gsplat code
RUN mkdir -p ${CODE_DIR}/src && cd ${CODE_DIR}/src && \
    git clone https://github.com/nerfstudio-project/gsplat.git

### Change owner
RUN chown -R $UNAME:$UNAME ${CODE_DIR}/src

### Switch user
USER $UNAME
WORKDIR ${CODE_DIR}

### Install uv and make virtual environment
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/gsplat/.local/bin:$PATH"
RUN cd ${CODE_DIR}/src/gsplat && \
    uv sync
