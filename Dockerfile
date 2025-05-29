FROM nvcr.io/nvidia/pytorch:23.09-py3

# Install QT6 and its dependencies for Nsight Compute GUI
# https://leimao.github.io/blog/Docker-Nsight-Compute/
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    dbus \
    fontconfig \
    gnupg \
    libasound2 \
    libfreetype6 \
    libglib2.0-0 \
    libnss3 \
    libsqlite3-0 \
    libx11-xcb1 \
    libxcb-glx0 \
    libxcb-xkb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxml2 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    libgl1-mesa-glx \
    libxkbfile-dev \
    openssh-client \
    xcb \
    xkb-data \
    libxcb-cursor0 \
    qt6-base-dev && \
    apt-get clean

# Install Warp and its extra dependencies
# https://nvidia.github.io/warp/installation.html
RUN pip install --upgrade pip setuptools wheel && \
    pip install "warp-lang[extras]==1.7.1"

# Configure to run container as non-root user
ARG USERNAME=defaultuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID $USERNAME

USER $USERNAME

WORKDIR /home/$USERNAME/kernel-kitchen
