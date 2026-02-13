# ==============================================================================
# RTMDet-MotionBert-OpenSim Pipeline
# Video -> 3D Pose -> OpenSim IK -> GLB Animation
# ==============================================================================
# Build:  docker build -t rtm-opensim .
# Run:    docker run --gpus all -v /path/to/videos:/data/input -v /path/to/output:/data/output \
#           rtm-opensim --input /data/input/video.mp4 --height 1.69 --output /data/output
# ==============================================================================

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
        libimage-exiftool-perl \
        xz-utils \
        libxi6 libxkbcommon0 libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

# ── Miniconda ────────────────────────────────────────────────────────────────
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh
ENV PATH=${CONDA_DIR}/bin:${PATH}

# ── Main environment (mmpose, Python 3.10) ───────────────────────────────────
RUN conda create -n mmpose python=3.10 -y
SHELL ["conda", "run", "-n", "mmpose", "/bin/bash", "-c"]

# PyTorch with CUDA 12.1
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core scientific + CV
RUN pip install --no-cache-dir \
    "numpy>=1.24.0,<2.0.0" \
    "scipy>=1.10.0" \
    "opencv-python>=4.8.0" \
    "tqdm>=4.65.0" \
    easydict \
    "pyyaml>=6.0"

# MMPose ecosystem
RUN pip install --no-cache-dir \
    "mmengine>=0.7.0" \
    "mmcv>=2.0.0" \
    "mmdet>=3.0.0" \
    "mmpose>=1.0.0" \
    rtmpose3d \
    xtcocotools

# Pose2Sim (main env uses it for setup files / marker definitions)
RUN pip install --no-cache-dir "pose2sim>=0.10.0"

# ── Pose2Sim environment (OpenSim IK, Python 3.12) ──────────────────────────
SHELL ["/bin/bash", "-c"]
RUN conda create -n Pose2Sim python=3.12 -y \
    && conda install -n Pose2Sim -c opensim-org opensim=4.5.2 -y
RUN conda run -n Pose2Sim pip install --no-cache-dir "pose2sim>=0.10.0"

# ── Blender 5.0 (headless) ──────────────────────────────────────────────────
ENV BLENDER_VERSION=5.0.1
RUN wget -q https://download.blender.org/release/Blender5.0/blender-${BLENDER_VERSION}-linux-x64.tar.xz \
        -O /tmp/blender.tar.xz \
    && mkdir -p /opt/blender \
    && tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 \
    && rm /tmp/blender.tar.xz
ENV BLENDER_PATH=/opt/blender/blender

# ── Project files ────────────────────────────────────────────────────────────
WORKDIR /app

# Copy dependency-light files first (better layer caching)
COPY config/ config/
COPY opensim_setup/ opensim_setup/
COPY scripts/ scripts/
COPY src/ src/
COPY utils/ utils/
COPY models/ models/
COPY *.py ./
COPY *.txt ./
COPY *.blend ./
COPY *.md ./

# Fix Pose2Sim marker bug in the Pose2Sim env
RUN conda run -n Pose2Sim python fix_pose2sim.py || true

# ── Environment variables ────────────────────────────────────────────────────
ENV POSE2SIM_PYTHON=${CONDA_DIR}/envs/Pose2Sim/bin/python
# BLENDER_PATH already set above

# ── Default shell to mmpose env ──────────────────────────────────────────────
SHELL ["conda", "run", "--no-capture-output", "-n", "mmpose", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mmpose", "python", "run_pipeline.py"]
