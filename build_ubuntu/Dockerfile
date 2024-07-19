
ARG TRITON_VERSION=2.47.0
ARG TRITON_CONTAINER_VERSION=24.06
ARG BASE_IMAGE=gitlab-master.nvidia.com:5005/dl/dgx/cuda:12.5-devel-manylinux--24.06


############################################################################
##  Production stage: Create container with just inference server executable
############################################################################
FROM ${BASE_IMAGE}

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION

ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
LABEL com.nvidia.tritonserver.version="${TRITON_SERVER_VERSION}"

ENV PATH /opt/tritonserver/bin:${PATH}
# Remove once https://github.com/openucx/ucx/pull/9148 is available
# in the min container.
ENV UCX_MEM_EVENTS no

ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2
ENV TRITON_SERVER_GPU_ENABLED    0

# Create a user that can be used to run triton as
# non-root. Make sure that this user to given ID 1000. All server
# artifacts copied below are assign to this user.
ENV TRITON_SERVER_USER=triton-server
RUN userdel tensorrt-server > /dev/null 2>&1 || true \
      && if ! id -u $TRITON_SERVER_USER > /dev/null 2>&1 ; then \
          useradd $TRITON_SERVER_USER; \
        fi \
      && [ `id -u $TRITON_SERVER_USER` -eq 1000 ] \
      && [ `id -g $TRITON_SERVER_USER` -eq 1000 ]

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

# Common dependencies. FIXME (can any of these be conditional? For
# example libcurl only needed for GCS?)
# RUN apt-get update \
#       && apt-get install -y --no-install-recommends \
#               clang \
#               curl \
#               dirmngr \
#               git \
#               gperf \
#               libb64-0d \
#               libcurl4-openssl-dev \
#               libgoogle-perftools-dev \
#               libjemalloc-dev \
#               libnuma-dev \
#               libre2-9 \
#               software-properties-common \
#               wget \
#                \
#       && rm -rf /var/lib/apt/lists/*
RUN yum update && \
      yum install -y rapidjson-devel \
            protobuf-devel \
            re2-devel \
            zlib-devel \
            libarchive-devel \
            libxml2-devel \
            numactl-devel \
            openssl-devel \
            libcurl-devel \
            libb64-devel \
            gperftools-devel



# Set TCMALLOC_RELEASE_RATE for users setting LD_PRELOAD with tcmalloc
ENV TCMALLOC_RELEASE_RATE 200

WORKDIR /opt/tritonserver
RUN rm -fr /opt/tritonserver/*
ENV NVIDIA_PRODUCT_NAME="Triton Server"
COPY docker/entrypoint.d/ /opt/nvidia/entrypoint.d/

COPY docker/cpu_only/ /opt/nvidia/
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]

ENV NVIDIA_BUILD_ID <unknown>
LABEL com.nvidia.build.id=<unknown>
LABEL com.nvidia.build.ref=

WORKDIR /opt
COPY --chown=1000:1000 build/install tritonserver

WORKDIR /opt/tritonserver
COPY --chown=1000:1000 NVIDIA_Deep_Learning_Container_License.pdf .

