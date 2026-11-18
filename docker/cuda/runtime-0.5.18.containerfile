# NVIDIA SGLang 0.5.18 CI image.
#
# The build context must stage the two reviewed wheel artifacts under
# artifacts/ plus the one-line resolver exclusion list. All dependency
# resolution comes from their METADATA; tests and validation run against
# the completed image, never during image construction.
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3@sha256:edf29b9a28badaf72fc75fd0e057325df9610694914cca3f571e587038bf17e8
FROM ${BASE_IMAGE}

COPY artifacts/sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl \
     /tmp/wheels/sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl
COPY artifacts/sglang_fl-0.1.0-py3-none-any.whl \
     /tmp/wheels/sglang_fl-0.1.0-py3-none-any.whl
COPY artifacts/runtime-0.5.18.excludes.txt \
     /tmp/runtime-0.5.18.excludes.txt

# Normal resolver install of both wheels with the Runtime's own uv. The
# excludes file is the single deliberate platform exception: the Runtime
# provides the Triton compiler as the /opt/flagtree module tree, so the
# standalone triton distribution is dropped from resolution (nothing is
# installed, uninstalled, or stubbed for it). /flagos is a uv-managed
# virtual environment, so no system-package override is needed.
RUN /usr/local/bin/uv pip install \
      --python /flagos/bin/python3 \
      --excludes /tmp/runtime-0.5.18.excludes.txt \
      /tmp/wheels/sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl \
      '/tmp/wheels/sglang_fl-0.1.0-py3-none-any.whl[nvidia]'
