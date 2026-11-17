# NVIDIA SGLang 0.5.18 CI image.
#
# The build context must stage the two reviewed wheel artifacts under
# artifacts/. All dependency resolution comes from their METADATA; tests and
# validation run against the completed image, never during image construction.
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3@sha256:edf29b9a28badaf72fc75fd0e057325df9610694914cca3f571e587038bf17e8
FROM ${BASE_IMAGE}

COPY artifacts/sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl \
     /tmp/wheels/sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl
COPY artifacts/sglang_fl-0.1.0-py3-none-any.whl \
     /tmp/wheels/sglang_fl-0.1.0-py3-none-any.whl

RUN python3 -m pip install --break-system-packages \
      /tmp/wheels/sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl \
      '/tmp/wheels/sglang_fl-0.1.0-py3-none-any.whl[nvidia]'
