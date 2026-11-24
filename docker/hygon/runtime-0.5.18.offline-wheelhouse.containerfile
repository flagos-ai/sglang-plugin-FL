# SGLang 0.5.18 two-wheel DEVELOPMENT CANDIDATE: the release recipe
# (runtime-0.5.18.containerfile) resolved offline from a wheelhouse. TEMP,
# never a release input; the base is always passed as a verified digest.
# Evidence and boundaries: docker/hygon/README.md
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Same install as the release recipe, resolved only from the wheelhouse
# (--no-index --find-links) with no network. Evidence: docker/hygon/README.md
RUN --network=none \
    --mount=type=bind,source=artifacts,target=/tmp/artifacts,readonly \
    --mount=type=bind,source=wheelhouse,target=/wheelhouse,readonly \
    /usr/local/bin/uv pip install --python /flagos/bin/python3 \
      --no-cache \
      --no-index \
      --find-links /wheelhouse \
      --constraints /tmp/artifacts/runtime-constraints.txt \
      --excludes /tmp/artifacts/runtime-0.5.18.excludes.txt \
      '/tmp/artifacts/xgrammar-0.2.1+flagos.hcu1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl' \
      '/tmp/artifacts/sglang-0.5.18+flagos.hcu1-cp310-cp310-linux_x86_64.whl[srt_empty,runtime_common]' \
      '/tmp/artifacts/sglang_fl-0.1.0-py3-none-any.whl[hygon]'
