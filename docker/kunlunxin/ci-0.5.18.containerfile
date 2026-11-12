# Extend the validated Kunlunxin empty runtime with CI-only test dependencies.
# First build docker/kunlunxin/empty-0.5.18.containerfile as sglang-fl:kunlunxin-0.5.18.
#
#   docker build -f docker/kunlunxin/ci-0.5.18.containerfile \
#     -t harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:0.5.18-kunlunxin-ci .
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:kunlunxin-0.5.18-202611121806
FROM ${BASE_IMAGE}

ARG PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

RUN python -m pip install --index-url "$PYPI_INDEX" --no-build-isolation \
        pytest pytest-cov pytest-timeout pytest-json-report pytest-metadata numpy requests ruff pyyaml && \
    python -c "from importlib.metadata import version; \
        assert version('sglang').startswith('0.5.18')"