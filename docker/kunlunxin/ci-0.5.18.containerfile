# Extend the validated Kunlunxin empty runtime with CI-only test dependencies.
# First build docker/kunlunxin/empty-0.5.18.containerfile as sglang-fl:kunlunxin-0.5.18.
#
#   docker build -f docker/kunlunxin/ci-0.5.18.containerfile \
#     -t harbor.baai.ac.cn/plugin/sglang-plugin-fl:0.5.18-kunlunxin-ci .
ARG BASE_IMAGE=sglang-fl:kunlunxin-0.5.18
FROM ${BASE_IMAGE}

RUN python -m pip install --no-deps --no-build-isolation \
        pytest pytest-timeout==2.4.0 pyyaml && \
    python -c "from importlib.metadata import version; \
        assert version('sglang').startswith('0.5.18'); \
        assert version('pytest-timeout') == '2.4.0'"
