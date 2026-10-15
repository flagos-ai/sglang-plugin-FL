# Extend the validated HCU runtime with CI-only dependencies.
# First build docker/hygon/empty-0.5.18.containerfile as sglang-fl:hygon-0.5.18.
ARG BASE_IMAGE=sglang-fl:hygon-0.5.18
FROM ${BASE_IMAGE}
RUN python -m pip install --no-deps --no-build-isolation pytest-timeout==2.4.0 && \
    python /opt/sglang-plugin-fl/.github/scripts/hygon/verify_environment.py \
        --plugin-root /opt/sglang-plugin-fl \
        --baseline /opt/hygon-vendor-baseline.json --metadata-only && \
    python -c "from importlib.metadata import version; assert version('sglang') == '0.5.18'; assert version('pytest-timeout') == '2.4.0'"
