FROM scratch

LABEL org.opencontainers.image.description="FlagTree 0.7.1a1+xpu3.6 compiler artifact extracted from the validated Kunlunxin SGLang CI stack"
LABEL org.opencontainers.image.source="harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl@sha256:b0faa6a570501d8445aeb6fd78c5bbe58499b2573d5316a3d2902ed994f6245a"

COPY flagtree/ /vendor/site-packages/flagtree/
COPY flagtree-0.7.1a1+xpu3.6.dist-info/ /vendor/site-packages/flagtree-0.7.1a1+xpu3.6.dist-info/
COPY triton/ /vendor/site-packages/triton/
COPY bin/ /vendor/bin/
