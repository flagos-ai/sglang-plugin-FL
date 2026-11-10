FROM scratch

LABEL org.opencontainers.image.description="FlagTree 0.7.1a1+xpu3.6 compiler artifact extracted from the validated Kunlunxin SGLang CI stack"
LABEL org.opencontainers.image.source="harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl@sha256:b0faa6a570501d8445aeb6fd78c5bbe58499b2573d5316a3d2902ed994f6245a"
LABEL org.opencontainers.image.base="harbor.baai.ac.cn/flagos-runtime/flagos-runtime-kunlunxin-xre5.37.1@sha256:679182a04d187562a3e23bacb9d0eb876dabf09ac0c0a875c2b4a141b64cac69"

COPY site-packages/ /vendor/site-packages/
COPY bin/ /vendor/bin/
COPY remove-paths.txt /vendor/remove-paths.txt
