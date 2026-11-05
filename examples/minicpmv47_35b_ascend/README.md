# MiniCPM-V 4.7 35B-A3B on Ascend 910C

This example covers BF16 serving with SGLang 0.5.11, sglang-fl 0.1.0 and
tensor parallelism 2 on Ascend 910C. Supply the MiniCPM-V 4.7 35B-A3B
checkpoint as `MODEL_PATH`; weights are not included in this repository.
The checkpoint is described by the [FlagOS model card](https://www.modelscope.cn/models/FlagRelease/MiniCPM-V-4.7-35B-A3B-BF16-ascend-FlagOS).

The checkpoint must include `chat_template.jinja`. SGLang 0.5.11 otherwise
selects its legacy `minicpmv` conversation template for this model. That
template inserts an extra image closing tag, so requests with two or more
images can create reversed image bounds and fail during canvas mRoPE.
Always pass the checkpoint's own template explicitly. This uses SGLang's
public launch argument and does not change other models' chat formatting.

```bash
export MODEL_PATH=/path/to/MiniCPM-V-4.7-35B-A3B
export ASCEND_RT_VISIBLE_DEVICES=8,9  # Choose two available Ascend devices.
export USE_FLAGGEMS=1
export SGLANG_PLUGINS=sglang_fl
export SGLANG_FL_FLAGOS_BLACKLIST='argmax,index_put_,_index_put_impl_,index_select,index_copy_,max,sum,fill_scalar_,index,floor_divide,gather,conv1d,conv2d'
export SGLANG_FL_PREFER=vendor
export SGLANG_FL_PER_OP='silu_and_mul=flagos'
export SGLANG_FL_RECV_COALESCE_MS=100
export SGLANG_FL_CAUSAL_CONV_PREFILL_BUCKET=640
export SGLANG_FL_CAUSAL_CONV_PREFILL_IMPL=tiled_bucketed
export SGLANG_FL_CAUSAL_CONV_PREFILL_BUCKET_QUANTUM=128
export SGLANG_FL_CAUSAL_CONV_PREFILL_FAST_MIN_TOKENS=1024

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name MiniCPM-V-4.7-35B-A3B \
  --dtype bfloat16 \
  --host 0.0.0.0 --port 31162 --device npu \
  --tensor-parallel-size 2 \
  --trust-remote-code --enable-multimodal \
  --chat-template "$MODEL_PATH/chat_template.jinja" \
  --context-length 262144 \
  --mem-fraction-static 0.68 \
  --chunked-prefill-size 8192 \
  --max-running-requests 64 \
  --disable-radix-cache \
  --cuda-graph-bs 1 2 4 8 16 32 48 64 \
  --watchdog-timeout 3600
```

The device selection and port are examples; choose free resources on the
target host. Keep the model and its template from the same checkpoint. The
model-scoped environment above is the tested FlagOS image configuration;
`SGLANG_FL_FLAGOS_BLACKLIST` replaces the Ascend YAML default, so keep the
complete list when reproducing this run. `SGLANG_FL_PREFER=vendor` sets the
general backend preference, while `SGLANG_FL_PER_OP` selects FlagGems for
`silu_and_mul`. The causal-conv and receive-coalescing settings are tied to
this validated image stack. Recheck the effective routes, template selection,
and multimodal behavior after changing the image or SGLang version. Do not
apply this model-specific blacklist as a global Ascend default.

The API regression client requires Python with Pillow installed (included in
the tested FlagOS image). In a separate terminal, run:

```bash
python examples/minicpmv47_35b_ascend/verify.py \
  --base-url http://127.0.0.1:31162 \
  --model MiniCPM-V-4.7-35B-A3B
```

It checks text, one image, two images, 16 images, and four concurrent
single-image requests. The generated test images are solid colors. A passing
smoke run does not establish benchmark accuracy, long-run stability, or
throughput/latency acceptance.
