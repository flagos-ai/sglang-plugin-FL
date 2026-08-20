# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Precision alignment test: baseline SGLang vs sglang-plugin-FL.

Runs the same prompts with greedy decoding (temperature=0) under:
1. Baseline (no plugin)
2. Plugin (sglang_fl)

Compares outputs token-by-token to verify exact match.

Usage:
  # Default (0.5B, tp=1)
  python test_precision_align.py baseline|plugin|compare

  # Custom model / tp size
  MODEL_PATH=/path/to/model TP_SIZE=8 python test_precision_align.py baseline|plugin|compare
"""

import os
import sys
import json

MODEL_PATH = os.environ.get("MODEL_PATH", "")
if not MODEL_PATH:
    print("ERROR: MODEL_PATH environment variable is required.")
    print(
        "  Example: MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct python test_precision_align.py baseline"
    )
    sys.exit(1)
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))

PROMPTS = [
    "What is 2+3? Answer with just the number.",
    "Translate to French: Hello, how are you?",
    "Write a Python function that returns the factorial of n.",
    "What is the capital of Japan?",
    "Explain what a neural network is in one sentence.",
    "List the first 5 prime numbers.",
    "What is the square root of 144?",
    "Complete the sentence: The quick brown fox",
]

SAMPLING_PARAMS = {"max_new_tokens": 64, "temperature": 0}

ENGINE_KWARGS = dict(
    model_path=MODEL_PATH,
    tp_size=TP_SIZE,
    mem_fraction_static=0.5,
    disable_cuda_graph=True,
    disable_piecewise_cuda_graph=True,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Result file includes tp_size to avoid collision between runs
_suffix = f"_tp{TP_SIZE}" if TP_SIZE > 1 else ""
RESULT_FILE = os.path.join(_SCRIPT_DIR, f"precision_results{_suffix}.json")


def run_inference(mode: str):
    """Run inference and return list of output texts."""
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

    print(f"  Model: {MODEL_PATH}")
    print(f"  TP size: {TP_SIZE}")

    engine = Engine(**ENGINE_KWARGS)

    registry = MultiPlatformOp._oot_forward_registry
    print(f"  OOT forward registry keys: {list(registry.keys())}")

    results = []
    for i, prompt in enumerate(PROMPTS):
        result = engine.generate(prompt=prompt, sampling_params=SAMPLING_PARAMS)
        text = result["text"]
        results.append(text)
        print(f"  [{i + 1}/{len(PROMPTS)}] {prompt[:40]}... -> {repr(text[:60])}")
    engine.shutdown()
    return results


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else None

    if mode == "baseline":
        print(f"=== BASELINE (no plugin, tp={TP_SIZE}) ===")
        results = run_inference("baseline")
        data = {}
        if os.path.exists(RESULT_FILE):
            with open(RESULT_FILE) as f:
                data = json.load(f)
        data["baseline"] = results
        with open(RESULT_FILE, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\nBaseline results saved to {RESULT_FILE}")

    elif mode == "plugin":
        print(f"=== PLUGIN (sglang_fl, tp={TP_SIZE}) ===")
        results = run_inference("plugin")
        data = {}
        if os.path.exists(RESULT_FILE):
            with open(RESULT_FILE) as f:
                data = json.load(f)
        data["plugin"] = results
        with open(RESULT_FILE, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\nPlugin results saved to {RESULT_FILE}")

    elif mode == "compare":
        if not os.path.exists(RESULT_FILE):
            print(f"ERROR: {RESULT_FILE} not found. Run baseline and plugin first.")
            sys.exit(1)
        with open(RESULT_FILE) as f:
            data = json.load(f)
        if "baseline" not in data or "plugin" not in data:
            print("ERROR: Need both baseline and plugin results.")
            sys.exit(1)

        baseline = data["baseline"]
        plugin = data["plugin"]
        assert len(baseline) == len(plugin) == len(PROMPTS)

        all_match = True
        for i, (b, p) in enumerate(zip(baseline, plugin)):
            match = b == p
            status = "MATCH" if match else "MISMATCH"
            print(f"[{status}] Prompt {i + 1}: {PROMPTS[i][:50]}")
            if not match:
                all_match = False
                print(f"  Baseline: {repr(b[:100])}")
                print(f"  Plugin:   {repr(p[:100])}")
                for j, (bc, pc) in enumerate(zip(b, p)):
                    if bc != pc:
                        print(
                            f"  First diff at char {j}: baseline={repr(bc)} plugin={repr(pc)}"
                        )
                        break

        if all_match:
            print(f"\n[ALL MATCH] All {len(PROMPTS)} outputs are identical.")
        else:
            print("\n[MISMATCH FOUND] Some outputs differ.")
            sys.exit(1)

    else:
        print("Usage:")
        print("  python test_precision_align.py baseline|plugin|compare")
        print("")
        print(
            "  MODEL_PATH=/path/to/model TP_SIZE=8 python test_precision_align.py baseline"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
