# Copyright (c) 2026 BAAI. All rights reserved.

"""Model configuration loader for sglang-plugin-FL tests.

Model YAML files use SGLang-native engine parameter names such as
``tp_size``, ``context_length``, ``mem_fraction_static``, and
``disable_cuda_graph``.
"""

from __future__ import annotations

import os
import itertools
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
_ENGINE_OVERRIDES_ENV = "FL_TEST_ENGINE_OVERRIDES"

def _load_structured(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        data = json.loads(text)
    return data if isinstance(data, dict) else {}


def load_engine_overrides_from_env() -> dict[str, Any]:
    """Load platform engine overrides forwarded by ``tests/run.py``."""
    raw = os.environ.get(_ENGINE_OVERRIDES_ENV, "")
    if not raw:
        return {}
    try:
        overrides = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{_ENGINE_OVERRIDES_ENV} must contain valid JSON") from exc
    if not isinstance(overrides, dict):
        raise ValueError(f"{_ENGINE_OVERRIDES_ENV} must contain a JSON object")
    return overrides

def _translate_cuda_graph_args(params: dict[str, Any]) -> dict[str, Any]:
    """Translate 0.5.11-era cuda_graph kwargs to the installed sglang's ServerArgs schema.

    sglang v0.5.16 split ``disable_cuda_graph`` into ``disable_prefill_cuda_graph`` +
    ``disable_decode_cuda_graph`` and removed ``disable_piecewise_cuda_graph`` (folded
    into the ``cuda_graph_backend_*`` system). Test yamls still use the 0.5.11 names.
    The CLI accepts the old names as aliases, but ``ServerArgs(**kwargs)`` (the
    Engine/inference path) rejects them with TypeError. Introspect ``ServerArgs`` so
    this is a no-op on 0.5.11 (cuda/ascend) and only translates where the old names are
    gone (musa/v0.5.16).
    """
    try:
        from sglang.srt.server_args import ServerArgs
        import dataclasses as _dc

        if not _dc.is_dataclass(ServerArgs):
            return params
        valid = {f.name for f in _dc.fields(ServerArgs)}
    except Exception:
        return params
    if "disable_cuda_graph" in valid and "disable_piecewise_cuda_graph" in valid:
        return params  # 0.5.11 schema: old names valid, nothing to translate
    out: dict[str, Any] = {}
    for k, v in params.items():
        if k in valid:
            out[k] = v
        elif k == "disable_cuda_graph" and v and "disable_decode_cuda_graph" in valid:
            out.setdefault("disable_prefill_cuda_graph", True)
            out["disable_decode_cuda_graph"] = True
        elif k == "disable_piecewise_cuda_graph":
            continue  # removed in v0.5.16; prefill+decode disabled above covers it
        # else: drop unknown kwarg (ServerArgs would TypeError on it)
    return out


@dataclass
class GenerateConfig:
    modality: str = "text"
    prompts: list[Any] = field(default_factory=list)
    assets: list[str] = field(default_factory=list)
    sampling: dict[str, Any] = field(default_factory=dict)
    vl: dict[str, Any] = field(default_factory=dict)
    parametrize: dict[str, list[Any]] | list[dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GenerateConfig":
        return cls(
            modality=raw.get("modality", "text"),
            prompts=raw.get("prompts", []),
            assets=raw.get("assets", []),
            sampling=raw.get("sampling", {}),
            vl=raw.get("vl", {}),
            parametrize=raw.get("parametrize", {}),
        )

    def get_parametrize_combos(self) -> list[dict[str, Any]]:
        """Return explicit combos or the Cartesian product of dimensions."""
        if not self.parametrize:
            return [{}]
        if isinstance(self.parametrize, list):
            return [dict(combo) for combo in self.parametrize]

        keys = list(self.parametrize.keys())
        values = [self.parametrize[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


@dataclass
class ServeConfig:
    api_key: str = ""
    extra_engine: dict[str, Any] = field(default_factory=dict)
    served_model_name: str = ""
    startup_retries: int = 120
    endpoints: list[str] = field(default_factory=list)
    completion_prompt: str = "Hello"
    stream: bool = False
    max_tokens: int = 50
    chat_messages: list[dict[str, Any]] = field(default_factory=list)
    sampling: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    embedding_input: str = ""

    def request_model(self, model_path: str) -> str:
        return self.served_model_name or model_path

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ServeConfig":
        return cls(
            api_key=raw.get("api_key", ""),
            extra_engine=raw.get("extra_engine", {}),
            served_model_name=raw.get("served_model_name", ""),
            startup_retries=int(raw.get("startup_retries", 120)),
            endpoints=raw.get("endpoints", []),
            completion_prompt=raw.get("completion_prompt", "Hello"),
            stream=bool(raw.get("stream", False)),
            max_tokens=int(raw.get("max_tokens", 50)),
            chat_messages=raw.get("chat_messages", []),
            sampling=raw.get("sampling", {}),
            extra_body=raw.get("extra_body", {}),
            embedding_input=raw.get("embedding_input", ""),
        )

@dataclass
class ConcurrentConfig:
    modes: list[str] = field(default_factory=list)
    concurrent_n: int = 4
    text_prompts: list[dict[str, Any]] = field(default_factory=list)
    vl_cases: list[dict[str, Any]] = field(default_factory=list)
    text_sampling: dict[str, Any] = field(default_factory=dict)
    vl_sampling: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ConcurrentConfig":
        sampling = raw.get("sampling", {})
        return cls(
            modes=raw.get("modes", []),
            concurrent_n=int(raw.get("concurrent_n", 4)),
            text_prompts=raw.get("text", {}).get("prompts", []),
            vl_cases=raw.get("vl", {}).get("cases", []),
            text_sampling=sampling.get("text", {}),
            vl_sampling=sampling.get("vl", {}),
        )

@dataclass
class ModelConfig:
    model: str
    engine: dict[str, Any] = field(default_factory=dict)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    concurrent: ConcurrentConfig = field(default_factory=ConcurrentConfig)

    @classmethod
    def load(
        cls,
        model: str,
        case: str | None = None,
        models_dir: Path | None = None,
        engine_overrides: dict[str, Any] | None = None,
    ) -> "ModelConfig":
        models_dir = models_dir or _MODELS_DIR
        path = models_dir / model / f"{case}.yaml" if case else models_dir / f"{model}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Model config not found: {path}")
        config = cls.from_dict(_load_structured(path))
        if engine_overrides:
            config.engine.update(engine_overrides)
        return config

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        llm_raw = dict(raw.get("llm", {}))
        model = llm_raw.pop("model", raw.get("model_path", ""))
        engine = llm_raw or dict(raw.get("engine", {}))
        return cls(
            model=model,
            engine=engine,
            generate=GenerateConfig.from_dict(raw.get("generate", {})),
            serve=ServeConfig.from_dict(raw.get("serve", {})),
            concurrent=ConcurrentConfig.from_dict(raw.get("concurrent", {})),
        )

    def sglang_common_params(self) -> dict[str, Any]:
        """Return engine parameters using SGLang CLI names."""
        params = {"model_path": self.model, **self.engine}
        # MUSA-only: mate's flash_attn asserts ``page_table.stride(-1) == 1``, which
        # the default page_size (64) trips on batched decode. The contiguity patch is
        # intentionally not shipped, so force page_size=1 on MUSA only. Gated here
        # (not in the shared model YAML) because tests/benchmarks/configs/smoke.yaml
        # shares 06b_tp1 across platforms, so a YAML page_size would slow CUDA's
        # benchmark for no benefit. ``setdefault`` leaves explicit values (e.g. the
        # mamba-mandated page_size=1 on qwen3_6) untouched.
        if os.environ.get("FL_TEST_PLATFORM") == "musa":
            params.setdefault("page_size", 1)
        return params

    def engine_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Return engine parameters as Python kwargs for SGLang Engine."""
        params = self.sglang_common_params()
        params.update(overrides)
        return _translate_cuda_graph_args(params)

    def sampling_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Return sampling parameters using SGLang Engine names."""
        params = dict(self.generate.sampling)
        params.update(overrides)
        return params

    def benchmark_parameters(self, overrides: dict[str, Any]) -> dict[str, Any]:
        params = self.sglang_common_params()
        params.update(overrides)
        return _translate_cuda_graph_args(params)

    def server_parameters(self, overrides: dict[str, Any]) -> dict[str, Any]:
        params = self.sglang_common_params()
        params.update(overrides)
        if self.serve.served_model_name:
            params.setdefault("served_model_name", self.serve.served_model_name)
        return params

    def serve_args(self, **overrides: Any) -> list[str]:
        params = self.sglang_common_params()
        params.pop("model_path", None)
        params.pop("tp_size", None)
        params.update(overrides)

        args: list[str] = []
        for key, value in params.items():
            if value is None or value is False:
                continue
            flag = "--" + key.replace("_", "-")
            if value is True or value == "":
                args.append(flag)
            else:
                args.extend([flag, str(value)])
        return args




