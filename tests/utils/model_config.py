# Copyright (c) 2026 BAAI. All rights reserved.

"""Model configuration loader for sglang-plugin-FL tests.

Model YAML files use SGLang-native engine parameter names such as
``tp_size``, ``context_length``, ``mem_fraction_static``, and
``disable_cuda_graph``.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _load_structured(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        data = json.loads(text)
    return data if isinstance(data, dict) else {}


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
    ) -> "ModelConfig":
        models_dir = models_dir or _MODELS_DIR
        path = models_dir / model / f"{case}.yaml" if case else models_dir / f"{model}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Model config not found: {path}")
        return cls.from_dict(_load_structured(path))

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
        return {"model_path": self.model, **self.engine}

    def engine_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Return engine parameters as Python kwargs for SGLang Engine."""
        params = self.sglang_common_params()
        params.update(overrides)
        return params

    def sampling_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Return sampling parameters using SGLang Engine names."""
        params = dict(self.generate.sampling)
        params.update(overrides)
        return params

    def benchmark_parameters(self, overrides: dict[str, Any]) -> dict[str, Any]:
        params = self.sglang_common_params()
        params.update(overrides)
        return params

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




