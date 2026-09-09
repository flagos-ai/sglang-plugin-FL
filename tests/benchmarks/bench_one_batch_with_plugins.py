"""Run SGLang bench_one_batch after loading installed SGLang plugins."""

import runpy

from sglang.srt.plugins import load_plugins


def main() -> None:
    load_plugins()
    runpy.run_module("sglang.bench_one_batch", run_name="__main__")


if __name__ == "__main__":
    main()