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

"""Check MiniCPM-V 4.7 35B-A3B text and multi-image serving via the OpenAI API."""

import argparse
import base64
import io
import json
import logging
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

logger = logging.getLogger(__name__)


def image_url(color: str) -> str:
    image = Image.new("RGB", (128, 128), color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def request(
    base_url: str, model: str, prompt: str, colors: list[str], max_tokens: int
) -> str:
    content = [{"type": "text", "text": prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_url(color)}}
        for color in colors
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        data = json.load(response)
    choice = data["choices"][0]
    if choice["finish_reason"] != "stop":
        raise AssertionError(f"Unexpected finish reason: {choice['finish_reason']}")
    answer = choice["message"]["content"] or ""
    if not answer.strip():
        raise AssertionError("Empty model answer")
    return answer


def check_color(answer: str, color: str) -> None:
    if not re.search(rf"\b{re.escape(color)}\b", answer, re.IGNORECASE):
        raise AssertionError(f"Expected {color!r} in answer: {answer!r}")


def check_colors_in_order(answer: str, first: str, last: str) -> None:
    pattern = rf"\b{re.escape(first)}\b.*\b{re.escape(last)}\b"
    if not re.search(pattern, answer, re.IGNORECASE | re.DOTALL):
        raise AssertionError(f"Expected {first!r} then {last!r}: {answer!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31162")
    parser.add_argument("--model", default="MiniCPM-V-4.7-35B-A3B")
    args = parser.parse_args()

    answer = request(
        args.base_url,
        args.model,
        "What is 17 + 12? Reply with the number only.",
        [],
        64,
    )
    if "29" not in answer:
        raise AssertionError(f"Unexpected arithmetic answer: {answer!r}")

    answer = request(
        args.base_url,
        args.model,
        "What color is this image? Answer in English.",
        ["red"],
        128,
    )
    check_color(answer, "red")

    answer = request(
        args.base_url,
        args.model,
        "Name the color of image 1 and image 2 in order. Answer in English.",
        ["red", "green"],
        256,
    )
    check_colors_in_order(answer, "red", "green")

    colors = ["red", "green"] * 8
    answer = request(
        args.base_url,
        args.model,
        "What colors are image 1 and image 16, in that order? Answer in English.",
        colors,
        1024,
    )
    check_colors_in_order(answer, "red", "green")

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                request,
                args.base_url,
                args.model,
                "What color is this image? Answer in English.",
                [color],
                128,
            )
            for color in ("red", "green", "blue", "yellow")
        ]
        for color, future in zip(("red", "green", "blue", "yellow"), futures):
            check_color(future.result(), color)

    logger.info("MiniCPM-V 4.7 35B-A3B text and image smoke checks passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
