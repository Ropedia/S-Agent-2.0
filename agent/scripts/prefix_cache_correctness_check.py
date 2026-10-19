#!/usr/bin/env python3
"""Does prefix caching mix up images across concurrent requests?

The specific hazard
-------------------
This model is a Mamba/linear-attention hybrid, and vLLM issue #20261 reports
that with prefix caching on, *identical text prompts carrying different images*
produced garbled output under concurrency. That is exactly this project's load
shape: one shared system prompt, 200 tasks, different frames each, 8-32 in
flight. Issue #43587 additionally reports `num_cached_tokens` stuck at 0 for
incremental multimodal requests on this architecture.

So the acceptance question is NOT "is it faster". A cache that silently serves
one request's image to another would corrupt the dependent variable of the whole
study while merely looking like a surprising score -- the failure mode this
project has hit repeatedly.

How this catches it
-------------------
Each request carries a DIFFERENT generated image containing a large unique
number, and every request asks the SAME question ("what number do you see").
The correct answer is therefore known per request and differs per request, so a
cross-request image mix-up shows up as a wrong number rather than as something
that needs a human to judge. Requests are fired concurrently, because the bug is
reported only under concurrency.

temperature=0 so a mismatch cannot be explained away as sampling.

Usage:
    .venv/bin/python scripts/prefix_cache_correctness_check.py [--n 16] [--port 8077]

Exit code is non-zero if any answer does not match its own image.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def make_image(number: int) -> str:
    """A PNG showing `number` in large digits, base64-encoded.

    Deliberately high-contrast and large: the point is to test cache identity,
    not OCR difficulty. A first version upscaled a tiny default-font render with
    NEAREST, which produced a 1161 -> 1151 misread and briefly looked like cache
    contamination. Rendering at real size removes that confound.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (320, 200), "white")
    d = ImageDraw.Draw(img)
    text = str(number)
    font = None
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(path, 96)
            break
        except OSError:
            continue
    if font is None:  # no TTF on this box; fall back but keep it legible
        font = ImageFont.load_default()
    d.text((26, 50), text, fill="black", font=font)
    d.rectangle([4, 4, 315, 195], outline="black", width=5)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask(port: int, number: int, timeout: float) -> tuple[int, str]:
    body = {
        "model": "qwen3.6-35b-a3b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{make_image(number)}"
                        },
                    },
                    {
                        "type": "text",
                        # Identical for every request -- that is the point.
                        "text": (
                            "What number is shown in this image? "
                            "Reply with only the digits, nothing else."
                        ),
                    },
                ],
            }
        ],
        "max_tokens": 2000,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return number, f"<request failed: {type(exc).__name__}: {exc}>"
    msg = data["choices"][0]["message"]
    return number, (msg.get("content") or "").strip()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="concurrent requests")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args(argv[1:])

    numbers = [1000 + i * 7 for i in range(args.n)]
    print(f"firing {args.n} concurrent requests, identical text, distinct images")

    with ThreadPoolExecutor(max_workers=args.n) as pool:
        results = list(pool.map(lambda n: ask(args.port, n, args.timeout), numbers))

    ok = crossed = misread = unreadable = 0
    for expected, got in results:
        digits = "".join(c for c in got if c.isdigit())
        if got.startswith("<request failed"):
            unreadable += 1
            print(f"  {expected}: {got}")
        elif digits == str(expected):
            ok += 1
        elif not digits:
            unreadable += 1
            print(f"  {expected}: no digits in reply {got[:60]!r}")
        elif int(digits) in numbers:
            # THE signature of a cache mix-up: the reply is a number that
            # belongs to a DIFFERENT request in this same batch.
            crossed += 1
            print(f"  CROSSED  expected {expected}, got {digits} "
                  f"(another request's image in this batch)")
        else:
            # A number that belongs to nobody is the model misreading the
            # rendered digits, not the cache serving the wrong image. These are
            # different failures and only the first one is about caching --
            # scoring them together would condemn prefix caching for an OCR
            # slip, which is how a good feature gets discarded on bad evidence.
            misread += 1
            print(f"  misread  expected {expected}, got {digits} (not in batch)")

    print()
    print(f"correct {ok}/{args.n}   CROSSED {crossed}   misread {misread}   "
          f"unreadable {unreadable}")

    if crossed:
        print("VERDICT  UNSAFE -- images are being crossed between requests")
        return 1
    if ok == 0:
        print("VERDICT  INCONCLUSIVE -- the model read none of the images; "
              "this test cannot say anything about caching")
        return 2
    if misread + unreadable > args.n // 4:
        print(f"VERDICT  WEAK -- {misread + unreadable} replies unusable; the "
              "test is too noisy to clear caching, rerun with clearer images")
        return 2
    print(f"VERDICT  no cross-request contamination detected "
          f"({misread} benign misread(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
