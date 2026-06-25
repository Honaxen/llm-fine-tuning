"""
data/prepare.py — Download and prepare Alpaca instruction dataset.

Usage:
    python3 data/prepare.py --sample 200
"""

import argparse
import json
import os
import ssl
import urllib.request

ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca"
    "/main/alpaca_data.json"
)
SYSTEM_PROMPT = (
    "You are a helpful, respectful, and honest assistant. "
    "Always answer as helpfully as possible."
)


def _ssl_ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def format_chatml(instruction: str, input_text: str, output: str) -> dict:
    user_message = f"{instruction}\n\n{input_text}" if input_text.strip() else instruction
    return {
        "text": (
            f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
            f"<|user|>\n{user_message}</s>\n"
            f"<|assistant|>\n{output}</s>"
        )
    }


def download_alpaca(output_path: str) -> list[dict]:
    print("Downloading Alpaca dataset...")
    req = urllib.request.Request(
        ALPACA_URL,
        headers={"User-Agent": "llm-fine-tuning/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30, context=_ssl_ctx()) as resp:
        data = json.loads(resp.read().decode())
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → {len(data)} examples saved to {output_path}")
    return data


def prepare(raw: list[dict], output_path: str, max_samples: int = None) -> list[dict]:
    if max_samples:
        raw = raw[:max_samples]
    formatted = [
        format_chatml(item["instruction"], item.get("input", ""), item["output"])
        for item in raw
        if item.get("instruction") and item.get("output")
    ]
    with open(output_path, "w") as f:
        json.dump(formatted, f, indent=2)
    print(f"  → {len(formatted)} formatted examples saved to {output_path}")
    return formatted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, "alpaca_raw.json")
    raw = download_alpaca(raw_path)

    prepare(raw, os.path.join(args.out_dir, "alpaca_formatted.json"))
    prepare(raw, os.path.join(args.out_dir, "sample.json"), max_samples=200)

    print(f"\nReady:")
    print(f"  Full    : data/alpaca_formatted.json")
    print(f"  Sample  : data/sample.json")


if __name__ == "__main__":
    main()