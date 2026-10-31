"""One explicitly allocated GPU for one ablation; no discovery or reservation."""
import argparse
import os
import re


def devices(text, count):
    values = [v.strip() for v in text.split(",")]
    if (len(values) != count or len(set(values)) != count
            or any(not re.fullmatch(r"(?:0|[1-9]\d*|GPU-[A-Za-z0-9-]+)", v) for v in values)):
        raise ValueError(f"Expected {count} distinct whole GPU IDs (not MIG/duplicates)")
    return values


def layout(env=None):
    env = os.environ if env is None else env
    return {"method": devices(env.get("TRAIN_GPUS", env.get("CUDA_VISIBLE_DEVICES", "0")), 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["sft", "opsd"], required=True)
    a = p.parse_args()
    groups = layout()
    print(",".join(groups["method"]))


if __name__ == "__main__":
    main()
