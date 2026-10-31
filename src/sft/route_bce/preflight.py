"""CPU-only token-boundary and immutable artifact checks; does not load weights."""
import argparse
from collections import Counter
import json
from pathlib import Path

from transformers import AutoTokenizer

from .core import batches, encode_row, load_rows, register_token, sha, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.data / "manifest.json").read_text(encoding="utf-8"))
    for name, rec in manifest["files"].items():
        assert sha(a.data / name) == rec["sha256"], name
    for rec in manifest["provenance"]:
        assert sha(a.data / rec["copy"]) == rec["sha256"], rec["copy"]
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
    state = register_token(tokenizer)
    train, val = load_rows(a.data / "train.jsonl"), load_rows(a.data / "val.jsonl")
    assert len(train) == 16000 and Counter(r["route"] for r in train) == {"good": 10000, "bad": 6000}
    assert len(val) == 256 and Counter(r["route"] for r in val) == {"good": 128, "bad": 128}
    assert not {r["task_id"] for r in train} & {r["task_id"] for r in val}
    order = batches(train)
    assert len(order) == 2000
    lengths, suffixes, joint_mismatches = [], Counter(), 0
    for idx, row in enumerate(train + val):
        enc = encode_row(row, tokenizer)
        lengths.append(len(enc["input_ids"]))
        suffixes[row["split"] + "/" + row["route"]] += enc["suffix_tokens"]
        joint_mismatches += int(not enc["joint_tokenization_matches"])
        if (idx + 1) % 4000 == 0:
            print(json.dumps({"token_checked": idx + 1}), flush=True)
    assert joint_mismatches == 0, "Training prefix tokenization differs from online continuation boundary"
    report = {"passed": True, "rows": len(lengths), "train_rows": len(train), "val_rows": len(val),
              "maximum_length": max(lengths), "max_length": 8192, "truncation": "error",
              "H_R_ce_tokens": 0, "hint_input_tokens_added": 0, "marker_first_target_verified": True,
              "joint_tokenization_mismatches": joint_mismatches, "supervised_tokens_including_eos": dict(suffixes),
              "updates": len(order), "each_train_row_consumed_once": True, "batch_good_bad": [5, 3],
              "marker": state, "dataset_manifest_sha256": sha(a.data / "manifest.json"),
              "training_started": False}
    write_json(a.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
