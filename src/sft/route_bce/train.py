"""Explicitly armed four-80GB-GPU full-parameter Route-SFT entry.

The legacy sampler, prefix tokenization, AdamW schedule and sample-mean suffix
objective are retained. A separate full-vocabulary BCE is added. This minimal
entry uses native PyTorch SDPA/autocast rather than an RL/vLLM/Analyzer stack.
No background launch or automatic OPSD is provided.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .core import batches, encode_row, generation_metrics, load_rows, losses, register_token, sha, write_json
from . import distributed as parallel
from torch.nn.parallel import DistributedDataParallel


def safe_child(root, relative):
    value = (root / relative).resolve()
    if not value.is_relative_to(root.resolve()) or value == root.resolve():
        raise ValueError("Configured path escaped task root")
    return value


def check_config(c):
    fixed = {"route_lambda": .1, "suffix_ce_weight": 1., "reference_kl_coef": 0.,
             "global_batch_size": 8, "good_per_batch": 5, "bad_per_batch": 3,
             "micro_batch_size": 1,
             "max_length": 8192, "truncation": "error", "full_parameter_training": True,
             "lora_rank": 0, "loss_reduction": "sample_mean", "prepared_world_size": 1,
             "eval_every": 100, "checkpoint_every": 500, "generation_every": 500,
             "generation_assistant_prefix": "", "generation_skip_special_tokens": False,
             "automatic_early_stop": False, "trigger_rate_gate": False, "automatic_opsd": False,
             "lr": 2e-6, "betas": [.9, .95], "weight_decay": .01, "warmup_ratio": .1,
             "scheduler": "cosine", "clip_grad": 1., "seed": 0,
             "parameter_dtype": "float32", "autocast_dtype": "bfloat16"}
    for k, v in fixed.items():
        if c.get(k) != v:
            raise ValueError("Frozen training contract mismatch: " + k)
    if (c.get("epochs"), c.get("total_steps"), c.get("scheduler_total_steps")) != (1, 2000, 4000):
        raise ValueError("Stop at 2000 while retaining the original 4000-step LR horizon")
    if c.get("full_resume_checkpoint_steps") != [2000]:
        raise ValueError("The final step2000 checkpoint must include complete resume state")


def training_order(rows, c):
    order = [batch for epoch in range(c["epochs"])
             for batch in batches(rows, c["seed"], epoch=epoch)]
    if len(order) != c["total_steps"]:
        raise ValueError("Data size does not match complete-epoch schedule")
    return order


def check_resources(c, root, resume=False):
    if not torch.cuda.is_available() or torch.cuda.device_count() != c["prepared_world_size"]:
        raise RuntimeError("Assign one explicitly allocated GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support required")
    free, total = torch.cuda.mem_get_info()
    if total < c["minimum_gpu_memory_gib"] * 2**30 or free < 65 * 2**30:
        raise RuntimeError("Insufficient free VRAM; do not silently change recipe")
    limits = []
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        p = Path(name)
        if p.exists():
            raw = p.read_text().strip()
            if raw.isdigit(): limits.append(int(raw))
    meminfo = Path("/proc/meminfo").read_text()
    limits.append(int(re.search(r"MemTotal:\s+(\d+)", meminfo)[1]) * 1024)
    if min(limits) < c["minimum_host_memory_gib"] * 2**30:
        raise RuntimeError("At least 48GiB container RAM required")
    minimum_disk_gib = 45 if resume else 100
    if shutil.disk_usage(root).free < minimum_disk_gib * 2**30:
        raise RuntimeError("Insufficient disk headroom for atomic checkpoint and resume-state saving")


def one_forward(model, encoded, marker_id, c):
    ids = torch.tensor(encoded["input_ids"], device="cuda", dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids[None, :], attention_mask=torch.ones_like(ids)[None, :],
                       use_cache=False).logits[0]
        out = losses(logits, ids, encoded["suffix_start"], encoded["route_label"], marker_id, c["route_lambda"])
    return out


def train_batch(model, encoded, indices, marker_id, c, forward=one_forward):
    """Same eight sample-mean losses, split 2/rank; DDP averages the gradients."""
    positions = parallel.local_indices(list(range(len(indices))))
    records = []
    scale = parallel.backward_scale(c["global_batch_size"])
    if len(indices) != c["global_batch_size"]:
        raise ValueError("Unexpected global batch size")
    for j, position in enumerate(positions):
        item = encoded[indices[position]]
        with parallel.accumulation_context(model, j == len(positions) - 1):
            out = forward(model, item, marker_id, c)
            if not torch.isfinite(out["loss"]):
                raise FloatingPointError("Non-finite loss")
            (out["loss"] * scale).backward()
        records.append((position, {"route_label": item["route_label"],
                        **{k: float(v.detach()) for k, v in out.items()}}))
    return parallel.ordered_records(records)


def make_scheduler(optimizer, c):
    # Stopping the run at 2000 is NOT rescheduling the cosine to end at 2000.
    horizon = c["scheduler_total_steps"]
    return get_cosine_schedule_with_warmup(optimizer, int(c["warmup_ratio"] * horizon), horizon)


def progress(output, phase, step, total):
    if parallel.primary():
        write_json(output / "progress.json",
                   {"phase": phase, "step": step, "total_steps": total, "time": time.time()})


def evaluate(model, encoded, marker_id, c, step, output):
    model = parallel.unwrap(model)
    was_training = model.training
    model.eval()
    records = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
        for position in range(parallel.rank(), len(encoded), parallel.world_size()):
            item = encoded[position]
            out = one_forward(model, item, marker_id, c)
            records.append((position, {"audit_id": item["audit_id"], "label": item["route_label"],
                            **{k: float(v) for k, v in out.items()}}))
    records = parallel.ordered_records(records)
    assert len(records) == len(encoded)
    model.train(was_training)
    report = {"step": step, "rows": len(records), "teacher_forced": True}
    for group, selected in (("all", records), ("good", [x for x in records if x["label"] == 0]),
                             ("bad", [x for x in records if x["label"] == 1])):
        report[group] = {"n": len(selected), **{k: sum(x[k] for x in selected) / len(selected)
            for k in ("loss", "suffix_ce", "route_bce", "marker_probability", "binary_logit")}}
    if parallel.primary():
        write_json(output / f"eval/step_{step:04d}.json", {"summary": report, "rows": records})
    return report


def generate(model, tokenizer, rows, marker_id, c, step, output):
    model = parallel.unwrap(model)
    was_training = model.training
    checkpointing = model.is_gradient_checkpointing
    if checkpointing: model.gradient_checkpointing_disable()
    model.eval()
    records = []
    eos = model.generation_config.eos_token_id
    eos_ids = list(eos) if isinstance(eos, (tuple, list)) else [eos]
    if tokenizer.eos_token_id not in eos_ids: eos_ids.append(tokenizer.eos_token_id)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
        for position in range(parallel.rank(), len(rows), parallel.world_size()):
            row = rows[position]
            # Only H, NEVER a saved R or a forced <think> prefix.
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                                   tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.generate(**inputs, do_sample=False, num_beams=1,
                    max_new_tokens=c["generation_max_new_tokens"], eos_token_id=eos_ids,
                    pad_token_id=tokenizer.pad_token_id, use_cache=True)
            ids = result[0, inputs["input_ids"].shape[1]:].tolist()
            text = tokenizer.decode(ids, skip_special_tokens=False)
            metrics = generation_metrics(text, ids, marker_id, eos_ids)
            metrics["length_truncated"] = len(ids) == c["generation_max_new_tokens"] and not metrics["terminated_with_eos"]
            metrics["stored_current_executable"] = metrics["parsed_action"] in row["admissible_actions_before"]
            metrics["oracle_action_equal"] = metrics["parsed_action"] == row["oracle_action"]
            records.append((position, {"audit_id": row["audit_id"], "source_route": row["route"], "text": text,
                            "token_ids": ids, **metrics}))
    records = parallel.ordered_records(records)
    assert len(records) == len(rows)
    if checkpointing: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train(was_training)
    keys = ("reasoning_open", "reasoning_closed", "strict_protocol", "marker_present", "length_truncated",
            "stored_current_executable", "oracle_action_equal")
    summary = {"step": step, "rows": len(rows), "empty_assistant_prefix": True,
               "temperature": 0, "max_new_tokens": c["generation_max_new_tokens"],
               "skip_special_tokens": False, "forced_route_or_think": False,
               "new_generated_reasoning_has_NOT_been_semantically_relabelled": True,
               "fresh_environment_execution": False}
    for group, selected in (("all", records), ("source_good", [r for r in records if r["source_route"] == "good"]),
                             ("source_bad", [r for r in records if r["source_route"] == "bad"])):
        summary[group] = {"n": len(selected), **{k: sum(r[k] for r in selected) / len(selected) for k in keys}}
    if parallel.primary():
        write_json(output / f"generation/step_{step:04d}.json", {"summary": summary, "rows": records})
    return summary


def wait_checkpoint_space(model, output):
    # A model plus two Adam moments are written while the preceding state is
    # still valid. Never delete an unverified backup to make room.
    required = sum(p.numel() * p.element_size() for p in model.parameters()) * 3 + 2 * 2**30
    if shutil.disk_usage(output).free < required:
        write_json(output / "progress.json", {"phase": "waiting_for_verified_backup_disk_space",
                   "required_free_bytes": required, "time": time.time()})
        print(json.dumps({"waiting_for_backup_disk_space": True, "required_bytes": required}), flush=True)
        raise RuntimeError("Insufficient atomic-checkpoint space; prior complete checkpoint preserved")


def checkpoint(model, tokenizer, optimizer, scheduler, step, output, binding, report, best,
               *, epoch_end=False, data_position=None, rank_rng_states=None):
    target = output / f"global_step_{step}"
    temporary = output / f"global_step_{step}.incomplete"
    if target.exists() or temporary.exists():
        raise FileExistsError("Refuse to overwrite checkpoint: " + str(target))
    temporary.mkdir()
    model.save_pretrained(temporary, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(temporary)
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in temporary.iterdir() if p.is_file()}
    old_pointer = (json.loads((output / "latest_resume.json").read_text())
                   if (output / "latest_resume.json").exists() else None)
    state = {"step": step, "binding": binding, "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if parallel.world_size() == 1 else [],
             "python_rng": random.getstate(),
             "data_position": data_position or {"next_global_batch": step}}
    if rank_rng_states is not None:
        state["rank_rng_states"] = rank_rng_states
        state["world_size"] = len(rank_rng_states)
    # At epoch boundaries the full state is IMMUTABLE inside the checkpoint,
    # so an asynchronous backup cannot race with a rolling-state replacement.
    state_final = temporary / "training_state.pt" if epoch_end else output / f"resume_state_step_{step}.pt"
    state_next = state_final.with_name(state_final.name + ".next")
    if state_final.exists() or state_next.exists():
        raise FileExistsError("Refuse to overwrite a training state")
    torch.save(state, state_next)
    state_sha = sha(state_next)
    state_next.rename(state_final)
    state_info = {"file": "training_state.pt", "bytes": state_final.stat().st_size,
                  "sha256": state_sha} if epoch_end else None
    write_json(temporary / "sft_stage_manifest.json", {"stage": "semantic_hr_route_bce_sft", "step": step,
        "binding": binding, "model_files": files, "validation": report, "checkpoint_complete": True,
        "weights_only_directory": not epoch_end, "epoch_end": epoch_end,
        "training_state": state_info, "data_position": state["data_position"],
        "latest_optimizer_state_is_separate": not epoch_end})
    temporary.rename(target)
    state_relative = f"{target.name}/training_state.pt" if epoch_end else state_final.name
    write_json(output / "latest_resume.json", {"step": step, "checkpoint": target.name,
        "state": state_relative, "state_sha256": state_sha, "binding": binding})
    # Delete only our prior rolling STATE file, after its successor is fully
    # published. Epoch checkpoints are removed only by the verified-backup tool.
    if old_pointer and re.fullmatch(r"resume_state_step_\d+\.pt", old_pointer["state"]):
        old_state = safe_child(output, old_pointer["state"])
        if old_state.is_file() and old_state != state_final:
            old_state.unlink()
    if best is None or report["all"]["loss"] < best["loss"]:
        best = {"step": step, "checkpoint": target.name, "loss": report["all"]["loss"],
                "selection": "lowest composite eval loss among saved 500-step checkpoints; not environment success"}
        write_json(output / "best_saved_checkpoint.json", best)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume-latest", action="store_true")
    p.add_argument("--resume-checkpoint", help="Full epoch checkpoint, e.g. global_step_2000")
    p.add_argument("--gpu-preflight-only", action="store_true")
    a = p.parse_args()
    if not json.loads((a.root/'configs/ablation.json').read_text())['sft_enabled']:
        raise SystemExit('The external API ablation starts from the base; Route-SFT is disabled')
    if os.environ.get("ALLOW_ROUTE_SEMANTIC_BCE_TRAINING") != "1":
        raise SystemExit("Preparation only: explicit GPU launch authorization not armed")
    c = json.loads(a.config.read_text(encoding="utf-8"))
    check_config(c)
    local_rank = parallel.initialize(c["prepared_world_size"])
    root = a.root.resolve()
    parent, data, output = [safe_child(root, c[k]) for k in ("parent_dir", "data_dir", "output_dir")]
    resuming = a.resume_latest or bool(a.resume_checkpoint)
    if a.resume_latest and a.resume_checkpoint:
        raise ValueError("Choose one resume source")
    check_resources(c, root, resuming)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        assert sha(data / name) == info["sha256"], name
    preflight = json.loads((root / "reports/cpu_token_preflight_remote.json").read_text())
    assert preflight["passed"] and preflight["dataset_manifest_sha256"] == sha(data / "manifest.json")
    receipt = json.loads((parent / "parent_download_receipt.json").read_text())
    assert receipt["revision"] == c["base_revision"] and receipt["model_id"] == c["base_model"]
    for info in receipt["files"]:
        assert sha(parent / info["name"]) == info["sha256"], info["name"]
    binding = {"config_sha256": sha(a.config), "dataset_manifest_sha256": sha(data / "manifest.json"),
               "parent_receipt_sha256": sha(parent / "parent_download_receipt.json"),
               "code": {str(x.relative_to(Path(__file__).parent.parent)): sha(x)
                        for folder in (Path(__file__).parent, Path(__file__).parent.parent / "legacy")
                        for x in sorted(folder.glob("*.py"))},
               "config": c, "no_analyzer_teacher_api": True,
               "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "safetensors")}}
    if not resuming and not a.gpu_preflight_only and parallel.primary():
        output.mkdir(parents=True, exist_ok=False)
    parallel.barrier()
    random.seed(c["seed"])
    torch.manual_seed(c["seed"])
    torch.cuda.manual_seed_all(c["seed"])
    tokenizer = AutoTokenizer.from_pretrained(parent, local_files_only=True, trust_remote_code=False)
    token_state = register_token(tokenizer)
    marker_id = token_state["token_id"]
    train_rows, val_rows = load_rows(data / "train.jsonl"), load_rows(data / "val.jsonl")
    order = training_order(train_rows, c)
    binding["sampler_order_sha256"] = hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest()
    encoded_train = [encode_row(r, tokenizer, c["max_length"]) for r in train_rows]
    encoded_val = [encode_row(r, tokenizer, c["max_length"]) for r in val_rows]
    generation_rows = load_rows(data / "generation_val32.jsonl")
    resume, start, best = None, 0, None
    model_source = parent
    if resuming:
        if a.resume_checkpoint:
            source = safe_child(output, a.resume_checkpoint)
            full = json.loads((source / "sft_stage_manifest.json").read_text())
            assert full["epoch_end"] and full["training_state"]
            pointer = {"binding": full["binding"], "checkpoint": source.name,
                       "state": f"{source.name}/training_state.pt", "state_sha256": full["training_state"]["sha256"]}
        else:
            pointer = json.loads((output / "latest_resume.json").read_text())
        assert pointer["binding"] == binding
        assert sha(output / pointer["state"]) == pointer["state_sha256"]
        model_source = safe_child(output, pointer["checkpoint"])
        cp = json.loads((model_source / "sft_stage_manifest.json").read_text())
        assert cp["checkpoint_complete"] and cp["binding"] == binding
        for name, info in cp["model_files"].items(): assert sha(model_source / name) == info["sha256"]
        # Own locally generated optimizer artifact, verified above; no untrusted pickle.
        resume = torch.load(output / pointer["state"], map_location="cpu", weights_only=False)
        assert resume["binding"] == binding
        start = resume["step"]
        if (output / "best_saved_checkpoint.json").exists():
            best = json.loads((output / "best_saved_checkpoint.json").read_text())
    model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=torch.float32,
        attn_implementation=c["attention_backend"], low_cpu_mem_usage=True,
        local_files_only=True, trust_remote_code=False).to("cuda")
    token_state = register_token(tokenizer, model)
    assert all(p.requires_grad for p in model.parameters())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], betas=tuple(c["betas"]),
                                  weight_decay=c["weight_decay"])
    scheduler = make_scheduler(optimizer, c)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                    broadcast_buffers=False, gradient_as_bucket_view=True)
    if a.gpu_preflight_only:
        model.train()
        torch.cuda.reset_peak_memory_stats()
        selected = []
        for label, count in ((0, 5), (1, 3)):
            selected.extend(sorted((x for x in encoded_train if x["route_label"] == label),
                                   key=lambda x: len(x["input_ids"]), reverse=True)[:count])
        begin = time.monotonic()
        terms = train_batch(model, selected, list(range(8)), marker_id, c)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c["clip_grad"], error_if_nonfinite=True)
        optimizer.step()  # Allocates actual Adam state; this disposable model is NOT saved.
        torch.cuda.synchronize()
        devices = parallel.gather({"rank": parallel.rank(), "gpu": torch.cuda.get_device_name(local_rank),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})
        if parallel.primary():
            write_json(root / "reports/gpu_preflight_4gpu.json", {"passed": True, "binding": binding,
                "formal_training_started": False, "disposable_model_not_saved": True, "rows": terms,
                "grad_norm": float(norm), "seconds": time.monotonic() - begin, "devices": devices})
            print("GPU_PREFLIGHT_PASSED_DISPOSABLE_MODEL_NOT_SAVED", flush=True)
        return
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        if resume.get("world_size") != parallel.world_size():
            raise ValueError("Resume world size differs; do not silently change rank RNG mapping")
        parallel.restore_rng(resume["rank_rng_states"][parallel.rank()])
        del resume
    if parallel.primary():
        write_json(output / "run_binding.json", {"binding": binding, "marker": token_state,
                   "resume_start_step": start, "training_started": True,
                   "world_size": parallel.world_size(), "local_examples_per_update": 2})
    if start == 0:
        progress(output, "initial_eval", 0, len(order))
        evaluate(model, encoded_val, marker_id, c, 0, output)
        progress(output, "initial_generation", 0, len(order))
        generate(model, tokenizer, generation_rows, marker_id, c, 0, output)
    log = output / "metrics.jsonl"
    if resuming:
        # Preserve abandoned post-checkpoint metrics rather than mixing replayed
        # updates into one apparent run. The binding identifies the same recipe.
        log = output / f"metrics_resume_from_{start}_{time.time_ns()}.jsonl"
    with (log.open("x", encoding="utf-8", buffering=1) if parallel.primary() else nullcontext()) as stream:
        for index in range(start, len(order)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            begin = time.monotonic()
            lr_used = optimizer.param_groups[0]["lr"]
            terms = train_batch(model, encoded_train, order[index], marker_id, c)
            assert Counter(x["route_label"] for x in terms) == {0: 5, 1: 3}
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c["clip_grad"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step = index + 1
            metric = {"step": step, "total_steps": len(order), "lr_used": lr_used,
                      "epoch": index // 2000 + 1, "step_in_epoch": index % 2000 + 1,
                      "lr_next": scheduler.get_last_lr()[0], "grad_norm": float(norm),
                      "seconds": time.monotonic() - begin, "route_counts": dict(Counter(x["route_label"] for x in terms)),
                      "consumed_ids": [train_rows[i]["audit_id"] for i in order[index]]}
            for group, chosen in (("all", terms), ("good", [r for r in terms if r["route_label"] == 0]),
                                   ("bad", [r for r in terms if r["route_label"] == 1])):
                metric[group] = {k: sum(r[k] for r in chosen) / len(chosen) for k in ("loss", "suffix_ce", "route_bce", "marker_probability")}
            if parallel.primary():
                stream.write(json.dumps(metric) + "\n")
                print(json.dumps({k: v for k, v in metric.items() if k != "consumed_ids"}), flush=True)
            progress(output, "training", step, len(order))
            if step % c["eval_every"] == 0:
                progress(output, "eval", step, len(order))
                report = evaluate(model, encoded_val, marker_id, c, step, output)
                if parallel.primary():
                    print(json.dumps({"eval": report}), flush=True)
            if step % c["generation_every"] == 0:
                progress(output, "generation", step, len(order))
                generated = generate(model, tokenizer, generation_rows, marker_id, c, step, output)
                if parallel.primary():
                    print(json.dumps({"generation": generated}), flush=True)
            if step % c["checkpoint_every"] == 0:
                states = parallel.gather(parallel.rng_state())
                if parallel.primary():
                    wait_checkpoint_space(model, output)
                    progress(output, "checkpoint", step, len(order))
                    best = checkpoint(parallel.unwrap(model), tokenizer, optimizer, scheduler, step,
                        output, binding, report, best, rank_rng_states=states,
                        epoch_end=step % 2000 == 0, data_position={"completed_epochs": step // 2000,
                        "next_global_batch": step, "next_epoch_index": step // 2000, "next_batch_in_epoch": step % 2000,
                        "sampler_order_sha256": binding["sampler_order_sha256"]})
                parallel.barrier()
                gc.collect()
                torch.cuda.empty_cache()
    if parallel.primary():
        write_json(output / "completion.json", {"completed": True, "steps": len(order), "binding": binding,
            "best_saved_checkpoint": best, "training_process_must_still_exit_before_GPU_release_claim": True})
    progress(output, "completed", len(order), len(order))
    parallel.barrier()


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
