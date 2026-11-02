"""One ablation pipeline; never auto-launch an additional baseline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gpu_layout import layout
from ablation import contract
from runtime_paths import resolve_python


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    pending = path.with_suffix(".next")
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def sft_complete():
    path = ROOT / "checkpoints/route_sft/completion.json"
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    if not (result["completed"] and result["steps"] == 2000
            and result["binding"]["config_sha256"] == sha(ROOT / "configs/sft.json")):
        raise RuntimeError("SFT completion belongs to a different recipe")
    cp = path.parent / "global_step_2000"
    manifest = json.loads((cp / "sft_stage_manifest.json").read_text())
    if not manifest["epoch_end"] or not manifest["checkpoint_complete"]:
        raise RuntimeError("SFT2000 is not a complete epoch checkpoint")
    for name, info in manifest["model_files"].items():
        if sha(cp / name) != info["sha256"]:
            raise RuntimeError("SFT model checksum mismatch")
    info = manifest["training_state"]
    if sha(cp / info["file"]) != info["sha256"]:
        raise RuntimeError("SFT resume-state checksum mismatch")
    return True


def commands(arm, resume=False):
    if arm!='opsd':raise ValueError('This repository runs only its declared ablation, not an extra baseline')
    py = resolve_python(os.environ.get("OPSD_PYTHON", ".venv-opsd/bin/python"), ROOT)
    result = []
    if contract()["sft_enabled"] and arm == "opsd" and not sft_complete():
        existing = ROOT / "checkpoints/route_sft"
        options = []
        if existing.exists():
            if not resume or not (existing / "latest_resume.json").is_file():
                raise RuntimeError("Existing SFT needs a verified resume pointer; inspect before retrying")
            options = ["--resume-latest"]
        else:
            result.append(("sft_preflight", ["bash", "scripts/train.sh", "sft", "--gpu-preflight-only"]))
        result.append(("sft", ["bash", "scripts/train.sh", "sft", *options]))
    result.append(("training", ["bash", "scripts/train.sh", arm,
                               *(["--allow-api-cost"] if arm == "opsd" else [])]))
    result.append(("export", [py, ".deps/sdar/scripts/model_merger.py", "merge", "--backend", "fsdp",
        "--local_dir", str(ROOT / f"checkpoints/{arm}/checkpoints/global_step_150/actor"),
        "--target_dir", str(ROOT / f"checkpoints/{arm}_hf150")]))
    result.append(("evaluation", ["bash", "scripts/eval.sh",
        "--model", str(ROOT / f"checkpoints/{arm}_hf150"),
        "--out", str(ROOT / f"runs/{arm}_seen140"), "--step", "150"]))
    return result


def verify_stage(arm, stage):
    if stage == "sft":
        if not sft_complete():
            raise RuntimeError("SFT exited without verified step2000 completion")
    elif stage == "training":
        folder = ROOT / f"checkpoints/{arm}/checkpoints/global_step_150/actor"
        for rank in range(1):
            for kind in ("model", "optim", "extra_state"):
                path = folder / f"{kind}_world_size_1_rank_{rank}.pt"
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"Incomplete final checkpoint: {path.name}")
    elif stage == "export":
        folder = ROOT / f"checkpoints/{arm}_hf150"
        if not (folder / "config.json").is_file() or not list(folder.glob("*.safetensors")):
            raise RuntimeError("Incomplete Hugging Face export")
    elif stage == "evaluation":
        result = json.loads((ROOT / f"runs/{arm}_seen140/output/summary.json").read_text())
        if not (result["status"] == "passed" and result["step"] == 150 and result["tasks"] == 140
                and result["task_manifest_sha256"] == sha(ROOT / "configs/seen140_tasks.json")):
            raise RuntimeError("Final evaluation did not match fixed seen140")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--allow-api-cost", action="store_true", required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if os.name != "posix":
        p.error("GPU experiment orchestration requires Linux")
    groups = layout()
    if not os.environ.get("RETHINK_API_KEY") or not os.environ.get("RETHINK_API_ENDPOINT"):
        p.error("Configure your own hint API before launching this ablation pipeline")
    import fcntl
    folder = ROOT / "runs/ablation"
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "launch.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error("An ablation launcher is already active; do not duplicate jobs")
    binding = {name: sha(ROOT / name) for name in (
        "configs/ablation.json", "configs/sft.json", "configs/opsd.yaml", "configs/data.json", "configs/upstream.json",
        "configs/seen140_tasks.json", "src/sft/route_bce/core.py", "src/sft/route_bce/train.py", "src/sft/route_bce/distributed.py")}
    state_path = folder / "state.json"
    if state_path.exists():
        if not a.resume:
            p.error("A prior ablation run exists; inspect it and use --resume")
        state = json.loads(state_path.read_text())
        if state["binding"] != binding or state["gpu_groups"] != groups:
            p.error("Run recipe or GPU assignment changed; automatic resume refused")
        # A stale controller must not duplicate still-running child sessions.
        for row in state["arms"].values():
            if row.get("pid") and row.get("status") == "running":
                try:
                    os.kill(row["pid"], 0)
                except ProcessLookupError:
                    pass
                else:
                    p.error("Previous child PID still exists; verify ownership before resume")
    else:
        if any((ROOT / f"checkpoints/{arm}").exists() for arm in ("baseline", "opsd")):
            p.error("Existing OPSD outputs: inspect the saved state before resuming")
        state = {"binding": binding, "gpu_groups": groups, "arms": {
            arm: {"completed_stages": [], "status": "pending"} for arm in ("opsd",)}}
    queues = {arm: commands(arm, a.resume) for arm in state["arms"]}
    active, logs = {}, {}

    def persist():
        state["updated_at_unix"] = time.time()
        write_json(state_path, state)

    def stop_owned(signum, _frame):
        for child in active.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_owned)
    signal.signal(signal.SIGINT, stop_owned)
    try:
        while queues or active:
            for arm in list(queues):
                row = state["arms"][arm]
                if arm in active:
                    continue
                while queues[arm] and queues[arm][0][0] in row["completed_stages"]:
                    verify_stage(arm, queues[arm].pop(0)[0])
                if not queues[arm]:
                    row.update(status="completed", pid=None)
                    del queues[arm]
                    persist()
                    continue
                stage, command = queues[arm].pop(0)
                env = dict(os.environ, RAY_ADDRESS="local",
                           PYTHONPATH=str(ROOT / ".deps/sdar"))
                assigned = groups["method"]
                env["CUDA_VISIBLE_DEVICES"] = ",".join(assigned)
                if stage == "evaluation":
                    env["EVAL_GPU"] = assigned[0]  # retain the existing TP1 final evaluator
                log = (folder / f"{arm}_{stage}.log").open("a", encoding="utf-8")
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                active[arm], logs[arm] = child, log
                row.update(status="running", stage=stage, pid=child.pid)
                persist()
                print(json.dumps({"arm": arm, "stage": stage, "pid": child.pid}), flush=True)
            for arm, child in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                logs.pop(arm).close()
                del active[arm]
                row = state["arms"][arm]
                stage = row["stage"]
                error = None
                if code == 0:
                    try:
                        verify_stage(arm, stage)
                    except Exception as exc:
                        code, error = 1, type(exc).__name__ + ": " + str(exc)
                if code:
                    row.update(status="failed", exit_code=code, error=error, pid=None)
                    queues.pop(arm, None)  # other arm can continue; no blind restart
                else:
                    row["completed_stages"].append(stage)
                    row.update(status="pending", pid=None)
                persist()
                print(json.dumps({"arm": arm, "stage": stage, "exit_code": code}), flush=True)
            if active:
                time.sleep(5)
        if any(x["status"] != "completed" for x in state["arms"].values()):
            raise SystemExit("One experiment failed; inspect its log and complete checkpoint before --resume")
        summary=json.loads((ROOT/"runs/opsd_seen140/output/summary.json").read_text())
        result={key:summary[key] for key in ("status","step","tasks","successes","success_rate","task_manifest_sha256")}
        result.update(ablation=contract()["variant"],evaluation_API_used=False)
        destination=ROOT/"results/tables"
        destination.mkdir(parents=True,exist_ok=True)
        write_json(destination/(contract()["variant"]+"_seen140.json"),result)
    finally:
        for child in active.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for log in logs.values():
            log.close()
        lock.close()


if __name__ == "__main__":
    main()
