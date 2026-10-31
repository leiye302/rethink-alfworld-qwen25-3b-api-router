"""Real four-process CPU DDP test of the same accumulation used by GPU SFT."""
from datetime import timedelta
import json
from pathlib import Path
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from route_bce import distributed as parallel
from route_bce.core import losses
from route_bce.train import train_batch


def rows():
    result = []
    for i in range(8):
        label = int(i >= 5)
        ids = torch.tensor([1, 2, 3, 8 if label else 4] + [5] * (1 + i % 3) + [2])
        result.append({"route_label": label, "ids": ids, "position": i})
    return result


def forward(model, item, marker, config):
    # Variable suffix lengths expose accidental token-mean vs sample-mean changes.
    inputs = torch.nn.functional.one_hot(item["ids"], num_classes=12).double()
    return losses(model(inputs), item["ids"], 3, item["route_label"], marker, .1)


def step(model, optimizer, sample_rows):
    optimizer.zero_grad(set_to_none=True)
    records = train_batch(model, sample_rows, list(range(8)), 8, {"global_batch_size": 8}, forward)
    gradients = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return records, gradients, norm


def worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(27)
        model = DistributedDataParallel(torch.nn.Linear(12, 12).double())
        opt = torch.optim.AdamW(model.parameters(), lr=2e-6, betas=(.9, .95), weight_decay=.01)
        first, gradient, norm = step(model, opt, rows())
        rngs = parallel.gather(parallel.rng_state())
        # Persist and reload each rank state, exercising exact DDP RNG mapping.
        path = Path(directory) / f"rank{rank}.pt"
        torch.save({"weights": model.module.state_dict(), "optimizer": opt.state_dict(),
                    "rngs": rngs}, path)
        state = torch.load(path, weights_only=False)
        restored = torch.nn.Linear(12, 12).double()
        restored.load_state_dict(state["weights"])
        restored = DistributedDataParallel(restored)
        opt2 = torch.optim.AdamW(restored.parameters(), lr=2e-6, betas=(.9, .95), weight_decay=.01)
        opt2.load_state_dict(state["optimizer"])
        parallel.restore_rng(state["rngs"][rank])
        second, gradient2, norm2 = step(restored, opt2, rows()[::-1])
        torch.save({"first": first, "second": second, "gradient": gradient, "gradient2": gradient2,
                    "norm": norm, "norm2": norm2, "weights": restored.module.state_dict()},
                   Path(directory) / f"result{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="PyTorch Gloo required")
@pytest.mark.skipif(os.name == "nt", reason="Pinned torch 2.8 Windows wheel lacks Gloo transport; Linux CI runs the four-process test")
def test_four_rank_matches_serial_loss_gradient_adam_and_resume(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(27)
    serial = torch.nn.Linear(12, 12).double()
    opt = torch.optim.AdamW(serial.parameters(), lr=2e-6, betas=(.9, .95), weight_decay=.01)
    first, gradient, norm = step(serial, opt, rows())
    second, gradient2, norm2 = step(serial, opt, rows()[::-1])
    mp.spawn(worker, args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)), nprocs=4, join=True)
    for rank in range(4):
        result = torch.load(tmp_path / f"result{rank}.pt", weights_only=False)
        assert result["first"] == first and result["second"] == second
        for key, value in (("gradient", gradient), ("gradient2", gradient2), ("norm", norm), ("norm2", norm2)):
            torch.testing.assert_close(result[key], value, atol=1e-8, rtol=1e-6)
        for key, value in serial.state_dict().items():
            torch.testing.assert_close(result["weights"][key], value, atol=1e-10, rtol=1e-6)


def test_four_replica_arithmetic_matches_serial_sample_mean():
    """Transport-independent check; does NOT replace the real Linux DDP test."""
    import copy
    torch.manual_seed(27)
    serial = torch.nn.Linear(12, 12).double()
    replicas = [copy.deepcopy(serial) for _ in range(4)]
    for item in rows():
        (forward(serial, item, 8, {})["loss"] / 8).backward()
    for rank, model in enumerate(replicas):
        for index in parallel.local_indices(list(range(8)), rank, 4):
            (forward(model, rows()[index], 8, {})["loss"] * (4 / 8)).backward()
    for index, param in enumerate(serial.parameters()):
        averaged = torch.stack([list(m.parameters())[index].grad for m in replicas]).mean(0)
        torch.testing.assert_close(param.grad, averaged, atol=1e-8, rtol=1e-6)


def test_global_batch_partition_is_exact_no_per_rank_quota():
    groups = [parallel.local_indices(list(range(8)), r, 4) for r in range(4)]
    assert groups == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert sorted(i for group in groups for i in group) == list(range(8))
    assert sum(rows()[i]["route_label"] for group in groups for i in group) == 3
    with pytest.raises(ValueError):
        parallel.local_indices(list(range(7)), 0, 4)
