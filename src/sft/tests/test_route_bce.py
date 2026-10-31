import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from legacy.stage1_training import masked_sft_loss
from route_bce.core import batches, encode_row, generation_metrics, load_rows, losses, register_token
from route_bce.train import check_config, checkpoint, make_scheduler, safe_child, training_order

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def tokenizer():
    path = Path(os.environ.get("RETHINK_TOKENIZER", ROOT / "checkpoints/Qwen2.5-3B-Instruct"))
    if not path.is_dir():
        pytest.skip("Set RETHINK_TOKENIZER to the separately licensed base tokenizer directory")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    register_token(tok)
    return tok


def example(label):
    action = "go to desk 1"
    return {"audit_id": "test", "route": label, "prompt": "Current observation: a desk is visible.",
            "assistant_prefix": "<think>\nI should inspect the desk.\n</think>\n",
            "response": ("<RETHINK>" if label == "bad" else "") + f"<action>{action}</action>",
            "supervision_action": action, "source_executed_action": action,
            "admissible_actions_before": [action]}


@pytest.mark.parametrize("label", [0, 1])
def test_full_vocab_bce_and_offset_invariance(label):
    torch.manual_seed(4)
    ids = torch.tensor([0, 1, 3 if label else 2, 4, 1])
    logits = torch.randn(5, 7, requires_grad=True)
    out = losses(logits, ids, 2, label, 3)
    q = logits[1].softmax(-1)[3]
    wanted = -torch.log(q) if label else -torch.log1p(-q)
    torch.testing.assert_close(out["route_bce"], wanted)
    torch.testing.assert_close(out["marker_probability"], q)
    shifted = losses(logits + 100, ids, 2, label, 3)
    torch.testing.assert_close(out["route_bce"], shifted["route_bce"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out["loss"], out["suffix_ce"] + .1 * wanted)


@pytest.mark.parametrize("label", [0, 1])
def test_exact_position_and_no_context_token_ce(label):
    logits = torch.randn(6, 8, requires_grad=True)
    ids = torch.tensor([0, 1, 5, 4 if label else 3, 2, 1])
    out = losses(logits, ids, 3, label, 4)
    out["loss"].backward()
    assert logits.grad[:2].abs().sum() == 0  # mask all H/R token targets
    assert logits.grad[2].abs().sum() > 0  # final R predicts first suffix token
    assert logits.grad[3:5].abs().sum() > 0  # action/EOS remain supervised
    assert logits.grad[-1].abs().sum() == 0


@pytest.mark.parametrize("magnitude,label", [(1000., 0), (-1000., 1), (1000., 1), (-1000., 0)])
def test_extreme_logit_stability(magnitude, label):
    z = torch.zeros(3, 6, requires_grad=True)
    with torch.no_grad(): z[0, 4] = magnitude
    ids = torch.tensor([0, 4 if label else 2, 1])
    result = losses(z, ids, 1, label, 4)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert torch.isfinite(z.grad).all()


def test_sample_mean_not_long_suffix_weighted():
    ce = torch.tensor([[1., 3., 0., 0.], [4., 4., 4., 4.]])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    assert masked_sft_loss(ce, mask, reduction="sample_mean") == 3.


@pytest.mark.parametrize("label", ["good", "bad"])
def test_tokens_and_mask_identical_to_legacy_dataset(tokenizer, label):
    row = example(label)
    encoded = encode_row(row, tokenizer)
    # Execute the unchanged legacy __getitem__ method without importing its
    # unrelated RL/vision dependencies. This is a real implementation diff.
    tree = ast.parse((ROOT / "src/sft/legacy/sft_dataset.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SFTDataset")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__getitem__")
    ns = {"torch": torch, "compute_position_id_with_mask": lambda m: (m.cumsum(-1) - 1).clamp_min(0)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "legacy_getitem", "exec"), ns)
    obj = SimpleNamespace(tokenizer=tokenizer, prompts=[row["prompt"]], responses=[row["response"]],
        assistant_prefixes=[row["assistant_prefix"]], enable_thinking_values=[None],
        apply_chat_template_kwargs={}, mask_response_thinking=False, reasoning_kl_on_masked_thinking=False,
        max_length=256, truncation="error", route_labels=[int(label == "bad")])
    legacy = ns["__getitem__"](obj, 0)
    size = len(encoded["input_ids"])
    assert legacy["input_ids"][:size].tolist() == encoded["input_ids"]
    positions = legacy["loss_mask"].nonzero().flatten().tolist()
    assert positions == list(range(encoded["suffix_start"] - 1, size - 1))
    assert encoded["joint_tokenization_matches"]


def test_good_original_bad_can_equal_original(tokenizer):
    good, bad = example("good"), example("bad")
    encode_row(good, tokenizer)
    encode_row(bad, tokenizer)  # deliberately SAME action, allowed
    good["source_executed_action"] = "look"
    with pytest.raises(ValueError, match="Good action"): encode_row(good, tokenizer)
    bad["admissible_actions_before"] = []
    with pytest.raises(ValueError, match="admissibility"): encode_row(bad, tokenizer)


def test_ignore_and_truncation_rejected(tokenizer):
    row = example("ignore")
    with pytest.raises(ValueError): encode_row(row, tokenizer)
    with pytest.raises(ValueError, match="overlength"): encode_row(example("good"), tokenizer, 2)


def test_marker_registration_never_shrinks(tokenizer):
    config = Qwen2Config(vocab_size=len(tokenizer) + 4, hidden_size=8, intermediate_size=16,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
    model = Qwen2ForCausalLM(config)
    before = model.get_input_embeddings().weight.detach().clone()
    state = register_token(tokenizer, model)
    assert not state["resized"]
    assert torch.equal(before, model.get_input_embeddings().weight)


def test_one_epoch_fixed_five_three_no_repeat():
    rows = [{"route": "good"}] * 10000 + [{"route": "bad"}] * 6000
    order = batches(rows)
    assert len(order) == 2000 and order == batches(rows)
    for group in order:
        assert sum(rows[i]["route"] == "good" for i in group) == 5
    assert len({i for group in order for i in group}) == 16000
    with pytest.raises(ValueError): batches(rows[:-1])


def test_generation_rejects_illegal_protocol():
    invalid = ["<action>look</action><think>x</think>", "<think></think><action>look</action>",
               "<think>x</think><action>look</action><action>look</action>",
               "<think>x</think><action>look</action><RETHINK>"]
    for text in invalid:
        assert not generation_metrics(text, [1, 2], 3, [2])["strict_protocol"]
    result = generation_metrics("<think>x</think><RETHINK><action>look</action><|im_end|>", [4, 3, 2], 3, [2])
    assert result["strict_protocol"] and result["marker_present"] and result["marker_text_agrees"]


def test_tiny_model_cpu_backward_and_optimizer_step():
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=24, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1))
    ids = torch.tensor([[1, 2, 3, 8, 4, 2]])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-6, betas=(.9, .95), weight_decay=.01)
    before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    out = losses(model(ids).logits[0], ids[0], 3, 1, 8)
    out["loss"].backward()
    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    opt.step()
    assert not torch.equal(before, model.model.layers[0].self_attn.q_proj.weight)


def test_checkpoint_atomic_export_and_latest_resume(tmp_path, tokenizer):
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=24, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
    best = checkpoint(model, tokenizer, opt, sched, 500, tmp_path, {"test": True}, {"all": {"loss": 1.}}, None)
    assert best["step"] == 500
    assert (tmp_path / "global_step_500/sft_stage_manifest.json").exists()
    assert not (tmp_path / "global_step_500.incomplete").exists()
    with pytest.raises(FileExistsError): checkpoint(model, tokenizer, opt, sched, 500, tmp_path, {}, {}, best)
    checkpoint(model, tokenizer, opt, sched, 1000, tmp_path, {"test": True}, {"all": {"loss": 1.1}}, best)
    assert json.loads((tmp_path / "latest_resume.json").read_text())["step"] == 1000
    assert json.loads((tmp_path / "best_saved_checkpoint.json").read_text())["step"] == 500
    assert (tmp_path / "global_step_500").exists()  # old weights never deleted


def test_launch_config_and_safe_paths(tmp_path):
    c = json.loads((ROOT / "configs/sft.json").read_text())
    check_config(c)
    c["route_lambda"] = 1.
    with pytest.raises(ValueError): check_config(c)
    with pytest.raises(ValueError): safe_child(tmp_path, "../other_project")


def test_first_epoch_exact_coverage_and_frozen_recipe():
    c = json.loads((ROOT / "configs/sft.json").read_text())
    check_config(c)
    rows = [{"route": "good"}] * 10000 + [{"route": "bad"}] * 6000
    order = training_order(rows, c)
    assert len(order) == 2000
    assert order == batches(rows, epoch=0)
    assert sorted(i for group in order for i in group) == list(range(16000))
    assert all(sum(rows[i]["route"] == "good" for i in group) == 5 for group in order)
    c["total_steps"] = 4000
    with pytest.raises(ValueError): check_config(c)


def test_stop2000_preserves_legacy_first2000_learning_rates():
    from transformers import get_cosine_schedule_with_warmup
    c = json.loads((ROOT / "configs/sft.json").read_text())
    a, b = [torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=c["lr"]) for _ in range(2)]
    current = make_scheduler(a, c)
    legacy = get_cosine_schedule_with_warmup(b, 400, 4000)
    for _ in range(2000):
        assert a.param_groups[0]["lr"] == b.param_groups[0]["lr"]
        a.step(); b.step(); current.step(); legacy.step()
    assert current.get_last_lr() == legacy.get_last_lr()
    assert current.get_last_lr()[0] > 0
    c["scheduler_total_steps"] = 2000
    with pytest.raises(ValueError): check_config(c)


def test_epoch_full_state_and_exact_optimizer_resume(tmp_path, tokenizer):
    from route_bce.core import sha
    torch.manual_seed(10)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=24, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: 1. / (n + 1))
    ids = torch.tensor([[1, 2, 3, 8, 4, 2]])
    def update(m, o, s):
        o.zero_grad(set_to_none=True)
        losses(m(ids).logits[0], ids[0], 3, 1, 8)["loss"].backward()
        o.step(); s.step()
    update(model, opt, sched)
    checkpoint(model, tokenizer, opt, sched, 1500, tmp_path, {}, {"all": {"loss": 1.}}, None)
    checkpoint(model, tokenizer, opt, sched, 2000, tmp_path, {}, {"all": {"loss": .9}}, None,
        epoch_end=True, data_position={"next_global_batch": 2000, "next_epoch_index": 1})
    cp = tmp_path / "global_step_2000"
    manifest = json.loads((cp / "sft_stage_manifest.json").read_text())
    assert manifest["epoch_end"] and not manifest["weights_only_directory"]
    assert sha(cp / "training_state.pt") == manifest["training_state"]["sha256"]
    assert not (tmp_path / "resume_state_step_1500.pt").exists()
    restored = Qwen2ForCausalLM.from_pretrained(cp, torch_dtype=torch.float32)
    restored.train()
    state = torch.load(cp / "training_state.pt", map_location="cpu", weights_only=False)
    opt2 = torch.optim.AdamW(restored.parameters(), lr=2e-6)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda n: 1. / (n + 1))
    opt2.load_state_dict(state["optimizer"]); sched2.load_state_dict(state["scheduler"])
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(opt.state.values(), opt2.state.values()):
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    update(model, opt, sched)
    update(restored, opt2, sched2)
    for a, b in zip(model.parameters(), restored.parameters()):
        # HF may select a different CPU attention kernel on reload; states above
        # are bit-exact, while the following update permits float32 roundoff.
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-10)
    checkpoint(model, tokenizer, opt, sched, 2500, tmp_path, {}, {"all": {"loss": 1.}}, None)
    assert (cp / "training_state.pt").exists()
    assert json.loads((tmp_path / "latest_resume.json").read_text())["state"] == "resume_state_step_2500.pt"
