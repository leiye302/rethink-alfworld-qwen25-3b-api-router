"""Apply the user's target contract; never infer a reasoning label from action.

This pure helper does not annotate, export a dataset, run an Oracle, or train.
The caller must bind source rows to reviewed IDs and the pinned source SHA.
Stored Oracle evidence is inherited, not a fresh environment replay.
"""
from __future__ import annotations

CONTRACT = "route-semantic-action-target-v2"


def _action(row: dict, field: str) -> str:
    action = row.get(field)
    if not isinstance(action, str) or not action.strip():
        raise ValueError(f"Missing original/audited action: {field}")
    if any(c in action for c in "<>\r\n"):
        raise ValueError(f"Action contains protocol/control text: {field}")
    return action  # No rewriting, normalization, or Oracle fallback for Good.


def assemble_target(row: dict, label: str) -> dict | None:
    if label == "ignore":
        return None
    if label not in {"good", "bad"}:
        raise ValueError("Explicit reviewed good/bad/ignore label required")
    source = row.get("source_executed_action")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Missing original action: source_executed_action")
    if label == "good":
        action, field, marker = _action(row, "source_executed_action"), "source_executed_action", ""
    else:
        action, field, marker = _action(row, "oracle_action"), "oracle_action", "<RETHINK>"
        admissible = row.get("admissible_actions_before")
        if not isinstance(admissible, list) or action not in admissible:
            raise ValueError("Stored Oracle action lacks current admissibility evidence")
    return {
        "contract": CONTRACT,
        "route": label,
        "response": f"{marker}<action>{action}</action>",
        "supervision_action": action,
        "action_source_field": field,
        "action_equals_original": action == source,
        "reasoning_label_changed_by_action": False,
        "new_environment_replay_performed": False,
    }
