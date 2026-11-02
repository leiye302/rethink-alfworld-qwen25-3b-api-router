"""Pinned seen140 identity, independent of harmless JSON serialization changes."""
import hashlib
import json
from pathlib import Path

# Both raw manifests were independently compared: identical ordered rows, seeds,
# task/game hashes and splits. Do not accept arbitrary caller-supplied hashes.
SEMANTIC_SHA256 = '0bf2e79a39098cd511eafc0ddabac12939fa3659316ee5dbe8fa5696c6c51916'
LEGACY_RAW_SHA256 = frozenset({
    '4421ab3e6db4cd2c4e74b4e6bcfe2c49cd8ff5f31663aa436f1580ada981afc2',
    'bb3a1d67751b04ef0d03bcba0bd4b7c97d98f86ebcf20160985918256f957cb8',
})


def semantic_sha256(manifest):
    rows = json.loads(Path(manifest).read_text(encoding='utf-8'))
    if not isinstance(rows, list) or len(rows) != 140:
        raise ValueError('Expected the complete ordered seen140 manifest')
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != SEMANTIC_SHA256:
        raise ValueError('seen140 task identity, order, seed, game hash or split changed')
    return digest


def require_seen140(summary, manifest):
    semantic_sha256(manifest)
    if summary.get('task_manifest_sha256') not in LEGACY_RAW_SHA256:
        raise ValueError('Unrecognized seen140 manifest checksum')
    semantic = summary.get('task_manifest_semantic_sha256')
    if semantic is not None and semantic != SEMANTIC_SHA256:
        raise ValueError('Conflicting seen140 semantic identity')
