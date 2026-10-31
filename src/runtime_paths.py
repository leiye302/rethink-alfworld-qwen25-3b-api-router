"""Machine-independent, short Ray socket directories (no training-state storage)."""
import os
from pathlib import Path
import tempfile


def new_ray_directory():
    # Unix-domain sockets have a short path limit. A nested repository/run path
    # can exceed it even though the checkpoint paths themselves are valid.
    base = Path(os.environ.get('RETHINK_RUNTIME_ROOT', tempfile.gettempdir())).resolve()
    if not base.is_dir():
        raise ValueError('RETHINK_RUNTIME_ROOT must be an existing writable directory')
    if os.name == 'posix' and len(os.fsencode(base)) > 25:
        raise ValueError('Choose a shorter RETHINK_RUNTIME_ROOT for Ray Unix sockets')
    # Keep logs after exit for diagnostics; never remove a shared temporary root.
    return Path(tempfile.mkdtemp(prefix='rh-ray-', dir=base))
