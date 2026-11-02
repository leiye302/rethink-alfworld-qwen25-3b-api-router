"""Machine-independent, short Ray socket directories (no training-state storage)."""
import os
from pathlib import Path
import shutil
import tempfile


def resolve_python(value, root):
    """Resolve a PATH command or a repository-relative interpreter before launch."""
    if not value or not value.strip():
        raise ValueError('OPSD_PYTHON must name a Python interpreter')
    if not any(separator in value for separator in ('/', '\\')):
        found = shutil.which(value)
        if found is None:
            raise ValueError('OPSD_PYTHON was not found on PATH')
        path = Path(found).resolve()
    else:
        path = Path(value)
        path = (path if path.is_absolute() else Path(root) / path).resolve()
    if not path.is_file() or (os.name == 'posix' and not os.access(path, os.X_OK)):
        raise ValueError('OPSD_PYTHON must be an existing executable interpreter')
    return str(path)


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
