"""Atomic artifacts, execution locks, deterministic state, and content fingerprints."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def seed_everything(seed, threads=4):
    """Seed model operations and enable deterministic kernels for this process."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def file_hash(path):
    """Compute a streaming checksum without loading a checkpoint into memory."""
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def code_hash():
    """Fingerprint executable project sources, excluding data and generated artifacts."""
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "pfhmm_vision").glob("*.py"))
    sources += sorted(root.glob("*.py"))
    value = hashlib.sha256()
    for path in sources:
        value.update(path.relative_to(root).as_posix().encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def atomic_json(path, value):
    """Replace JSON only after a complete, finite document has been written."""
    _atomic_write(path, lambda temporary: temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"))


def atomic_torch(path, value):
    """Write a complete checkpoint before replacing its previous committed version."""
    _atomic_write(path, lambda temporary: torch.save(value, temporary))


def atomic_npz(path, **values):
    """Commit numeric arrays without pickle payloads."""
    def write(temporary):
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **values)
    _atomic_write(path, write)


def _atomic_write(path, writer):
    """Keep temporary files on the same filesystem as their final destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    """Load only locally generated, trusted checkpoints containing optimizer/RNG state."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def rng_state():
    """Capture all stochastic model state needed for an exact training continuation."""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    """Restore captured RNGs after constructing models and optimizers."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def execution_lock(root):
    """Prevent two Linux/macOS processes from writing the same experiment directory."""
    import fcntl
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".execution.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is writing to {root}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def resolve_device(name):
    """Fail immediately when requested CUDA execution is unavailable."""
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; verify the Conda PyTorch installation")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("The requested CUDA device does not exist")
        probe = torch.ones(2, device=device, requires_grad=True)
        probe.square().sum().backward()
        torch.cuda.synchronize(device)
    elif device.type != "cpu":
        raise ValueError("This reproducible pilot supports cpu and cuda devices")
    return device
