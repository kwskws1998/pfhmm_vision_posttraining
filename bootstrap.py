"""Install pinned CPU/CUDA wheels into a dedicated Conda environment using Python only."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def pins(root):
    """Read the one canonical set of exact dependency versions."""
    result = {}
    for raw in (Path(root) / "requirements.txt").read_text().splitlines():
        line = raw.strip()
        if line:
            name, version = line.split("==")
            result[name] = version
    return result


def wheel_requirements(expected, device, system):
    """Select exact wheel build tags so an existing CPU build cannot satisfy a CUDA request."""
    if system not in ("Linux", "Darwin"):
        raise ValueError("The supported pilot hosts are Linux CUDA/CPU and macOS CPU")
    if device == "cuda" and system != "Linux":
        raise ValueError("CUDA experiments require Linux")
    suffix = "+cu128" if device == "cuda" else ("+cpu" if system == "Linux" else "")
    return [f"{name}=={expected[name]}{suffix}" for name in ("torch", "torchvision")]


def environment_probe(expected, device):
    """Verify a real Conda prefix and execute kernels on the requested device."""
    import torch
    import torchvision
    prefix = Path(sys.prefix).resolve()
    if not (prefix / "conda-meta" / "history").is_file():
        raise RuntimeError("The requested Python is not inside a Conda environment")
    for name, version in expected.items():
        actual = importlib.metadata.version(name)
        if actual.split("+", 1)[0] != version:
            raise RuntimeError(f"Expected {name}=={version}, observed {actual}")
    if not Path(torch.__file__).resolve().is_relative_to(prefix):
        raise RuntimeError("PyTorch was imported from outside the requested Conda prefix")
    if not Path(torchvision.__file__).resolve().is_relative_to(prefix):
        raise RuntimeError("torchvision was imported from outside the requested Conda prefix")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("This recipe requires Python 3.11")
    target = "cpu"
    if device == "cuda":
        if not torch.cuda.is_available() or torch.version.cuda != "12.8":
            raise RuntimeError("CUDA 12.8 PyTorch and an available compatible GPU are required")
        target = "cuda"
    elif torch.version.cuda is not None:
        raise RuntimeError("The CPU recipe requires a CPU-only PyTorch build")
    inputs = torch.randn(2, 3, 16, 16, device=target, requires_grad=True)
    layer = torch.nn.Conv2d(3, 4, 3).to(target)
    loss = layer(inputs).square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not torch.isfinite(inputs.grad).all():
        raise RuntimeError("The requested device produced nonfinite convolution gradients")
    if target == "cuda":
        torch.cuda.synchronize()
    result = {"prefix": str(prefix), "python": platform.python_version(),
              "torch": torch.__version__, "torchvision": torchvision.__version__,
              "cuda_runtime": torch.version.cuda, "kernel_test_passed": True,
              "requested_device": device,
              "packages": {name: importlib.metadata.version(name) for name in expected}}
    if target == "cuda":
        result["gpu"] = torch.cuda.get_device_name(0)
    return result


def setup(root, device, prefix=None, conda=None, verify_only=False):
    """Create one isolated Conda environment without touching base or an existing unrelated folder."""
    root = Path(root).resolve()
    prefix = Path(prefix or root / ".conda").resolve()
    conda = conda or os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise RuntimeError("Conda is required; pass --conda /absolute/path/to/conda")
    wheels = wheel_requirements(pins(root), device, platform.system())
    base = Path(subprocess.check_output([conda, "info", "--base"], text=True).strip()).resolve()
    if prefix == base or prefix == root or prefix in root.parents:
        raise ValueError("Choose a dedicated environment directory; modifying Conda base is forbidden")
    marker = prefix / "conda-meta" / "history"
    if prefix.exists() and not marker.is_file():
        raise ValueError("The selected existing prefix is not a Conda environment")
    environment = {**os.environ, "PYTHONNOUSERSITE": "1",
                   "PIP_CACHE_DIR": str(root / ".cache" / "pip"),
                   "CONDA_PKGS_DIRS": str(root / ".cache" / "conda_pkgs")}
    if not marker.is_file():
        if verify_only:
            raise RuntimeError("The Conda environment does not exist yet")
        subprocess.run([conda, "create", "--yes", "--prefix", str(prefix), "--override-channels",
                        "--channel", "conda-forge", "python=3.11", "pip"], check=True, env=environment)
    executable = prefix / "bin" / "python"
    expected = pins(root)
    if not verify_only:
        index = f"https://download.pytorch.org/whl/{'cu128' if device == 'cuda' else 'cpu'}"
        subprocess.run([str(executable), "-m", "pip", "install", "--index-url", index,
                        *wheels],
                       check=True, env=environment)
        subprocess.run([str(executable), "-m", "pip", "install", "-r", str(root / "requirements.txt")],
                       check=True, env=environment)
    command = [str(executable), str(root / "bootstrap.py"), "--internal-probe", device]
    result = subprocess.run(command, check=True, env=environment, text=True, capture_output=True)
    report = json.loads(result.stdout)
    if Path(report["prefix"]).resolve() != prefix:
        raise RuntimeError("The installed interpreter did not use the requested prefix")
    destination = root / "verification"
    destination.mkdir(exist_ok=True)
    (destination / f"environment_{device}.json").write_text(json.dumps(report, indent=2) + "\n")
    frozen = subprocess.check_output([str(executable), "-m", "pip", "freeze"], env=environment, text=True)
    (destination / f"requirements_{device}_observed.txt").write_text(frozen)
    print(json.dumps(report, indent=2), flush=True)
    return executable


def main():
    """Provide a standard-library-only installer usable before project dependencies exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prefix")
    parser.add_argument("--conda")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--internal-probe", choices=("cpu", "cuda"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.internal_probe:
        print(json.dumps(environment_probe(pins(root), args.internal_probe)))
        return
    setup(root, args.device, args.prefix, args.conda, args.verify_only)


if __name__ == "__main__":
    main()
