"""Package portable source and tests without datasets, environments, or experiment outputs."""

from pathlib import Path
import zipfile


def main():
    """Create one source archive that extracts to its own project folder."""
    root = Path(__file__).resolve().parent
    destination = root / "release" / f"{root.name}.zip"
    destination.parent.mkdir(exist_ok=True)
    files = list(root.glob("*.py")) + list(root.glob("*.md"))
    files += [root / "requirements.txt", root / "environment.yml", root / ".gitignore"]
    for directory, pattern in (("pfhmm_vision", "*.py"), ("tests", "*.py"), ("configs", "*.json")):
        files.extend((root / directory).glob(pattern))
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, arcname=str(Path(root.name) / path.relative_to(root)))
    print(destination)


if __name__ == "__main__":
    main()
