from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent

MAX_FILE_SIZE_MB = 5
MAX_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

EXCLUDED_EXTENSIONS = {
    # Images and rendered media (curated figures are tracked separately in Git).
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff",
    ".webp", ".svg", ".ico", ".mp4", ".avi", ".mov",
    # Documents / presentations are optional in the source-only bundle.
    ".pdf", ".docx", ".pptx", ".xlsx",
    # Data archives and model artifacts.
    ".gz", ".zip", ".7z", ".rar", ".pt", ".pth", ".onnx",
    ".h5", ".hdf5", ".npy", ".npz", ".pkl", ".pickle",
    ".joblib", ".parquet", ".pyc", ".pyo",
}
EXCLUDED_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints",
}

def make_archive(output: Path) -> tuple[list[str], list[tuple[str, str]]]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    archived: list[str] = []
    excluded: list[tuple[str, str]] = []

    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if any(part in EXCLUDED_DIRS for part in relative.parts):
                continue
            if path.resolve() == output:
                continue
            if path.suffix.lower() in EXCLUDED_EXTENSIONS:
                excluded.append((relative.as_posix(), "excluded extension"))
                continue

            if path.suffix.lower() == ".ipynb":
                try:
                    notebook = json.loads(path.read_text(encoding="utf-8"))
                    for cell in notebook.get("cells", []):
                        cell.pop("outputs", None)
                        cell.pop("execution_count", None)
                        cell.pop("attachments", None)
                    content = json.dumps(notebook, ensure_ascii=False)
                    if len(content.encode("utf-8")) > MAX_BYTES:
                        excluded.append((relative.as_posix(), "notebook still too large"))
                        continue
                    zf.writestr(relative.as_posix(), content)
                    archived.append(relative.as_posix())
                    continue
                except Exception as exc:
                    excluded.append((relative.as_posix(), f"notebook error: {exc}"))
                    continue

            if path.stat().st_size > MAX_BYTES:
                excluded.append((relative.as_posix(), "over size limit"))
                continue
            zf.write(path, relative.as_posix())
            archived.append(relative.as_posix())

    return archived, excluded

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a source-only project archive.")
    parser.add_argument("--output", type=Path, default=ROOT.parent / f"{ROOT.name}.zip")
    args = parser.parse_args()
    archived, excluded = make_archive(args.output)
    print(f"Archive: {args.output.resolve()}")
    print(f"Archived files: {len(archived)}")
    print(f"Excluded files: {len(excluded)}")
    print(f"ZIP size: {args.output.resolve().stat().st_size / 1024 / 1024:.2f} MB")
    if excluded:
        print("\nExcluded:")
        for path, reason in excluded:
            print(f"- {path} [{reason}]")
