"""Check the search index has not been changed before it is loaded.

FAISS stores part of the index as a pickle file, and loading a pickle can run code. So the app
only loads the index if its files still match the checksums written when the index was built.
"""
import hashlib
import json
from pathlib import Path

INDEX_FILES = ("index.faiss", "index.pkl")
CHECKSUM_FILE = "checksums.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(folder: str = "vectorstore") -> dict:
    folder_path = Path(folder)
    sums = {name: sha256_of(folder_path / name) for name in INDEX_FILES}
    (folder_path / CHECKSUM_FILE).write_text(json.dumps(sums, indent=2) + "\n")
    return sums


def verify_checksums(folder: str = "vectorstore") -> bool:
    folder_path = Path(folder)
    try:
        expected = json.loads((folder_path / CHECKSUM_FILE).read_text())
        return all(sha256_of(folder_path / name) == expected.get(name) for name in INDEX_FILES)
    except (OSError, ValueError):
        return False
