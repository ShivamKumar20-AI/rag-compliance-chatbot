import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from integrity import CHECKSUM_FILE, INDEX_FILES, verify_checksums, write_checksums  # noqa: E402
from ratelimit import SlidingWindowLimiter  # noqa: E402


def make_index(folder: Path):
    folder.mkdir()
    for name in INDEX_FILES:
        (folder / name).write_bytes(b"original " + name.encode())


def test_unchanged_index_passes(tmp_path):
    make_index(tmp_path / "vs")
    write_checksums(str(tmp_path / "vs"))
    assert verify_checksums(str(tmp_path / "vs"))


def test_changed_index_file_is_refused(tmp_path):
    make_index(tmp_path / "vs")
    write_checksums(str(tmp_path / "vs"))
    (tmp_path / "vs" / "index.pkl").write_bytes(b"swapped for something else")
    assert not verify_checksums(str(tmp_path / "vs"))


def test_missing_or_broken_checksums_are_refused(tmp_path):
    make_index(tmp_path / "vs")
    assert not verify_checksums(str(tmp_path / "vs"))            # no checksum file
    (tmp_path / "vs" / CHECKSUM_FILE).write_text("not json")
    assert not verify_checksums(str(tmp_path / "vs"))


def test_the_committed_index_matches_its_checksums():
    assert verify_checksums(str(Path(__file__).resolve().parent.parent / "vectorstore"))


def test_shared_rate_limit_blocks_then_recovers():
    limiter = SlidingWindowLimiter(max_requests=3, window_seconds=60)
    assert [limiter.allow(now=t) for t in (0, 1, 2, 3)] == [True, True, True, False]
    assert limiter.allow(now=61) is True
