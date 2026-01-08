import pytest
from pathlib import Path
from src.progress import ProgressTracker

def test_progress_tracker_initialization(tmp_path):
    tracker = ProgressTracker("test_file.mp3", tmp_path)
    assert tracker.total_chunks == 0

def test_progress_tracker_update(tmp_path):
    tracker = ProgressTracker("test.mp3", tmp_path, total_chunks=10)
    # Mark first 5 chunks as complete
    for i in range(5):
        tracker.update(i)
    assert tracker.get_progress() == 0.5

def test_progress_tracker_persistence(tmp_path):
    tracker = ProgressTracker("test.mp3", tmp_path, total_chunks=10)
    # Mark first 5 chunks as complete
    for i in range(5):
        tracker.update(i)

    # Load new instance
    tracker2 = ProgressTracker("test.mp3", tmp_path, total_chunks=10)
    assert tracker2.get_completed() == 5

def test_progress_tracker_completion(tmp_path):
    tracker = ProgressTracker("test.mp3", tmp_path, total_chunks=10)
    # Mark all 10 chunks as complete
    for i in range(10):
        tracker.update(i)
    assert tracker.is_complete()

def test_progress_tracker_reset(tmp_path):
    tracker = ProgressTracker("test.mp3", tmp_path, total_chunks=10)
    # Mark some chunks as complete
    for i in range(5):
        tracker.update(i)
    tracker.reset()
    assert tracker.get_completed() == 0
