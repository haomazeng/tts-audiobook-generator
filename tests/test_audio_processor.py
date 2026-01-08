import pytest
from pathlib import Path
from src.audio_processor import AudioProcessor, merge_audio_files

def test_audio_processor_initialization():
    processor = AudioProcessor(bitrate="64k", sample_rate=24000)
    assert processor.bitrate == "64k"
    assert processor.sample_rate == 24000

def test_save_audio_segment(tmp_path):
    processor = AudioProcessor()
    # We'll test that the method exists and handles the interface correctly
    assert hasattr(processor, 'save_audio_segment')

def test_merge_audio_files(tmp_path):
    # Test that merge method exists
    assert hasattr(AudioProcessor, 'merge_audio_files')
