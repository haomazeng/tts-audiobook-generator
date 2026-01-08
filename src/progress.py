import json
from pathlib import Path
from typing import Set
import logging

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Track conversion progress for resumable downloads.
    Stores progress in .progress files alongside output files.
    """

    def __init__(self, output_file: str, work_dir: str, total_chunks: int = 0):
        self.output_file = output_file
        self.work_dir = Path(work_dir)
        self.total_chunks = total_chunks
        self.progress_file = self.work_dir / f"{Path(output_file).stem}.progress"
        self.completed_chunks: Set[int] = self._load_progress()

    def _load_progress(self) -> Set[int]:
        """Load existing progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('completed', []))
            except Exception as e:
                logger.warning(f"Failed to load progress: {e}")
                return set()
        return set()

    def _save_progress(self) -> None:
        """Save current progress to file."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({
                    'output_file': self.output_file,
                    'total_chunks': self.total_chunks,
                    'completed': list(self.completed_chunks)
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    def update(self, chunk_index: int) -> None:
        """Mark a chunk as completed."""
        self.completed_chunks.add(chunk_index)
        self._save_progress()
        logger.debug(f"Progress: {len(self.completed_chunks)}/{self.total_chunks}")

    def get_completed(self) -> int:
        """Get number of completed chunks."""
        return len(self.completed_chunks)

    def get_progress(self) -> float:
        """Get progress as fraction (0.0 to 1.0)."""
        if self.total_chunks == 0:
            return 0.0
        return len(self.completed_chunks) / self.total_chunks

    def is_complete(self) -> bool:
        """Check if all chunks are completed."""
        return self.total_chunks > 0 and len(self.completed_chunks) >= self.total_chunks

    def is_chunk_completed(self, chunk_index: int) -> bool:
        """Check if a specific chunk is already completed."""
        return chunk_index in self.completed_chunks

    def reset(self) -> None:
        """Reset progress."""
        self.completed_chunks.clear()
        if self.progress_file.exists():
            self.progress_file.unlink()
        logger.info("Progress reset")

    def cleanup(self) -> None:
        """Remove progress file after successful completion."""
        if self.progress_file.exists():
            self.progress_file.unlink()
            logger.info("Progress file cleaned up")
