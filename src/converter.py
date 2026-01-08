import asyncio
from pathlib import Path
from typing import List, Optional
import logging

from .tts_client import CosyVoiceClient, TTSRequestError
from .audio_processor import AudioProcessor
from .text_extractor import extract_and_split
from .progress import ProgressTracker
from .config import Config

logger = logging.getLogger(__name__)


class AudioConverter:
    """
    Main conversion engine: orchestrates text extraction, TTS synthesis, and audio merging.
    """

    def __init__(self, config: Config, work_dir: str = "./output"):
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.audio_processor = AudioProcessor(
            format=config.audio_format,
            bitrate=config.audio_bitrate,
            sample_rate=config.audio_sample_rate
        )

    async def convert_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        resume: bool = True
    ) -> str:
        """
        Convert a single file to audio.

        Args:
            input_file: Path to input PDF or MD file
            output_file: Path to output MP3 file (optional)
            resume: Whether to resume from previous progress

        Returns:
            Path to generated audio file
        """
        input_path = Path(input_file)
        if not output_file:
            output_file = self.work_dir / f"{input_path.stem}.mp3"
        else:
            output_file = Path(output_file)

        logger.info(f"Converting {input_file} -> {output_file}")

        # Extract text and split into chunks
        logger.info("Extracting text from file...")
        chunks = extract_and_split(
            input_file,
            max_chunk_size=self.config.chunk_size
        )
        total_chunks = len(chunks)
        logger.info(f"Extracted {len(chunks)} chunks")

        # Setup progress tracking
        tracker = ProgressTracker(
            str(output_file),
            str(self.work_dir),
            total_chunks=total_chunks
        )

        if resume and not tracker.is_complete():
            logger.info(f"Resuming from {tracker.get_completed()}/{total_chunks} completed")

        # Check if already complete
        if tracker.is_complete() and output_file.exists():
            logger.info("File already converted, skipping")
            return str(output_file)

        # Synthesize audio for each chunk
        audio_files = []
        async with CosyVoiceClient(
            api_key=self.config.aliyun_api_key,
            api_endpoint=self.config.aliyun_api_endpoint,
            voice=self.config.tts_voice,
            retry_attempts=self.config.retry_attempts,
            retry_delay=self.config.retry_delay
        ) as client:
            # Process chunks concurrently with semaphore
            semaphore = asyncio.Semaphore(self.config.max_concurrent)

            tasks = []
            for i, chunk in enumerate(chunks):
                if resume and tracker.is_chunk_completed(i):
                    logger.debug(f"Skipping already completed chunk {i}")
                    continue

                task = self._synthesize_chunk(
                    client,
                    chunk,
                    i,
                    self.work_dir,
                    semaphore,
                    tracker
                )
                tasks.append(task)

            # Wait for all synthesis tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect successful audio files
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed: {result}")
                elif result:
                    audio_files.append(result)

        # Merge all audio files
        if audio_files:
            logger.info(f"Merging {len(audio_files)} audio segments...")
            # Sort by index to maintain order
            audio_files.sort()
            self.audio_processor.merge_audio_files(
                audio_files,
                str(output_file),
                silence_duration=2000  # 2 seconds between sentences
            )

            # Cleanup progress
            tracker.cleanup()
            logger.info(f"Conversion complete: {output_file}")
        else:
            logger.warning("No audio files generated")

        return str(output_file)

    async def _synthesize_chunk(
        self,
        client: CosyVoiceClient,
        text: str,
        index: int,
        work_dir: Path,
        semaphore: asyncio.Semaphore,
        tracker: ProgressTracker
    ) -> Optional[str]:
        """Synthesize a single chunk with semaphore control."""
        async with semaphore:
            try:
                logger.info(f"Synthesizing chunk {index}: {text[:50]}...")
                audio_data = await client.synthesize(text)

                # Save to temp file
                temp_file = work_dir / f"chunk_{index:04d}.mp3"
                self.audio_processor.save_audio_segment(audio_data, str(temp_file))

                # Update progress
                tracker.update(index)

                return str(temp_file)

            except TTSRequestError as e:
                logger.error(f"Failed to synthesize chunk {index}: {e}")
                raise

            except Exception as e:
                logger.error(f"Unexpected error synthesizing chunk {index}: {e}")
                raise

    async def convert_batch(
        self,
        input_files: List[str],
        output_dir: Optional[str] = None,
        resume: bool = True
    ) -> List[str]:
        """
        Convert multiple files to audio.

        Args:
            input_files: List of input file paths
            output_dir: Directory for output files (optional)
            resume: Whether to resume from previous progress

        Returns:
            List of generated audio file paths
        """
        if output_dir:
            self.work_dir = Path(output_dir)
            self.work_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for input_file in input_files:
            try:
                output = await self.convert_file(input_file, resume=resume)
                results.append(output)
            except Exception as e:
                logger.error(f"Failed to convert {input_file}: {e}")
                # Continue with next file

        return results
