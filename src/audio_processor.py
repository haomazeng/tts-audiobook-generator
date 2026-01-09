from pathlib import Path
from typing import List
import asyncio
from pydub import AudioSegment
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Handle audio processing: save segments, merge files, add silence.
    """

    def __init__(
        self,
        format: str = "mp3",
        bitrate: str = "64k",
        sample_rate: int = 24000
    ):
        self.format = format
        self.bitrate = bitrate
        self.sample_rate = sample_rate

    def save_audio_segment(self, audio_data: bytes, output_path: str) -> None:
        """
        Save raw audio data to file.
        Supports PCM raw data from Qwen-TTS-Realtime.
        """
        try:
            # Qwen-TTS-Realtime returns raw PCM data
            # PCM_24000HZ_MONO_16BIT: 24000Hz, mono, 16-bit (sample_width=2)
            audio = AudioSegment(
                data=audio_data,
                sample_width=2,  # 16-bit
                frame_rate=self.sample_rate,  # 24000 Hz
                channels=1  # mono
            )

            # Export to target format (MP3)
            audio.export(
                output_path,
                format=self.format,
                bitrate=self.bitrate
            )

            logger.info(f"Saved audio to {output_path} ({len(audio_data)} bytes)")

        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise

    def merge_audio_files(
        self,
        audio_files: List[str],
        output_path: str,
        silence_duration: int = 1000
    ) -> None:
        """
        Merge multiple audio files with silence between segments.
        silence_duration: ms of silence between segments (1000ms = 1s for sentences)
        """
        if not audio_files:
            logger.warning("No audio files to merge")
            return

        logger.info(f"Merging {len(audio_files)} audio files")

        combined = AudioSegment.empty()

        for i, audio_file in enumerate(audio_files):
            try:
                audio = AudioSegment.from_file(audio_file)
                combined += audio

                # Add silence between segments (not after last one)
                if i < len(audio_files) - 1:
                    combined += AudioSegment.silent(duration=silence_duration)

                logger.debug(f"Added {audio_file} ({len(audio)}ms)")

            except Exception as e:
                logger.error(f"Failed to process {audio_file}: {e}")
                raise

        # Export final merged audio
        combined.export(
            output_path,
            format=self.format,
            bitrate=self.bitrate
        )

        logger.info(f"Merged audio saved to {output_path} (total: {len(combined)}ms)")

        # Clean up individual files
        for audio_file in audio_files:
            Path(audio_file).unlink()

    def create_podcast_style_output(
        self,
        audio_files: List[str],
        output_path: str
    ) -> None:
        """
        Create podcast-style output with proper pauses:
        - 2 seconds (2000ms) between sentences
        - 3 seconds (3000ms) between paragraphs
        """
        self.merge_audio_files(
            audio_files,
            output_path,
            silence_duration=2000
        )


def merge_audio_files(
    audio_files: List[str],
    output_path: str,
    silence_duration: int = 1000
) -> None:
    """Convenience function for merging audio files."""
    processor = AudioProcessor()
    processor.merge_audio_files(audio_files, output_path, silence_duration)
