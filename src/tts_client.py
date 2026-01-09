"""
TTS Client using Aliyun DashScope SDK for CosyVoice.
"""
import asyncio
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat

logger = logging.getLogger(__name__)


class TTSRequestError(Exception):
    """Custom exception for TTS API errors."""
    pass


class CosyVoiceClient:
    """
    Async client for Aliyun CosyVoice TTS API using DashScope SDK.
    """

    # Model to voice mapping
    VOICE_MAP_V1 = {
        "zhixiaobai": "longxiaobai",
        "longwan": "longwan",
        "zhichu": "longxiaochun",
        "longcheng": "longcheng",
        "longhua": "longhua",
        "longshu": "longshu",
    }

    # Voice options that support v1 and v2
    VOICE_MAP_V2 = {
        "zhixiaobai": "longxiaobai_v2",
        "longwan": "longwan_v2",
        "zhichu": "longxiaochun_v2",
        "longcheng": "longcheng_v2",
        "longhua": "longhua_v2",
        "longshu": "longshu_v2",
        "longxiaobai": "longxiaobai_v2",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "cosyvoice-v1",
        voice: str = "zhixiaobai",
        retry_attempts: int = 3,
        retry_delay: float = 2.0
    ):
        self.api_key = api_key
        self.model = model
        self.voice = self._map_voice(voice, model)
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Set API key for DashScope
        dashscope.api_key = api_key

    def _map_voice(self, voice: str, model: str) -> str:
        """Map legacy voice names to actual voice IDs based on model version."""
        if model == "cosyvoice-v2":
            return self.VOICE_MAP_V2.get(voice, voice)
        else:
            return self.VOICE_MAP_V1.get(voice, voice)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)

    def _synthesize_sync(self, text: str) -> bytes:
        """
        Synchronous synthesis using DashScope SDK.
        Must be run in a thread pool to avoid blocking.
        """
        try:
            logger.info(f"DashScope: Synthesizing {len(text)} chars with {self.model}/{self.voice}")

            # Map audio format based on model - all models require explicit format
            if self.model == "cosyvoice-v2":
                audio_format = AudioFormat.MP3_24000HZ_MONO_256KBPS
            else:
                # v1 also needs explicit format
                audio_format = AudioFormat.MP3_24000HZ_MONO_256KBPS

            # Create synthesizer
            synthesizer = SpeechSynthesizer(
                model=self.model,
                voice=self.voice,
                format=audio_format
            )

            # Synthesize
            audio = synthesizer.call(text)

            logger.info(f"DashScope: Successfully synthesized {len(audio)} bytes")
            return audio

        except Exception as e:
            logger.error(f"DashScope synthesis failed: {e}")
            raise TTSRequestError(f"DashScope error: {e}")

    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to audio with retry logic.
        Returns audio data as bytes.
        """
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                # Run blocking SDK call in thread pool
                loop = asyncio.get_event_loop()
                audio_data = await loop.run_in_executor(
                    self.executor,
                    self._synthesize_sync,
                    text
                )

                return audio_data

            except TTSRequestError as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retry {attempt + 1}/{self.retry_attempts} after {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    raise

            except Exception as e:
                last_error = e
                logger.error(f"Request failed: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)

        raise TTSRequestError(f"Failed after {self.retry_attempts} attempts: {last_error}")
