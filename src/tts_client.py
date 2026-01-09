"""
TTS Client using Aliyun DashScope Qwen-TTS-Realtime.
This provides larger free tier quota.
"""
import asyncio
import base64
import threading
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import dashscope
from dashscope.audio.qwen_tts_realtime import QwenTtsRealtime, QwenTtsRealtimeCallback, AudioFormat

logger = logging.getLogger(__name__)


class TTSRequestError(Exception):
    """Custom exception for TTS API errors."""
    pass


class QwenTTSRealtimeClient:
    """
    Async client for Aliyun Qwen-TTS-Realtime using DashScope SDK.
    Uses WebSocket for streaming audio synthesis.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-tts-flash-realtime",
        voice: str = "Cherry",
        sample_rate: int = 24000,
        retry_attempts: int = 3,
        retry_delay: float = 2.0
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.sample_rate = sample_rate
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Set API key for DashScope
        dashscope.api_key = api_key

        # Audio data storage
        self.audio_chunks = []
        self.complete_event = threading.Event()
        self.qwen_tts = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.qwen_tts:
            try:
                self.qwen_tts.close()
            except:
                pass
        self.executor.shutdown(wait=True)

    class _Callback(QwenTtsRealtimeCallback):
        """Internal callback class for Qwen TTS Realtime."""

        def __init__(self, parent):
            super().__init__()
            self.parent = parent

        def on_open(self) -> None:
            logger.info("WebSocket connection opened")

        def on_close(self, close_status_code, close_msg) -> None:
            logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")

        def on_event(self, response: dict) -> None:
            try:
                event_type = response.get('type', '')

                if event_type == 'session.created':
                    logger.info(f"Session created: {response['session']['id']}")

                elif event_type == 'response.audio.delta':
                    # Decode base64 audio and store
                    audio_b64 = response.get('delta', '')
                    audio_data = base64.b64decode(audio_b64)
                    self.parent.audio_chunks.append(audio_data)
                    logger.debug(f"Received audio chunk: {len(audio_data)} bytes")

                elif event_type == 'response.audio.done':
                    logger.info("Audio generation complete")

                elif event_type == 'response.done':
                    logger.info(f"Response done: {self.parent.qwen_tts.get_last_response_id() if self.parent.qwen_tts else 'N/A'}")

                elif event_type == 'session.finished':
                    logger.info("Session finished")
                    self.parent.complete_event.set()

            except Exception as e:
                logger.error(f"Error processing event: {e}")

    def _synthesize_sync(self, text: str) -> bytes:
        """
        Synchronous synthesis using QwenTtsRealtime.
        Must be run in a thread pool to avoid blocking.
        """
        try:
            logger.info(f"Qwen-TTS: Synthesizing {len(text)} chars with {self.model}/{self.voice}")

            # Clear previous audio data
            self.audio_chunks.clear()
            self.complete_event.clear()

            # Create callback
            callback = self._Callback(self)

            # Create TTS instance
            self.qwen_tts = QwenTtsRealtime(
                model=self.model,
                callback=callback,
                url='wss://dashscope.aliyuncs.com/api-ws/v1/realtime'
            )

            # Connect
            self.qwen_tts.connect()

            # Configure session
            self.qwen_tts.update_session(
                voice=self.voice,
                response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                mode='commit'  # Use commit mode for batch processing
            )

            # Send text
            self.qwen_tts.append_text(text)
            self.qwen_tts.commit()

            # Wait for completion
            self.complete_event.wait(timeout=120)  # 2 minute timeout
            self.qwen_tts.finish()

            # Combine all audio chunks
            audio_data = b''.join(self.audio_chunks)

            logger.info(f"Qwen-TTS: Successfully synthesized {len(audio_data)} bytes")
            return audio_data

        except Exception as e:
            logger.error(f"Qwen-TTS synthesis failed: {e}")
            raise TTSRequestError(f"Qwen-TTS error: {e}")

    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to audio with retry logic.
        Returns audio data as bytes (PCM format).
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


# Keep backward compatibility
CosyVoiceClient = QwenTTSRealtimeClient
