import aiohttp
import asyncio
import base64
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSRequestError(Exception):
    """Custom exception for TTS API errors."""
    pass


class CosyVoiceClient:
    """
    Async client for Aliyun CosyVoice TTS API.
    """

    def __init__(
        self,
        api_key: str,
        api_endpoint: str = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/generation",
        model: str = "cosyvoice-v1",
        voice: str = "zhixiaobai",
        retry_attempts: int = 3,
        retry_delay: float = 2.0
    ):
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.model = model
        self.voice = voice
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _build_payload(self, text: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "input": {
                "text": text
            },
            "parameters": {
                "text_type": "PlainText",
                "voice": self.voice
            }
        }

    async def _make_request(self, text: str) -> Dict[str, Any]:
        """Make a single API request."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        headers = self._get_headers()
        payload = self._build_payload(text)

        logger.info(f"Sending TTS request for text length: {len(text)}")

        async with self.session.post(
            self.api_endpoint,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                logger.warning(f"Rate limited. Retrying after {retry_after}s")
                await asyncio.sleep(retry_after)
                raise TTSRequestError("Rate limited, retry required")

            if response.status != 200:
                error_text = await response.text()
                raise TTSRequestError(f"API error {response.status}: {error_text}")

            return await response.json()

    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to audio with retry logic.
        Returns audio data as bytes.
        """
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                response = await self._make_request(text)

                # Extract base64 audio data
                if "output" not in response or "audio" not in response["output"]:
                    raise TTSRequestError(f"Invalid response format: {response}")

                audio_base64 = response["output"]["audio"]
                audio_data = base64.b64decode(audio_base64)

                logger.info(f"Successfully synthesized {len(text)} chars -> {len(audio_data)} bytes")
                return audio_data

            except TTSRequestError as e:
                last_error = e
                if "Rate limited" in str(e) and attempt < self.retry_attempts - 1:
                    # Exponential backoff
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
