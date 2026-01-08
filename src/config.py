import os
import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

class ConfigError(Exception):
    """Configuration error."""
    pass


@dataclass
class Config:
    """Application configuration."""

    # Aliyun settings
    aliyun_api_key: str
    aliyun_api_endpoint: str = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/generation"

    # TTS settings
    tts_model: str = "cosyvoice-v1"
    tts_voice: str = "zhixiaobai"

    # Audio settings
    audio_format: str = "mp3"
    audio_bitrate: str = "64k"
    audio_sample_rate: int = 24000

    # Processing settings
    max_concurrent: int = 5
    chunk_size: int = 500
    retry_attempts: int = 3
    retry_delay: float = 2.0

    @classmethod
    def load(cls, config_path: str) -> 'Config':
        """Load config from YAML file."""
        path = Path(config_path)

        if not path.exists():
            raise ConfigError(f"Config file not found: {config_path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Extract values with environment variable override
        aliyun = data.get('aliyun', {})
        tts = data.get('tts', {})
        audio = data.get('audio', {})
        processing = data.get('processing', {})

        return cls(
            aliyun_api_key=os.getenv('ALIYUN_API_KEY', aliyun.get('api_key')),
            aliyun_api_endpoint=aliyun.get('api_endpoint', cls.aliyun_api_endpoint),
            tts_model=tts.get('model', cls.tts_model),
            tts_voice=tts.get('voice', cls.tts_voice),
            audio_format=audio.get('format', cls.audio_format),
            audio_bitrate=audio.get('bitrate', cls.audio_bitrate),
            audio_sample_rate=audio.get('sample_rate', cls.audio_sample_rate),
            max_concurrent=processing.get('max_concurrent', cls.max_concurrent),
            chunk_size=processing.get('chunk_size', cls.chunk_size),
            retry_attempts=processing.get('retry_attempts', cls.retry_attempts),
            retry_delay=processing.get('retry_delay', cls.retry_delay)
        )

    @classmethod
    def load_defaults(cls) -> 'Config':
        """Load config with defaults, environment variables can override."""
        return cls(
            aliyun_api_key=os.getenv('ALIYUN_API_KEY', ''),
            aliyun_api_endpoint=os.getenv('ALIYUN_API_ENDPOINT', cls.aliyun_api_endpoint),
            tts_voice=os.getenv('TTS_VOICE', cls.tts_voice),
            audio_bitrate=os.getenv('AUDIO_BITRATE', cls.audio_bitrate)
        )

    def validate(self) -> None:
        """Validate required configuration."""
        if not self.aliyun_api_key:
            raise ConfigError("Aliyun API key is required. Set ALIYUN_API_KEY or configure in config.yaml")
