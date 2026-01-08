import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from src.converter import AudioConverter

@pytest.mark.asyncio
async def test_converter_initialization():
    from src.config import Config
    config = Config.load_defaults()
    config.aliyun_api_key = "test_key"
    converter = AudioConverter(config)
    assert converter is not None
