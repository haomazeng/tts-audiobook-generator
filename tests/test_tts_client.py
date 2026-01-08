import pytest
from unittest.mock import patch, AsyncMock
from src.tts_client import CosyVoiceClient, TTSRequestError

@pytest.mark.asyncio
async def test_tts_client_initialization():
    client = CosyVoiceClient(api_key="test_key")
    assert client.api_key == "test_key"

@pytest.mark.asyncio
async def test_synthesize_single_chunk():
    client = CosyVoiceClient(api_key="test_key")

    # Valid base64 encoded data (this decodes to "test")
    mock_response = {
        "output": {
            "audio": "dGVzdA=="
        }
    }

    with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_response
        result = await client.synthesize("测试文本")
        assert result is not None
        assert result == b"test"

@pytest.mark.asyncio
async def test_synthesize_with_retry():
    client = CosyVoiceClient(api_key="test_key", retry_attempts=2)

    with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_req:
        mock_req.side_effect = [Exception("Network error"), {"output": {"audio": "data"}}]
        result = await client.synthesize("测试文本")
        assert result is not None
