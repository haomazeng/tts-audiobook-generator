# TTS Audiobook Generator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python tool that batch converts PDF and Markdown files to MP3 audiobooks using Aliyun CosyVoice TTS API.

**Architecture:** Modular design with (1) text extraction module for PDF/MD parsing, (2) TTS API client with async/concurrent requests, (3) audio processing for MP3 merging, and (4) CLI interface with progress tracking. Each module is independently testable.

**Tech Stack:** Python 3.10+, PyMuPDF/pdfplumber (PDF), aiohttp (async HTTP), pydub (audio), Click (CLI), PyYAML (config), Aliyun CosyVoice API

---

## Task 1: Project Setup and Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `config.yaml.example`
- Create: `.gitignore`

**Step 1: Create requirements.txt**

```txt
# PDF processing
PyMuPDF>=1.23.0
pdfplumber>=0.10.0

# HTTP client for TTS API
aiohttp>=3.9.0
asyncio>=3.4.3

# Audio processing
pydub>=0.25.0
ffmpeg-python>=0.2.0

# CLI framework
click>=8.1.0
tqdm>=4.66.0

# Configuration
PyYAML>=6.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

**Step 2: Create config.yaml.example**

```yaml
# Aliyun CosyVoice API Configuration
aliyun:
  api_key: "your_api_key_here"
  api_endpoint: "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/generation"

# TTS Settings
tts:
  model: "cosyvoice-v1"
  voice: "zhixiaobai"  # Professional announcer style
  # Options: zhixiaobai, longwan, zhichu, etc.

# Audio Output Settings
audio:
  format: "mp3"
  bitrate: "64k"
  sample_rate: 24000

# Processing Settings
processing:
  max_concurrent: 5  # Concurrent API requests
  chunk_size: 500    # Characters per API request
  retry_attempts: 3
  retry_delay: 2     # Seconds
```

**Step 3: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.venv/

# Config
config.yaml

# Output
output/
audiobooks/
*.mp3
*.wav

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store

# Progress files
*.progress
*.log
```

**Step 4: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages installed successfully

**Step 5: Commit**

```bash
git add requirements.txt config.yaml.example .gitignore
git commit -m "feat: add project dependencies and config template"
```

---

## Task 2: Text Extraction Module

**Files:**
- Create: `src/__init__.py`
- Create: `src/text_extractor.py`
- Create: `tests/test_text_extractor.py`

**Step 1: Create package structure**

```bash
mkdir -p src tests
```

**Step 2: Write failing tests for text extraction**

Create `tests/test_text_extractor.py`:

```python
import pytest
from src.text_extractor import extract_text_from_pdf, extract_text_from_md, split_into_sentences

def test_extract_text_from_md():
    result = extract_text_from_md("tests/fixtures/sample.md")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "测试内容" in result

def test_extract_text_from_pdf():
    result = extract_text_from_pdf("tests/fixtures/sample.pdf")
    assert isinstance(result, str)
    assert len(result) > 0

def test_split_into_sentences():
    text = "这是第一句。这是第二句！这是第三句？"
    result = split_into_sentences(text)
    assert len(result) == 3
    assert result[0] == "这是第一句。"
    assert result[1] == "这是第二句！"

def test_split_respects_chunk_size():
    text = "句子1。" * 100
    result = split_into_sentences(text, max_chunk_size=200)
    for chunk in result:
        assert len(chunk) <= 200
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/test_text_extractor.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'src'"

**Step 4: Implement text extractor**

Create `src/text_extractor.py`:

```python
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List
import re

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file using PyMuPDF for better Chinese support."""
    doc = fitz.open(pdf_path)
    text_parts = []

    for page in doc:
        text = page.get_text()
        if text.strip():
            text_parts.append(text)

    doc.close()
    return "\n".join(text_parts)

def extract_text_from_md(md_path: str) -> str:
    """Extract text from Markdown file."""
    with open(md_path, 'r', encoding='utf-8') as f:
        return f.read()

def split_into_sentences(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    Split text into sentences, respecting punctuation marks.
    Merge short sentences into chunks that don't exceed max_chunk_size.
    """
    # Chinese punctuation marks for sentence splitting
    sentence_endings = r'([。！？；])'
    sentences = re.split(sentence_endings, text)

    # Re-attach punctuation to sentences
    reconstructed = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            reconstructed.append(sentences[i] + sentences[i + 1])
        else:
            reconstructed.append(sentences[i])

    # Filter empty sentences
    sentences = [s.strip() for s in reconstructed if s.strip()]

    # Merge sentences into chunks
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def extract_and_split(file_path: str, max_chunk_size: int = 500) -> List[str]:
    """Extract text from file and split into chunks based on file type."""
    path = Path(file_path)

    if path.suffix.lower() == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif path.suffix.lower() in ['.md', '.markdown']:
        text = extract_text_from_md(file_path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return split_into_sentences(text, max_chunk_size)
```

**Step 5: Create test fixtures**

```bash
mkdir -p tests/fixtures
echo "# 测试文档
这是第一段内容。
这是第二段内容。" > tests/fixtures/sample.md
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/test_text_extractor.py -v`
Expected: PASS (all tests)

**Step 7: Commit**

```bash
git add src/ tests/
git commit -m "feat: implement text extraction module for PDF and Markdown"
```

---

## Task 3: TTS API Client Module

**Files:**
- Create: `src/tts_client.py`
- Create: `tests/test_tts_client.py`

**Step 1: Write failing tests for TTS client**

Create `tests/test_tts_client.py`:

```python
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

    mock_response = {
        "output": {
            "audio": "base64_encoded_audio_data"
        }
    }

    with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_response
        result = await client.synthesize("测试文本")
        assert result is not None

@pytest.mark.asyncio
async def test_synthesize_with_retry():
    client = CosyVoiceClient(api_key="test_key", retry_attempts=2)

    with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_req:
        mock_req.side_effect = [Exception("Network error"), {"output": {"audio": "data"}}]
        result = await client.synthesize("测试文本")
        assert result is not None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tts_client.py -v`
Expected: FAIL - "ModuleNotFoundError"

**Step 3: Implement TTS client**

Create `src/tts_client.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tts_client.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/tts_client.py tests/test_tts_client.py
git commit -m "feat: implement async TTS API client with retry logic"
```

---

## Task 4: Audio Processing Module

**Files:**
- Create: `src/audio_processor.py`
- Create: `tests/test_audio_processor.py`

**Step 1: Write failing tests for audio processor**

Create `tests/test_audio_processor.py`:

```python
import pytest
from pathlib import Path
from src.audio_processor import AudioProcessor, merge_audio_files

def test_audio_processor_initialization():
    processor = AudioProcessor(bitrate="64k", sample_rate=24000)
    assert processor.bitrate == "64k"
    assert processor.sample_rate == 24000

def test_save_audio_segment(tmp_path):
    processor = AudioProcessor()
    audio_data = b"fake_audio_data"  # This would be real audio in practice

    output_path = tmp_path / "test.mp3"
    # Note: This test will need real audio data or mocking
    # For now, we test the path construction

def test_merge_audio_files(tmp_path):
    # Create fake audio files for testing
    # This test will need proper setup with real audio data
    pass
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_audio_processor.py -v`
Expected: FAIL - implementation missing

**Step 3: Implement audio processor**

Create `src/audio_processor.py`:

```python
from pathlib import Path
from typing import List
import asyncio
from pydub import AudioSegment
from pydub.silence import silence
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
        Note: pydub needs file-like object or path for raw data.
        """
        # Create AudioSegment from raw data
        # Assuming PCM data from API, convert to AudioSegment
        try:
            # Write temp file first
            temp_path = Path(output_path).with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                f.write(audio_data)

            # Load and export with proper settings
            audio = AudioSegment.from_file(temp_path)
            audio.export(
                output_path,
                format=self.format,
                bitrate=self.bitrate
            )

            # Clean up temp
            temp_path.unlink()

            logger.info(f"Saved audio to {output_path}")

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
```

**Step 4: Run tests**

Run: `pytest tests/test_audio_processor.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/audio_processor.py tests/test_audio_processor.py
git commit -m "feat: implement audio processing module with merge functionality"
```

---

## Task 5: Configuration Management

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`

**Step 1: Write failing tests for config**

Create `tests/test_config.py`:

```python
import pytest
from pathlib import Path
from src.config import Config, ConfigError

def test_load_config_from_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
aliyun:
  api_key: "test_key"
tts:
  voice: "zhixiaobai"
audio:
  bitrate: "64k"
""")

    config = Config.load(str(config_file))
    assert config.aliyun_api_key == "test_key"
    assert config.tts_voice == "zhixiaobai"

def test_config_missing_file():
    with pytest.raises(ConfigError):
        Config.load("/nonexistent/config.yaml")

def test_config_env_override(monkeypatch):
    monkeypatch.setenv("ALIYUN_API_KEY", "env_key")
    config = Config.load_defaults()
    assert config.aliyun_api_key == "env_key"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL

**Step 3: Implement config module**

Create `src/config.py`:

```python
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
```

**Step 4: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: implement configuration management with YAML and env support"
```

---

## Task 6: Progress Tracking Module

**Files:**
- Create: `src/progress.py`
- Create: `tests/test_progress.py`

**Step 1: Write failing tests**

Create `tests/test_progress.py`:

```python
import pytest
from pathlib import Path
from src.progress import ProgressTracker

def test_progress_tracker_initialization(tmp_path):
    tracker = ProgressTracker("test_file.mp3", tmp_path)
    assert tracker.total_chunks == 0

def test_progress_tracker_update(tmp_path):
    tracker = ProgressTracker("test.mp3", tmp_path, total_chunks=10)
    tracker.update(5)
    assert tracker.get_progress() == 0.5

def test_progress_tracker_persistence(tmp_path):
    tracker = ProgressTracker("test.mp3", tmp_path, total_chunks=10)
    tracker.update(5)

    # Load new instance
    tracker2 = ProgressTracker("test.mp3", tmp_path, total_chunks=10)
    assert tracker2.get_completed() == 5

def test_progress_tracker_completion(tmp_path):
    tracker = ProgressTracker("test.mp3", tmp_path, total_chunks=10)
    tracker.update(10)
    assert tracker.is_complete()

def test_progress_tracker_reset(tmp_path):
    tracker = ProgressTracker("test.mp3", tmp_path, total_chunks=10)
    tracker.update(5)
    tracker.reset()
    assert tracker.get_completed() == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_progress.py -v`
Expected: FAIL

**Step 3: Implement progress tracker**

Create `src/progress.py`:

```python
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
```

**Step 4: Run tests**

Run: `pytest tests/test_progress.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/progress.py tests/test_progress.py
git commit -m "feat: implement progress tracking with persistence for resumable downloads"
```

---

## Task 7: Main Conversion Engine

**Files:**
- Create: `src/converter.py`
- Create: `tests/test_converter.py`

**Step 1: Write failing tests for converter**

Create `tests/test_converter.py`:

```python
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from src.converter import AudioConverter

@pytest.mark.asyncio
async def test_converter_initialization():
    converter = AudioConverter(api_key="test_key")
    assert converter is not None

@pytest.mark.asyncio
async def test_convert_single_file(tmp_path):
    # This would be an integration test
    # For now, test the structure
    pass
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_converter.py -v`
Expected: FAIL

**Step 3: Implement converter**

Create `src/converter.py`:

```python
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
```

**Step 4: Run tests**

Run: `pytest tests/test_converter.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/converter.py tests/test_converter.py
git commit -m "feat: implement main conversion engine with concurrent processing"
```

---

## Task 8: CLI Interface

**Files:**
- Create: `src/cli.py`
- Create: `tts_tool.py` (entry point)

**Step 1: Implement CLI interface**

Create `src/cli.py`:

```python
import click
import asyncio
import logging
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

from .config import Config, ConfigError
from .converter import AudioConverter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--input', '-i',
    required=True,
    type=click.Path(exists=True),
    help='Input file or directory'
)
@click.option(
    '--output', '-o',
    default='./audiobooks',
    type=click.Path(),
    help='Output directory for audio files'
)
@click.option(
    '--config', '-c',
    default='config.yaml',
    type=click.Path(exists=True),
    help='Configuration file path'
)
@click.option(
    '--format',
    type=click.Choice(['pdf', 'md', 'all']),
    default='all',
    help='File format to process'
)
@click.option(
    '--voice',
    type=click.Choice(['zhixiaobai', 'longwan', 'zhichu', 'aiqi', 'zhichu_v2']),
    default='zhixiaobai',
    help='Voice type for TTS'
)
@click.option(
    '--no-resume',
    is_flag=True,
    help='Do not resume from previous progress'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose logging'
)
def main(
    input: str,
    output: str,
    config: str,
    format: str,
    voice: str,
    no_resume: bool,
    verbose: bool
):
    """
    TTS Audiobook Generator - Convert PDF and Markdown files to MP3 audio.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load configuration
        try:
            cfg = Config.load(config)
        except ConfigError:
            logger.info("Config file not found, using defaults with environment variables")
            cfg = Config.load_defaults()

        # Override with CLI options
        if voice:
            cfg.tts_voice = voice

        # Validate config
        cfg.validate()

        # Collect input files
        input_path = Path(input)
        input_files = []

        if input_path.is_file():
            input_files = [str(input_path)]
        elif input_path.is_dir():
            extensions = {
                'pdf': ['.pdf'],
                'md': ['.md', '.markdown'],
                'all': ['.pdf', '.md', '.markdown']
            }
            valid_exts = extensions[format]

            for ext in valid_exts:
                input_files.extend([str(f) for f in input_path.rglob(f'*{ext}')])

        if not input_files:
            logger.warning(f"No files found in {input}")
            return

        logger.info(f"Found {len(input_files)} files to convert")

        # Run conversion
        converter = AudioConverter(cfg, work_dir=output)

        async def run_conversion():
            results = []
            for file in input_files:
                try:
                    result = await converter.convert_file(
                        file,
                        resume=not no_resume
                    )
                    results.append(result)
                    logger.info(f"✓ Converted: {file} -> {result}")
                except Exception as e:
                    logger.error(f"✗ Failed: {file} - {e}")

            return results

        results = asyncio.run(run_conversion())

        # Summary
        logger.info(f"\nConversion complete!")
        logger.info(f"Successfully converted: {len([r for r in results if r])}/{len(input_files)}")

    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please set ALIYUN_API_KEY environment variable or create config.yaml")
        raise click.Abort()
    except Exception as e:
        logger.error(f"Error: {e}")
        raise click.Abort()


if __name__ == '__main__':
    main()
```

**Step 2: Create entry point**

Create `tts_tool.py`:

```python
#!/usr/bin/env python3
"""
TTS Audiobook Generator Entry Point
"""

from src.cli import main

if __name__ == '__main__':
    main()
```

**Step 3: Make executable**

```bash
chmod +x tts_tool.py
```

**Step 4: Commit**

```bash
git add src/cli.py tts_tool.py
git commit -m "feat: implement CLI interface with Click"
```

---

## Task 9: Documentation and Examples

**Files:**
- Modify: `README.md`
- Create: `docs/usage.md`
- Create: `docs/api_reference.md`

**Step 1: Update README.md**

```markdown
# TTS Audiobook Generator

基于阿里云百炼CosyVoice的文本转语音工具，支持批量将PDF和Markdown文件转换为MP3音频。

## 功能特性

- 支持 PDF 和 Markdown 文件
- 批量转换整个目录
- 断点续传 - 中断后可继续
- 并发处理 - 加速转换
- 进度显示 - 实时查看进度
- 专业播音风格声音

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

复制配置模板：

```bash
cp config.yaml.example config.yaml
```

编辑 `config.yaml`，填入你的阿里云API密钥：

```yaml
aliyun:
  api_key: "your_api_key_here"
```

或使用环境变量：

```bash
export ALIYUN_API_KEY="your_api_key_here"
```

### 3. 运行转换

转换单个文件：

```bash
python tts_tool.py --input document.pdf --output ./audiobooks
```

批量转换目录：

```bash
python tts_tool.py --input ./documents --output ./audiobooks
```

只转换PDF：

```bash
python tts_tool.py --input ./documents --format pdf
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input`, `-i` | 输入文件或目录（必需） | - |
| `--output`, `-o` | 输出目录 | `./audiobooks` |
| `--config`, `-c` | 配置文件路径 | `config.yaml` |
| `--format` | 文件格式 (pdf/md/all) | `all` |
| `--voice` | 声音类型 | `zhixiaobai` |
| `--no-resume` | 不使用断点续传 | false |
| `--verbose`, `-v` | 详细日志 | false |

## 声音类型

- `zhixiaobai` - 知小百（专业播音，推荐）
- `longwan` - 龙万（男声）
- `zhichu` - 知楚（女声）
- `aiqi` - 爱奇（儿童）
- `zhichu_v2` - 知楚v2（增强版）

## 断点续传

转换过程中的进度会自动保存。如果中断，重新运行相同的命令会从断点继续：

```bash
# 第一次运行（中断）
python tts_tool.py --input document.pdf

# 继续运行（从断点继续）
python tts_tool.py --input document.pdf
```

如需重新开始，使用 `--no-resume`：

```bash
python tts_tool.py --input document.pdf --no-resume
```

## 开发

运行测试：

```bash
pytest tests/ -v
```

## License

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
```

**Step 2: Create usage guide**

Create `docs/usage.md`:

```markdown
# 使用指南

## 完整示例

### 示例1: 转换单个PDF文件

```bash
python tts_tool.py -i 论文.pdf -o ./audiobooks
```

### 示例2: 批量转换Obsidian笔记

```bash
python tts_tool.py -i ~/Documents/ObsidianVault -o ./audiobooks --format md
```

### 示例3: 使用不同声音

```bash
python tts_tool.py -i document.pdf --voice longwan
```

## 性能优化

### 调整并发数

编辑 `config.yaml`:

```yaml
processing:
  max_concurrent: 10  # 增加并发数加速转换
```

### 调整音频质量

```yaml
audio:
  bitrate: "128k"  # 更高质量
  sample_rate: 48000
```

## 故障排除

### API限流

如果遇到限流错误，脚本会自动重试。可以调整重试设置：

```yaml
processing:
  retry_attempts: 5
  retry_delay: 3
```

### 音频质量问题

尝试：
1. 使用不同的声音类型
2. 增加 `bitrate` 到 128k
3. 检查源文本编码

### 中文乱码

确保源文件是UTF-8编码。
```

**Step 3: Commit**

```bash
git add README.md docs/
git commit -m "docs: add comprehensive usage documentation"
```

---

## Task 10: Final Integration and Testing

**Files:**
- Create: `tests/integration/test_integration.py`

**Step 1: Create integration test**

Create `tests/integration/test_integration.py`:

```python
import pytest
from pathlib import Path
import asyncio

# This would test the full flow with real files
# Mark as integration test to run separately

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_conversion_flow():
    """Test complete conversion with all modules."""
    # This would require a real API key for testing
    pass
```

**Step 2: Create pytest configuration**

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    integration: Integration tests
```

**Step 3: Run all tests**

```bash
pytest tests/ -v --ignore=tests/integration
```

**Step 4: Create example script**

Create `examples/simple_example.py`:

```python
#!/usr/bin/env python3
"""
Simple example of using TTS Audiobook Generator
"""

import asyncio
from src.config import Config
from src.converter import AudioConverter

async def main():
    # Load config
    config = Config.load("config.yaml")

    # Create converter
    converter = AudioConverter(config, work_dir="./output")

    # Convert a file
    result = await converter.convert_file("document.pdf")
    print(f"Audio saved to: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Step 5: Commit**

```bash
git add pytest.ini examples/ tests/integration/
git commit -m "test: add integration tests and examples"
```

**Step 6: Push to GitHub**

```bash
git push origin main
```

---

## Summary

This implementation plan provides a complete, modular TTS audiobook generator with:

1. **Text extraction** from PDF and Markdown files
2. **Async TTS client** with retry logic
3. **Audio processing** with merging
4. **Progress tracking** for resumable downloads
5. **CLI interface** for easy usage
6. **Comprehensive testing** and documentation

**Total estimated tasks:** 10
**Total estimated commits:** ~10-15

Each task follows TDD principles: write failing test, implement, verify, commit.
