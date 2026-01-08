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
