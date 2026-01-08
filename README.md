# TTS Audiobook Generator

基于阿里云百炼CosyVoice的文本转语音工具，支持批量将PDF和Markdown文件转换为MP3音频。

## 功能特性

- 支持PDF和Markdown文件
- 批量转换
- 断点续传
- 进度显示
- 专业播音风格

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
python tts_tool.py --input ./ --output ./audiobooks/
```

## 配置

创建 `config.yaml` 文件并配置你的阿里云API密钥。

## License

MIT
